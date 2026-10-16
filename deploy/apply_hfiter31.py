#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第31轮：自适应高频改成**迭代收敛**（幂等）

## 问题
现役只做一次反算：
```python
hf_db = 10*log10(HF_TARGET / b4000_8000(t5))
apply highshelf=f=4000:g=hf_db
```
但 `highshelf` **不是理想砖墙** —— 4000Hz 处只有一半增益，往上才渐近到 g，
所以 4k~8k 这一段的实际变化量 ≠ g dB，一次反算到不了位。

实测（同一条 447s 裸输出，fix_chain 出的 wav）：
| | b4000_8000 | 相对目标 0.0129 |
|---|---|---|
| pw60 | 0.01385 | **+0.31 dB** |
| pw15 | 0.01397 | **+0.35 dB** |
| EL（同文案） | 0.01342 | +0.17 dB |
单看这一项，4k~8k 贡献了 Σz² 27.1 里的 **9.8（36%）**。

## 修法
量 → 补 → 再量，最多 3 轮，单轮残差 <0.25dB 即停：
```
g = adaptive_hf_db(t5)
repeat 3:
    应用 highshelf(g) + MALE_250 + volume → tmp
    r = b4000_8000(tmp)
    d = clip(10*log10(TARGET/r), ...)
    g += d
    if |d| < 0.25: 收工
```
每轮只是多跑一次 ffmpeg（全篇约 10~20s），可接受。

## 风控
- 累计增益仍钳在 `[HF_DB_MIN, HF_DB_MAX] = [-9.5, +6.0]`（历史教训：
  超过 -9.5dB 会把平坦度打到 -6σ 发假）
- 只改 4k 以上的架子高度，**不改时长、不改语速、不动任何别的滤波器**
- 确定性：同一输入 → 同一输出（无随机）

## 用法
    ./venv/bin/python deploy/apply_hfiter31.py
"""
import ast
import sys

P = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"

OLD = '''        hf_db = adaptive_hf_db(t5)
        print("自适应高频：boost 后 b4000_8000=%.5f (目标 %.5f) -> highshelf %+.2f dB"
              % (adaptive_hf_ratio(t5), HF_TARGET, hf_db), flush=True)
        # 🔴 250Hz 衰减放在**自适应之后**：不影响 adaptive_hf_db(t5) 的输入
        af = (("highshelf=f=%d:g=%g," % (HF_FREQ, hf_db)) + MALE_250
             + "volume=%gdB" % gain)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", t5, "-af", af,
                        a.out], check=True)'''

NEW = '''        hf_db = adaptive_hf_db(t5)
        print("自适应高频：boost 后 b4000_8000=%.5f (目标 %.5f) -> highshelf %+.2f dB"
              % (adaptive_hf_ratio(t5), HF_TARGET, hf_db), flush=True)
        # 🔴 250Hz 衰减放在**自适应之后**：不影响 adaptive_hf_db(t5) 的输入
        # 🔴 第31轮：改成**迭代收敛**。highshelf 不是理想砖墙（4000Hz 处只有
        #    一半增益、往上才渐近到 g），所以一次反算到不了位 ——
        #    实测残差 +0.31~+0.35 dB，而这一项占了 Σz² 的 36%。
        #    做法：量 → 补 → 再量，最多 3 轮，单轮残差 <0.25dB 即收工。
        _last = None
        for _it in range(3):
            af = (("highshelf=f=%d:g=%g," % (HF_FREQ, hf_db)) + MALE_250
                  + "volume=%gdB" % gain)
            _t = os.path.join(tmpd, "ad%d.wav" % _it)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", t5,
                            "-af", af, _t], check=True)
            _r = adaptive_hf_ratio(_t)
            _d = 0.0
            if _r > 0:
                _d = float(np.clip(10.0 * np.log10(HF_TARGET / _r),
                                   HF_DB_MIN - hf_db, HF_DB_MAX - hf_db))
            print("   迭代%d: b4000_8000=%.5f -> 补 %+.2f dB（累计 %+.2f）"
                  % (_it, _r, _d, hf_db + _d), flush=True)
            hf_db = float(np.clip(hf_db + _d, HF_DB_MIN, HF_DB_MAX))
            _last = _t
            if abs(_d) < 0.25:
                break
        shutil.move(_last, a.out)'''


def main():
    s = open(P, encoding="utf-8").read()
    if "迭代%d" in s:
        print("已打过补丁，跳过（幂等）")
        return 0
    if OLD not in s:
        print("❌ 未定位到自适应高频块")
        return 1
    s = s.replace(OLD, NEW, 1)
    try:
        ast.parse(s)
    except SyntaxError as e:
        print("❌ 语法错误 %s" % e)
        return 1
    open(P, "w", encoding="utf-8").write(s)
    print("✅ 已写入 %s（语法 OK）" % P)
    return 0


if __name__ == "__main__":
    sys.exit(main())
