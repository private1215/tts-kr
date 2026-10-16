#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第31轮：高频与 250Hz 一起迭代收敛（幂等，替代单次 hf 迭代）

## 为什么
r32（融合 + 15s 变调窗 + 高频迭代）把 Σz² 从 27.1 降到 **23.9**，
4k~8k 从 +3.13σ 收敛到 **+0.65σ**、平坦度从 −2.89σ 到 **+0.30σ**。

但迭代把高频压下来之后，**150~300Hz 反而冒头成最大项**：

| 项 | 对 Σz² 的贡献 | z |
|---|---|---|
| 发声占比 | 11.63 | +3.41σ |
| **150~300** | **6.81** | **+2.61σ** |
| 其余 7 项合计 | 5.46 | — |

原因：`MALE_250` 是**写死的 −0.45dB**（当年按端到端反馈手调出来的），
而总能量会因为高频架子的变化而变化 → 所有频段占比跟着浮动，
固定值必然不是每次都刚好。历史教训也印证了这一点：
「离线归零用 -0.75dB，端到端会过冲到 -2~-3.4σ，-1.5dB 过冲（Σz² 反升）」。

## 修法
把高频和 250Hz **一起迭代**（两者相距很远，互相影响很小，可以并行收敛）：
```
g_hf  = adaptive_hf_db(t5)      # 初值
g_250 = MALE_250_DB             # -0.45，初值沿用历史手调值
repeat 4:
    af = highshelf(4000, g_hf) + equalizer(250, g_250) + volume
    跑 ffmpeg → tmp
    d_hf  = clip(10log10(HF_TARGET / b4k_8k(tmp)),  ...)
    d_250 = clip(10log10(T250_TARGET / b150_300(tmp)), ...)
    g_hf += d_hf;  g_250 += d_250
    if |d_hf| < 0.2 and |d_250| < 0.2: 收工
```
目标值取自 EL 男声基线 `el_base_clean_male.json`：
    HF_TARGET = 0.0129（b4000_8000 均值）
    T250_TARGET = 0.254121（b150_300 均值）

## 风控
- 250Hz 增益钳在 `[-2.0, +1.0]`：历史教训 -1.5dB 就过冲，
  这里是**测出来**的而不是猜的，钳位只是防极端素材
- 顺序不变：250Hz 衰减仍与高频架子同处**第二段**，
  `t5`（只有 boost）的测量输入不受影响（插到前面实测 Σz² → 78.4）
- 只改两个架子的增益，**不动时长、不动语速、不加新的滤波器**

## 用法
    ./venv/bin/python deploy/apply_dual31.py
"""
import ast
import sys

P = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"

# ---- 1. 新增 250Hz 目标与钳位常量 ----
OLD_C = 'HF_DB_MIN, HF_DB_MAX = -9.5, 6.0'
NEW_C = ('HF_DB_MIN, HF_DB_MAX = -9.5, 6.0\n'
         '# 🔴 第31轮：150~300Hz 也改成自适应。写死的 -0.45dB 是按当年端到端\n'
         '#    手调出来的，而总能量会随高频架子变化 → 各频段占比跟着浮动，\n'
         '#    固定值不可能每次都刚好（历史：-1.5dB 过冲，Σz² 反而升到 44.9）。\n'
         'T250_FREQ = 250\n'
         'T250_TARGET = 0.254121      # EL 男声 b150_300 均值\n'
         'T250_DB_MIN, T250_DB_MAX = -2.0, 1.0   # -1.5dB 已过冲，上限定在这里\n'
         'MALE_250_DB = -0.45         # 迭代初值（沿用历史手调值）')

# ---- 2. 迭代块：高频 + 250Hz 一起收敛 ----
OLD = '''        _last = None
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

NEW = '''        g250 = MALE_250_DB
        _last = None
        for _it in range(4):
            af = (("highshelf=f=%d:g=%g," % (HF_FREQ, hf_db))
                  + ("equalizer=f=%d:t=q:w=1.1:g=%g," % (T250_FREQ, g250))
                  + "volume=%gdB" % gain)
            _t = os.path.join(tmpd, "ad%d.wav" % _it)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", t5,
                            "-af", af, _t], check=True)
            _r = adaptive_hf_ratio(_t)
            _r2 = _band_ratio(_t, 150, 300)
            _d = 0.0
            if _r > 0:
                _d = float(np.clip(10.0 * np.log10(HF_TARGET / _r),
                                   HF_DB_MIN - hf_db, HF_DB_MAX - hf_db))
            _d2 = 0.0
            if _r2 > 0:
                _d2 = float(np.clip(10.0 * np.log10(T250_TARGET / _r2),
                                    T250_DB_MIN - g250, T250_DB_MAX - g250))
            print("   迭代%d: b4k_8k=%.5f(补%+.2f)  b150_300=%.5f(补%+.2f)"
                  "  累计 hf%+.2f/250 %+.2f"
                  % (_it, _r, _d, _r2, _d2, hf_db + _d, g250 + _d2),
                  flush=True)
            hf_db = float(np.clip(hf_db + _d, HF_DB_MIN, HF_DB_MAX))
            g250 = float(np.clip(g250 + _d2, T250_DB_MIN, T250_DB_MAX))
            _last = _t
            if abs(_d) < 0.2 and abs(_d2) < 0.2:
                break
        shutil.move(_last, a.out)'''


def main():
    s = open(P, encoding="utf-8").read()
    if "T250_TARGET" in s:
        print("已打过补丁，跳过（幂等）")
        return 0
    n = 0
    for old, new in ((OLD_C, NEW_C), (OLD, NEW)):
        if old in s:
            s = s.replace(old, new, 1)
            n += 1
        else:
            print("❌ 未找到：%s" % old.split("\n")[0][:60])
    if n != 2:
        print("❌ 只替换 %d/2，放弃" % n)
        return 1
    try:
        ast.parse(s)
    except SyntaxError as e:
        print("❌ 语法错误 %s" % e)
        return 1
    open(P, "w", encoding="utf-8").write(s)
    print("✅ 已写入 %s（%d 处，语法 OK）" % (P, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
