#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第31轮：回退「250Hz 自适应」，只保留高频迭代

## 为什么要回退
同一配置连跑两次（r33 / r34）证明平坦度是**配置**导致的，不是生成波动：

| | r33 | r34 | 结论 |
|---|---|---|---|
| 平坦度 | 0.0309（−5.02σ） | 0.0323（−4.55σ） | 差 4.5% → **配置决定** |
| 发声占比 | 0.5553（−0.10σ） | 0.5743（+2.79σ） | 差 2.9σ → 生成随机量 |

而不带 250Hz 自适应的 r32 平坦度是 **0.0470（+0.30σ）**。
→ 加了 250Hz 自适应之后平坦度从 0.0470 崩到 0.031，**−33%**。

## 机理
`equalizer=f=250:t=q:w=1.1` 是个**窄带**滤波器。窄带衰减对
`平坦度 = 几何均值 / 算术均值` 是致命的：**几何均值对接近零的频点极其敏感**，
陷波一深，几何均值立刻掉；而算术均值几乎不动（那段只占总能量的一小部分）。
→ 分母不变、分子塌 → 平坦度崩。
历史教训同源：「highshelf -9.5dB 把 flat 打到 -6σ（发假）」、
「MALE_250 -1.5dB 过冲，Σz² 反而升到 44.9 / 33.7」。

带 250Hz 自适应后 g250 被推到约 −1.08 dB，正落在历史说的过冲区间。

## 回退到什么
保留**高频迭代**（r32 那一版，Σz² 23.9、平坦度 +0.30σ），
`MALE_250` 恢复成**写死的 −0.45dB**（当年按端到端反馈手调出来的值）。

## 用法
    ./venv/bin/python deploy/revert_dual31.py
"""
import ast
import sys

P = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"

OLD = '''        g250 = MALE_250_DB
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

NEW = '''        # 🔴 第31轮：250Hz **不参与**迭代，恢复写死的 MALE_250（−0.45dB）。
        #    试过把它也做成自适应（deploy/apply_dual31.py），同配置连跑两次
        #    （r33/r34）证明平坦度是配置导致的：0.0470 → 0.031（−33%）。
        #    机理：`equalizer=f=250:t=q:w=1.1` 是窄带，窄带衰减对
        #    「平坦度 = 几何均值/算术均值」是致命的 —— 几何均值对接近零的
        #    频点极敏感，陷波一深分子就塌，而分母几乎不动。
        #    历史同源教训：highshelf -9.5dB 把 flat 打到 -6σ（发假）；
        #    MALE_250 -1.5dB 过冲，Σz² 反而升到 44.9 / 33.7。
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
    if "g250 = MALE_250_DB" not in s:
        print("已经是回退状态（或从未打过 dual 补丁），跳过")
        return 0
    if OLD not in s:
        print("❌ 未定位到联合迭代块")
        return 1
    s = s.replace(OLD, NEW, 1)
    # 保留常量定义但标注不再使用，避免别处引用报错
    try:
        ast.parse(s)
    except SyntaxError as e:
        print("❌ 语法错误 %s" % e)
        return 1
    open(P, "w", encoding="utf-8").write(s)
    print("✅ 已回退为「只迭代高频，250Hz 固定 -0.45dB」（语法 OK）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
