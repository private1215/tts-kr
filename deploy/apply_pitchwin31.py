#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第31轮：变调窗从 60s 细化到可配置（--pitch-win，默认 30s）

## 为什么要做
融合补丁之后复测**跨窗一致性**，大部分指标大幅改善：

| 跨窗极差% | r30fix（旧） | **r31（融合）** | EL |
|---|---|---|---|
| 4k~8k | 117.3% | **77.2%** | 83.0% ✅ 已优于 EL |
| 150~300 | 41.3% | **29.6%** | 22.4% |
| 平坦度 | 69.2% | **54.7%** | 31.5% |
| **F0** | 15.9% | **21.2% ❌ 反而变差** | 15.1% |

逐窗 F0（30s 窗，Hz）：
```
r31  [207.7 205.3 207.7 204.2 192.7 174.7 203.0 213.2 190.5 189.4 200.7 217.6 198.9 199.5]
EL   [221.4 193.3 204.2 194.9 199.5 194.9 207.7 199.5 201.8 191.6 197.2 205.3 192.1 192.7 191.6]
     ↑ r31 有两个离群窗：174.7（第6窗）、217.6（第12窗），极差 42.9Hz vs EL 29.8Hz
```
根因：变调按 **60s** 一个窗校正，而音高漂移发生在**窗内**
（第6/7窗 174.7 与 203.0 同属一个 60s 校正窗，窗内就差了 28Hz）。
→ 把校正窗细化到 30s，让它跟得上漂移。

## 为什么速度窗不动（仍 60s）
`fs`（语速系数）也按 WIN 分窗。语速是**全局**属性，细分只会引入窗间
速度抖动，没有收益，所以只细化变调窗，速度窗保持 60s（已验证）。

## 用法
    ./venv/bin/python deploy/apply_pitchwin31.py
"""
import ast
import sys

P = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"

# ---- 1. 新增 --pitch-win 参数 ----
OLD_ARG = '''    ap.add_argument("--no-pitch", action="store_true",'''
NEW_ARG = '''    ap.add_argument("--pitch-win", type=float, default=30.0,
                    help="变调分窗长度(秒)。音高漂移发生在窗内，60s 太粗"
                         "（实测跨窗 F0 极差 42.9Hz vs EL 29.8Hz）。"
                         "速度窗仍用 WIN=60s，不受此参数影响。")
    ap.add_argument("--no-pitch", action="store_true",'''

# ---- 2. 变调段：改用子窗 ----
OLD_BLK = '''    tf = a.target_f0 if a.target_f0 else DEFAULT_F0[g]
    if a.no_pitch:
        ratios = [1.0] * len(segs)
    elif a.pitch:
        ratios = [a.pitch] * len(segs)
    else:
        ratios = []
        for s2 in segs:
            f0w = f0_median(s2, sr, win=1e9)   # 该窗**全段**，不是前 60s
            ratios.append(min(max(tf / f0w, 0.90), 1.30)
                          if f0w and f0w > 0 else 1.0)
        # 相邻窗之间 1:2:1 平滑，避免比例跳变在拼接点听出来
        if len(ratios) > 2:
            ratios = ([ratios[0]]
                      + [(ratios[i - 1] + 2 * ratios[i] + ratios[i + 1]) / 4
                         for i in range(1, len(ratios) - 1)]
                      + [ratios[-1]])
        print("分窗 F0: " + ", ".join(
            "%.0f" % x for x in
            [(f0_median(s2, sr, win=1e9) or 0) for s2 in segs]), flush=True)
        print("分窗变调: " + ", ".join("%.3f" % r for r in ratios)
              + " （目标 %.1f Hz）" % tf, flush=True)'''

NEW_BLK = '''    tf = a.target_f0 if a.target_f0 else DEFAULT_F0[g]

    # 🔴 第31轮：变调窗**单独细化**（--pitch-win，默认 30s）。
    #    融合模式下 buf 就是原始 y（变速已折进下面的 atempo，此处未拉伸），
    #    所以可以直接用 graw（原始停顿位置）来切子窗。
    PW = max(5.0, float(getattr(a, "pitch_win", 0) or WIN))
    _pw = [0.0]
    for i in range(1, int(dur // PW) + 1):
        want = i * PW
        cand = [gg for gg in graw if abs(gg[0] - want) <= 8.0 and gg[1] >= 0.08]
        b = min(cand, key=lambda gg: abs(gg[0] - want))[0] if cand else want
        _pw.append(max(b, _pw[-1] + 1.0))
    if _pw[-1] < dur - 0.05:
        _pw.append(dur)
    if len(_pw) < 2:
        _pw = [0.0, dur]
    sub_segs, fs_sub = [], []
    for i in range(len(_pw) - 1):
        s0, s1 = _pw[i], _pw[i + 1]
        if s1 - s0 < 0.5:
            continue
        sub_segs.append(buf[int(s0 * sr):int(s1 * sr)])
        # 语速系数按**原始时间**的中心点映射到 60s 的 fs 窗
        j = min(int(((s0 + s1) / 2.0) // WIN), len(fs) - 1)
        fs_sub.append(fs[max(0, j)])
    print("变调子窗 %d 个（每窗 %.0fs，速度窗仍 %.0fs）"
          % (len(sub_segs), PW, WIN), flush=True)

    if a.no_pitch:
        ratios = [1.0] * len(sub_segs)
    elif a.pitch:
        ratios = [a.pitch] * len(sub_segs)
    else:
        ratios = []
        for s2 in sub_segs:
            f0w = f0_median(s2, sr, win=1e9)   # 该窗**全段**，不是前 60s
            ratios.append(min(max(tf / f0w, 0.90), 1.30)
                          if f0w and f0w > 0 else 1.0)
        # 相邻窗之间 1:2:1 平滑，避免比例跳变在拼接点听出来
        if len(ratios) > 2:
            ratios = ([ratios[0]]
                      + [(ratios[i - 1] + 2 * ratios[i] + ratios[i + 1]) / 4
                         for i in range(1, len(ratios) - 1)]
                      + [ratios[-1]])
        print("分窗 F0: " + ", ".join(
            "%.0f" % x for x in
            [(f0_median(s2, sr, win=1e9) or 0) for s2 in sub_segs]), flush=True)
        print("分窗变调: " + ", ".join("%.3f" % r for r in ratios)
              + " （目标 %.1f Hz）" % tf, flush=True)'''

OLD_LOOP = '''        new_segs = []
        for i, s2 in enumerate(segs):
            R = ratios[i]
            if abs(R - 1.0) < 1e-6 and abs(fs[i] - 1.0) < 1e-6:'''
NEW_LOOP = '''        new_segs = []
        for i, s2 in enumerate(sub_segs):
            R = ratios[i]
            if abs(R - 1.0) < 1e-6 and abs(fs_sub[i] - 1.0) < 1e-6:'''

OLD_T = '''            _fuse_t = 1.0 / (R * fs[i]) if fs[i] > 0 else 1.0 / R'''
NEW_T = '''            _fuse_t = (1.0 / (R * fs_sub[i]) if fs_sub[i] > 0
                       else 1.0 / R)'''


def main():
    s = open(P, encoding="utf-8").read()
    if "--pitch-win" in s and "sub_segs" in s:
        print("已打过补丁，跳过（幂等）")
        return 0
    n = 0
    for old, new in ((OLD_ARG, NEW_ARG), (OLD_BLK, NEW_BLK),
                     (OLD_LOOP, NEW_LOOP), (OLD_T, NEW_T)):
        if old in s:
            s = s.replace(old, new, 1)
            n += 1
        else:
            print("❌ 未找到：%s" % old.split("\n")[0][:60])
    if n != 4:
        print("❌ 只替换 %d/4，放弃" % n)
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
