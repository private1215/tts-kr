# -*- coding: utf-8 -*-
"""用「平滑 shelf」替代窄带 peaking，看能不能既修频段又保住 flat。

## 依据（2026-10-14）
flat_ablate.py 实测：单个 peaking 各吃掉 flat 0.6~1.2σ，4 个累加 -2.8σ；
而 `highshelf4000 -9.5` 一个就要 -5.2σ。窄带滤波把频谱搞"尖"＝flat 掉。
shelf 是**单调平滑**的，理论上对 flat 温和得多。
频段缺口也很适合 shelf：
    60~150 -4.17 / 150~300 -3.91 / 300~500 -4.41  （低频整体不足）
    2000~4000 +12.52 / 4000~8000 +18.80           （高频整体过量）
两块各自都是单调的 -> 一个 lowshelf + 一个 highshelf 就能覆盖大半。

坐标下降：lowshelf(f,g) x highshelf(f,g) x 可选 1400 补峰，目标同 eq_search6.cost。
"""
import itertools
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eq_search6 as E  # noqa: E402

E.FORMANT_W = 0.0       # 共振峰太慢且 F1 已判定不可用，这里只管频段+flat

LF = [250, 350, 450, 550]
LG = [0, 2, 4, 6, 8]
HF = [1800, 2200, 2600, 3000]
HG = [-2, -4, -6, -8, -10]
PG = [0, 2, 4]          # 1400Hz 补峰（EL b1000_2000 略高）


def chain(lf, lg, hf, hg, pg):
    p = []
    if lg:
        p.append("lowshelf=f=%d:g=%g" % (lf, lg))
    if pg:
        p.append("equalizer=f=1400:t=q:w=1.1:g=%g" % pg)
    if hg:
        p.append("highshelf=f=%d:g=%g" % (hf, hg))
    p.append("lowpass=f=14000")
    return ",".join(p)


def run(src, params):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-af",
                    chain(*params), "-ar", "24000", "-ac", "1", t], check=True)
    j = E.cost(E.measure(t))
    os.unlink(t)
    return j


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "base_pw.wav"
    src = src if os.path.isabs(src) else os.path.join(HERE, src)
    best_p = (350, 4, 2600, -6, 0)
    best_j = run(src, best_p)
    print("起点 %s J=%.1f" % (chain(*best_p), best_j), flush=True)

    for rnd in range(3):
        improved = False
        for slot, choices in enumerate((LF, LG, HF, HG, PG)):
            cur = best_p[slot]
            bi, bj = cur, best_j
            for c in choices:
                if c == cur:
                    continue
                p = list(best_p)
                p[slot] = c
                try:
                    j = run(src, tuple(p))
                except Exception as e:
                    print("  FAIL", repr(e)[:70], flush=True)
                    continue
                if j < bj - 1e-9:
                    bj, bi = j, c
            if bi != cur:
                p = list(best_p)
                p[slot] = bi
                best_p, best_j, improved = tuple(p), bj, True
                print("  R%d slot%d -> %s  J=%.1f" % (rnd, slot, bi, bj),
                      flush=True)
        print("第 %d 轮 J=%.1f  %s" % (rnd, best_j, chain(*best_p)), flush=True)
        if not improved:
            break

    # 与 v10 对比
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-af",
                    chain(*best_p), "-ar", "24000", "-ac", "1", t], check=True)
    m = E.measure(t)
    os.unlink(t)
    print("\n最优 shelf EQ: %s" % chain(*best_p))
    print("J=%.1f" % E.cost(m))
    for k in E.BASE:
        v, (mm, ss) = m[k], E.BASE[k]
        z = (v - mm) / ss
        flag = "[X]" if abs(z) > 2 else ("[!]" if abs(z) > 1.5 else "[ok]")
        print("  %-4s %-11s %9.4f  EL %9.4f  z=%+6.2f" % (flag, k, v, mm, z))
    print("\n（参考 v10 现役：J=19.2，flat z=-3.51，频段 max|z|=1.67）")
    print("对比用：%s" % ",".join(str(x) for x in best_p))


if __name__ == "__main__":
    main()
