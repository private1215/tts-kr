# -*- coding: utf-8 -*-
"""tokstat_all48.py —— 全库 36 条案例文案的拉丁/数字占比分布。

单条文案（女声 3415 字）里拉丁字母 **0 个**、数字 **0 个**，
所以「영문약어 假错」对这条根本不存在。但员工会上传别的文案（半导体
材料、MLCC、AI、RD 之类），所以要看**全库分布**才知道值不值得修。

判据：CER 重掷阈值 0.35。若典型文案的拉丁+数字占比只有 0.5%，
就算每个 token 都被误判成 3 倍长度的错，也才 1.5%，
离 0.35 差 20 倍 → 这个优化永远触发不了重掷，不值得做。
"""
import glob
import os
import re
import sys

LAT = re.compile(r"[A-Za-z][A-Za-z0-9&\.\-]{0,11}")
DIG = re.compile(r"[0-9][0-9,\.]*")


def stat(t):
    n = max(len(t), 1)
    lat = LAT.findall(t)
    dig = DIG.findall(t)
    return (sum(len(x) for x in lat) / n, sum(len(x) for x in dig) / n,
            len(lat), len(dig), n)


def main():
    rows = []
    for p in sorted(glob.glob("/mnt/tts/data/cases/*/full.txt")):
        t = open(p, encoding="utf-8").read()
        r = stat(t)
        rows.append((p.split("/")[-2],) + r)
    rows.sort(key=lambda r: -(r[1] + r[2]))
    print("%-14s %8s %8s %8s %8s %8s" %
          ("案例", "字数", "拉丁%", "数字%", "拉丁数", "数字数"))
    for cid, lp, dp, ln, dn, n in rows:
        print("%-14s %8d %8.3f %8.3f %8d %8d"
              % (cid[:12], n, lp * 100, dp * 100, ln, dn))
    import statistics as st
    tot = [r[1] + r[2] for r in rows]
    print()
    print("拉丁+数字占比：中位 %.3f%%  均值 %.3f%%  最大 %.3f%%  （共 %d 条）"
          % (st.median(tot) * 100, sum(tot) / len(tot) * 100,
             max(tot) * 100, len(rows)))
    print("最坏情况上界（每个 token 错成 3 倍长）：%.3f%%"
          % (max(tot) * 3 * 100))
    print("🔴 重掷阈值 35%% → 还差 %.1f 倍" % (0.35 / max(max(tot) * 3, 1e-9)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
