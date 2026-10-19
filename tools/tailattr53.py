# -*- coding: utf-8 -*-
"""tailattr53.py —— 「句末粘连嗯」归因：我们 56 vs EL 36，**多出来的是哪一档？**

## 为什么必须做这一步
保采样率清理链（pipe_1019d）实测把句末从 5.99/分 清到 **0.64/分**，
而 EL 是 **3.89/分** —— 清过头了。文档 §1.2 铁律：
**清洗目标是对齐 EL 密度，严禁过度清洗**。

看日志能直接看出问题：56 个里有 45 个是 **5ms / -36~-40dB**（几乎听不见的
麦克风级碎屑），只有约 11 个是 **20~120ms / -24~-31dB**（真「嗯」）。
一刀切全清 = 把 EL 也有的正常句末收尾也抹掉了。

## 做法
把两边检出的簇按 (时长档 × 峰值dB档) 做二维桶，逐格算 **每分密度 我们/EL**。
- 比值 ≈1 的格子 = 正常句末收尾，**不许清**
- 比值 >>1 的格子 = 我们特有的缺陷，**只清这些**

输出一张表 + 各候选判据下的「清掉多少 / 残留密度 / 距 EL 多远」，
用来定 tailclean53b3 的判据。

用法: venv/bin/python tools/tailattr53.py <ours> <el>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tailclean53b import SR, find_tail_clusters, load16  # noqa: E402

# (下界ms, 上界ms, 标签)
DUR_BINS = [(0, 10, "5-10ms"), (10, 20, "10-20ms"), (20, 40, "20-40ms"),
            (40, 80, "40-80ms"), (80, 130, "80-130ms")]
# (下界dB, 上界dB, 标签)
PK_BINS = [(-45, -38, "-45..-38"), (-38, -34, "-38..-34"),
           (-34, -30, "-34..-30"), (-30, -26, "-30..-26"),
           (-26, -20, "-26..-20")]


def feats(p):
    x = load16(p)
    t = find_tail_clusters(x)
    dur = len(x) / SR
    rows = []
    for s, d, pk, zcr, low, gp, gn, nseg in t:
        rows.append({"s": s, "dms": d * 1000.0, "pk": pk, "zcr": zcr,
                     "low": low, "pre": gp * 1000.0, "nxt": gn * 1000.0,
                     "nseg": nseg})
    return rows, dur


def table(rows, dur):
    """返回 {(dur标签, pk标签): 每分密度}"""
    g = {}
    for r in rows:
        for lo, hi, dl in DUR_BINS:
            if lo <= r["dms"] < hi:
                break
        else:
            continue
        for lo, hi, pl in PK_BINS:
            if lo <= r["pk"] < hi:
                break
        else:
            continue
        g[(dl, pl)] = g.get((dl, pl), 0) + 1
    return {k: v / dur * 60 for k, v in g.items()}


def main():
    ours_p, el_p = sys.argv[1], sys.argv[2]
    ro, do = feats(ours_p)
    re_, de = feats(el_p)
    to, te = table(ro, do), table(re_, de)

    print("== 检出量 ==")
    print("  我们 %s  %.1fs  %d 个 (%.2f/分)"
          % (os.path.basename(ours_p), do, len(ro), len(ro) / do * 60))
    print("  EL   %s  %.1fs  %d 个 (%.2f/分)"
          % (os.path.basename(el_p), de, len(re_), len(re_) / de * 60))
    print("  总比值 %.2f" % ((len(ro) / do) / (len(re_) / de)))

    print("\n== 二维分布：每分密度（我们 / EL / 比值）==")
    hdr = "%-9s" % "时长\\峰值"
    print(hdr + "".join("%14s" % p for _, _, p in PK_BINS))
    for lo, hi, dl in DUR_BINS:
        line = "%-9s" % dl
        for plo, phi, pl in PK_BINS:
            o = to.get((dl, pl), 0.0)
            e = te.get((dl, pl), 0.0)
            if o == 0 and e == 0:
                line += "%14s" % "."
            elif e == 0:
                line += "%14s" % ("%.2f/0/∞" % o)
            else:
                line += "%14s" % ("%.2f/%.2f/%.1f" % (o, e, o / e))
        print(line)

    # ================= 候选判据扫描 =================
    print("\n== 候选判据扫描（只清满足条件的，目标：残留密度贴近 EL 的 %.2f/分）=="
          % (len(re_) / de * 60))
    print("%-34s %8s %8s %8s %8s"
          % ("判据", "清掉", "残留/分", "距EL", "清掉占比%"))
    cands = [
        ("全清（现状，过清）", lambda r: True),
        ("dms>=20", lambda r: r["dms"] >= 20),
        ("dms>=20 & pk>-34", lambda r: r["dms"] >= 20 and r["pk"] > -34),
        ("dms>=20 & pk>-32", lambda r: r["dms"] >= 20 and r["pk"] > -32),
        ("dms>=10 & pk>-32", lambda r: r["dms"] >= 10 and r["pk"] > -32),
        ("dms>=10 & pk>-34", lambda r: r["dms"] >= 10 and r["pk"] > -34),
        ("dms>=40", lambda r: r["dms"] >= 40),
        ("pk>-30", lambda r: r["pk"] > -30),
        ("pk>-32", lambda r: r["pk"] > -32),
        ("dms>=20 & pk>-32 & nseg>=2",
         lambda r: r["dms"] >= 20 and r["pk"] > -32 and r["nseg"] >= 2),
    ]
    el_den = len(re_) / de * 60
    for name, fn in cands:
        hit = [r for r in ro if fn(r)]
        left = len(ro) - len(hit)
        den = left / do * 60
        # EL 同样判据下会误伤多少（用来估「过度清洗」程度）
        elhit = [r for r in re_ if fn(r)]
        print("%-34s %8d %8.2f %8.2f %8.0f   (EL同判据会清掉%d)"
              % (name, len(hit), den, den - el_den,
                 100.0 * len(hit) / max(1, len(ro)), len(elhit)))

    print("\n== 我们多出来的候选（dms>=20 & pk>-32）明细 ==")
    for r in sorted([r for r in ro if r["dms"] >= 20 and r["pk"] > -32],
                    key=lambda r: r["s"]):
        print("   t=%7.2fs %5.0fms 峰值%.1fdB zcr=%.3f low=%.2f "
              "前%.0fms 后%.0fms 段%d"
              % (r["s"], r["dms"], r["pk"], r["zcr"], r["low"],
                 r["pre"], r["nxt"], r["nseg"]))
    print("\n== EL 同判据明细（用来确认 EL 也有、不能全清）==")
    for r in sorted([r for r in re_ if r["dms"] >= 20 and r["pk"] > -32],
                    key=lambda r: r["s"]):
        print("   t=%7.2fs %5.0fms 峰值%.1fdB zcr=%.3f low=%.2f "
              "前%.0fms 后%.0fms 段%d"
              % (r["s"], r["dms"], r["pk"], r["zcr"], r["low"],
                 r["pre"], r["nxt"], r["nseg"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
