# -*- coding: utf-8 -*-
"""离线试算：给已有成品铺房间底噪，看评分与静音电平怎么变。

🔴 必须在**成品口径**上叠（陷阱 #18）：线上是叠在 mp3 编码之前，
   这里就叠在成品 mp3 解码后的 wav 上，位置等价。

用法（服务器，qwen 环境）：
    venv/bin/python tools/try_roomtone.py            # 扫 -50/-53/-56/-60
    venv/bin/python tools/try_roomtone.py --db -56 --save /mnt/tts/diag/rt56
"""
import os
import sys
import glob
import argparse

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import roomtone                                    # noqa: E402
from tools.fidelity import measure, score, WEIGHTS          # noqa: E402

SR = 24000
CASES = [
    ("1a6c20a60481", "1a6c20", "male"),
    ("00b6a45ad6aa", "00b6a4", "female"),
    ("ba24c0925311", "ba24c0", "male"),
    ("8197d9ac8a35", "8197d9", "male"),
    ("6cdd25bc75f2", "6cdd25", "male"),
]
CASES_DIR = "/mnt/tts/data/cases"


def levels(x, sr=SR):
    """帧电平分位数：看静音底噪抬到哪了。"""
    W = int(0.05 * sr)
    n = len(x) // W
    e = np.array([np.sqrt(np.mean(x[i * W:(i + 1) * W] ** 2)) for i in range(n)])
    db = 20 * np.log10(e + 1e-12)
    return [float(np.percentile(db, q)) for q in (5, 10, 15, 25, 75)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="r22")
    ap.add_argument("--db", default="", help="-50,-53,-56,-60")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    grid = [float(v) for v in a.db.split(",")] if a.db else [-50.0, -53.0, -56.0, -60.0]

    print(f"{'案例':<10}{'性别':>7}{'底噪dB':>8}{'还原度':>9}{'Δ':>7}"
          f"{'p5':>8}{'p10':>8}{'p15':>8}   (案例 p5/p10/p15)")
    summary = {}
    for cid, short, gender in CASES:
        prod = sorted(glob.glob(f"/mnt/tts/output/{a.prefix}-{short}/*.mp3"))
        if not prod:
            continue
        work = f"/mnt/tts/diag/rtwork/{short}.wav"
        os.makedirs(os.path.dirname(work), exist_ok=True)
        import subprocess
        lst = work + ".txt"
        with open(lst, "w") as w:
            for f in prod:
                w.write("file '%s'\n" % f)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat",
                        "-safe", "0", "-i", lst, "-ac", "1", "-ar", str(SR), work],
                       check=True)
        case_w = os.path.join(CASES_DIR, cid, "merged.wav")
        cm = measure(case_w, "CASE")
        c5 = levels(sf.read(case_w, dtype="float32")[0])[:3]
        x0, _ = sf.read(work, dtype="float32")
        base = None
        for db in grid:
            roomtone.ENABLED = True
            y = roomtone.apply(x0, SR, db)
            p = work.replace(".wav", f"_rt{int(db)}.wav")
            sf.write(p, y, SR)
            pm = measure(p, "PROD")
            s, devs = score(cm, pm)
            lv = levels(y)
            if db == grid[0]:
                pm0 = measure(work, "PROD")
                base, _ = score(cm, pm0)
            print(f"{short:<10}{gender:>7}{db:>8.0f}{s:>9.2f}"
                  f"{(s - base if base else 0):>+7.2f}"
                  f"{lv[0]:>8.1f}{lv[1]:>8.1f}{lv[2]:>8.1f}"
                  f"   ({c5[0]:.1f}/{c5[1]:.1f}/{c5[2]:.1f})", flush=True)
            summary[(short, db)] = s
            if a.save:
                os.makedirs(a.save, exist_ok=True)
                sf.write(os.path.join(a.save, f"{short}_rt{int(db)}.wav"), y, SR)
    if len(grid) > 1:
        print("\n各电平均值：")
        for db in grid:
            v = [s for (sh, d), s in summary.items() if d == db]
            if v:
                print(f"  {db:>6.0f} dB -> {sum(v)/len(v):.2f}")


if __name__ == "__main__":
    main()
