# -*- coding: utf-8 -*-
"""tailgrunt53c.py —— 检测发言段**内部的双峰结构**（主音节 + 尾随「嗯」）。

## 为什么前面两个工具都没抓到时

tailgrunt53b.py 找的是「独立发声段」，结果我们 1 个、EL 0 个，差异太小。
员工说的是「说完话就跟带口头禅似的有**尾音**嗯」——它**连在发言段末尾**，
和主音节属于同一个连续发声段，所以分段检测根本看不到。

## 本工具判据

对每个「长停顿(>=PAUSE_MIN)之前的发声段」，算其 5ms RMS 包络，找双峰：
    主峰 → 凹陷(低于主峰 DIP_DB) → 次峰(仍高于主峰- SECOND_DB，且次峰时长
    在 SECOND_MS 范围) → 之后进静音
→ 「主音节 + 尾随小嗯」的波形正是这个形状。

对比我们与 EL 的双峰**比例**（不是绝对数，避免受发声段总数影响）。

用法：python3 tailgrunt53c.py <wav> [更多...]
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
PAUSE_MIN = 0.25
DIP_DB = 5.0          # 两峰之间至少凹陷这么多
SECOND_DB = 14.0      # 次峰不得低于主峰这么多
SECOND_MS = (30.0, 260.0)   # 次峰（那个「嗯」）的时长范围


def load16(p, sr=SR):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def is_double_peak(env_db):
    """env_db: 该发声段的 5ms RMS(dB) 序列。返回 (是否双峰, 次峰起始帧, 次峰长)。"""
    L = len(env_db)
    if L < 8:
        return False, 0, 0
    i1 = int(np.argmax(env_db))
    if i1 >= L - 4:
        return False, 0, 0
    # 主峰后必须掉到 DIP_DB 以下，再起来
    tail = env_db[i1:]
    dip = np.min(tail)
    if dip > env_db[i1] - DIP_DB:
        return False, 0, 0
    j = i1 + int(np.argmin(tail))
    if j >= L - 2:
        return False, 0, 0
    seg2 = tail[j - i1:]
    i2rel = int(np.argmax(seg2))
    i2 = j + i2rel
    if env_db[i2] < env_db[i1] - SECOND_DB:
        return False, 0, 0
    # 次峰时长：从 j 到该段结束
    dur_ms = (L - j) * FRAME * 1000
    if not (SECOND_MS[0] <= dur_ms <= SECOND_MS[1]):
        return False, 0, 0
    return True, j, L - j


def analyse(x):
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > np.percentile(db, 35)
    segs, i = [], 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    n_pause_end, dp = 0, []
    for k in range(1, len(segs)):
        a, b = segs[k - 1]
        # 后面是否长停顿
        nxt = segs[k][0] if k < len(segs) else nfr
        if (nxt - b) * FRAME < PAUSE_MIN:
            continue
        n_pause_end += 1
        ok, j0, ln = is_double_peak(db[a:b])
        if ok:
            dp.append(((a + j0) * FRAME, ln * FRAME * 1000))
    return n_pause_end, dp


def main():
    print("判据：长停顿(>=%.2fs)前的发言段内部呈「主峰→凹陷>=%.0fdB→次峰"
          "(不低于主峰%.0fdB，时长%d~%dms)」双峰" %
          (PAUSE_MIN, DIP_DB, SECOND_DB, SECOND_MS[0], SECOND_MS[1]))
    print("%-26s %10s %10s %12s" % ("文件", "停顿前段", "双峰数", "双峰占比"))
    for p in sys.argv[1:]:
        x = load16(p)
        n_end, dp = analyse(x)
        pct = (len(dp) / n_end * 100) if n_end else 0.0
        print("%-26s %10d %10d %11.1f%%" % (os.path.basename(p), n_end,
                                            len(dp), pct))
        for s, ms in dp[:10]:
            print("      t=%7.2fs  次峰长 %.0fms" % (s, ms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
