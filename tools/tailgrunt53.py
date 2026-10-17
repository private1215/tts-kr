# -*- coding: utf-8 -*-
"""tailgrunt53.py —— 检测「句末尾音嗯」（员工：说完话就跟带口头禅似的有尾音嗯）。

## 为什么 gruntclean53.py 没清干净

`gruntclean53.py` 的判据是「**前后都** >0.15s 静音的孤立极短段」。
但员工说的是**说完话带的尾音嗯**——它**紧跟在发音段之后**，前面没有静音，
所以整类被漏掉。

## 本工具判据

找所有「长停顿」（>= PAUSE_MIN），看它**前面紧贴的那个发声段**：
    · 时长 <= TAIL_MAX_MS（默认 150ms）
    · 或该段峰值能量明显低于相邻发声段（低 TAIL_DROP_DB 以上）
→ 标记为「疑似尾音嗯」。

再对比我们与 EL：若 EL 没有这类结构而我们大量存在，即为根因。

用法：python3 tailgrunt53.py <wav> [更多...]
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
TAIL_MAX_MS = 150.0
TAIL_DROP_DB = 8.0


def load16(p, sr=SR):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def analyse(x):
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > np.percentile(db, 35)
    # 所有发声段 [(start_s, dur_s, peak_db)]
    segs, i = [], 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            segs.append((i * FRAME, (j - i) * FRAME, float(db[i:j].max())))
            i = j
        else:
            i += 1
    tails = []
    for k in range(1, len(segs)):
        s, d, pk = segs[k - 1]
        nxt = segs[k][0]
        gap = nxt - (s + d)
        if gap >= PAUSE_MIN:
            prev_pk = segs[k - 2][2] if k >= 2 else pk
            short = d * 1000 <= TAIL_MAX_MS
            weak = (prev_pk - pk) >= TAIL_DROP_DB
            if short or weak:
                tails.append((s, d * 1000, pk, "短" if short else "弱"))
    return segs, tails


def main():
    print("判据：长停顿(>=%.2fs) 前紧贴的发声段，若 <=%.0fms 或比前段低 %.0fdB → 尾音嗯"
          % (PAUSE_MIN, TAIL_MAX_MS, TAIL_DROP_DB))
    print("%-24s %8s %10s %10s" % ("文件", "发声段", "尾音嗯", "次/分钟"))
    for p in sys.argv[1:]:
        x = load16(p)
        segs, tails = analyse(x)
        dur = len(x) / SR
        print("%-24s %8d %10d %10.2f" % (os.path.basename(p), len(segs),
                                         len(tails), len(tails) / dur * 60))
        for s, ms, pk, why in tails[:15]:
            print("      t=%7.2fs  %3.0fms  peak=%.1fdB  (%s)" % (s, ms, pk, why))
    return 0


if __name__ == "__main__":
    sys.exit(main())
