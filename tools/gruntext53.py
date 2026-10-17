# -*- coding: utf-8 -*-
"""gruntext53.py —— 验证「嗯」是否比检测到的段更长（清了一截、剩一截）。

假设（用户 2026-10-17 指出）：
    「整个嗯的发音，每个嗯清了一部分没有全部清理」
→ 「嗯」有起音/持续/收尾，能量起伏。我的判据（db > p35）只把
   **能量够高的那一截**判为发声段，起音和收尾能量低，被当成静音漏掉。
→ 所以每个「嗯」都剩前后两截。

本工具对每个检测到的目标段，dump 前后各 400ms 的 5ms RMS(dB) 包络，
并算出「以该段为种子向两侧扩张（只要不低于 种子峰值-EXT_DB）」能覆盖多长。

用法：python3 gruntext53.py <wav> [更多...]
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
EXT_DB = 12.0     # 从种子峰值往下这么多 dB 以内，都算「这个嗯的一部分」
PAD_MS = 400      # 前后各看这么长


def load16(p, sr=SR):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def voiced_quality(a, sr=SR):
    if len(a) < 40:
        return 1.0, 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(a)))) / 2)
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    fr = np.fft.rfftfreq(len(a), 1.0 / sr)
    low = float(sp[fr < 300].sum() / (sp.sum() + 1e-12))
    return zcr, low


def segments(x):
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
            segs.append((i * FRAME, (j - i) * FRAME, float(db[i:j].max())))
            i = j
        else:
            i += 1
    return segs, db, n


def main():
    p = sys.argv[1]
    x = load16(p)
    segs, db, n = segments(x)
    print("=== %s ===" % os.path.basename(p))
    print("种子判据：20~250ms + 前后静音>0.08s + 浊音")
    print("扩展判据：从种子峰值 -%.0fdB 以内都算同一个「嗯」" % EXT_DB)
    print()
    print("%8s %8s %8s %10s %10s %s"
          % ("位置s", "种子ms", "扩展ms", "左延ms", "右延ms", "扩展后总长"))
    n_short = n_ext = 0
    for k in range(1, len(segs) - 1):
        s, d, pk = segs[k]
        if not (0.020 <= d <= 0.250):
            continue
        pe = segs[k - 1][0] + segs[k - 1][1]
        nx = segs[k + 1][0]
        if (s - pe) <= 0.08 or (nx - (s + d)) <= 0.08:
            continue
        a = x[int(s * SR):int((s + d) * SR)]
        zcr, low = voiced_quality(a)
        if not (zcr < 0.12 and low > 0.45):
            continue
        if pk < -32:
            continue
        # ---- 向两侧扩张 ----
        pad = int(PAD_MS / 1000 / FRAME)
        ci = int(s / FRAME)          # 种子起始帧
        cj = int((s + d) / FRAME)    # 种子结束帧
        i0 = max(0, ci - pad)
        i1 = min(len(db), cj + pad)
        thr = pk - EXT_DB
        # 向左：只要 >= thr 就继续（允许最多 3 帧低于阈值不断）
        li = ci
        gap = 0
        while li > i0 and gap <= 3:
            if db[li - 1] >= thr:
                li -= 1
                gap = 0
            else:
                gap += 1
                li -= 1
        # 向右同理
        ri = cj
        gap = 0
        while ri < i1 and gap <= 3:
            if db[ri] >= thr:
                ri += 1
                gap = 0
            else:
                gap += 1
                ri += 1
        left_ms = (ci - li) * FRAME * 1000
        right_ms = (ri - cj) * FRAME * 1000
        tot_ms = (ri - li) * FRAME * 1000
        print("%8.2f %8.0f %8.0f %10.0f %10.0f   %.0fms"
              % (s, d * 1000, tot_ms, left_ms, right_ms, tot_ms))
        n_short += 1
        if tot_ms > d * 1000 + 20:
            n_ext += 1
    print("\n共 %d 个种子；其中 %d 个可向左/右扩展出更多（说明确实清了一截）"
          % (n_short, n_ext))
    return 0


if __name__ == "__main__":
    sys.exit(main())
