# -*- coding: utf-8 -*-
"""gruntdiff53.py —— dump 「D 判据比 A 判据多出来」的段，判断是否为漏掉的「嗯」。

A 原判据：20~120ms，前后静音 >0.15s（gruntclean53 实际用的）
D 放宽：  20~250ms，前后静音 >0.08s

对每个「只在 D 中出现的段」输出时长、峰值、**浊音特征**（过零率 + 低频占比），
并给出分类建议：
    · 浊音（zcr 低 + 低频高）→ 大概率是漏掉的「嗯」，应删
    · 清音/噪声 → 可能是正常语音碎片，慎删
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005


def load16(p, sr=SR):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


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
    return segs


def find(x, lo_ms, hi_ms, sil):
    segs = segments(x)
    lo, hi = lo_ms / 1000.0, hi_ms / 1000.0
    out = []
    for k in range(1, len(segs) - 1):
        s, d, pk = segs[k]
        if lo <= d <= hi:
            pe = segs[k - 1][0] + segs[k - 1][1]
            nx = segs[k + 1][0]
            if (s - pe) > sil and (nx - (s + d)) > sil:
                out.append((s, d, pk))
    return out


def quality(x, s, d):
    a = x[int(s * SR):int((s + d) * SR)]
    if len(a) < 40:
        return 1.0, 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(a)))) / 2)
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    fr = np.fft.rfftfreq(len(a), 1.0 / SR)
    low = float(sp[fr < 300].sum() / (sp.sum() + 1e-12))
    return zcr, low


def main():
    p = sys.argv[1]
    x = load16(p)
    A = find(x, 20, 120, 0.15)
    D = find(x, 20, 250, 0.08)
    a_set = set(round(s, 2) for s, _, _ in A)
    extra = [(s, d, pk) for s, d, pk in D if round(s, 2) not in a_set]
    print("%s  A=%d 个，D=%d 个，D 多出 %d 个" %
          (os.path.basename(p), len(A), len(D), len(extra)))
    print("%8s %8s %9s %7s %7s %s" %
          ("位置s", "时长ms", "峰值dB", "zcr", "低频", "判定"))
    n_voice = 0
    for s, d, pk in extra:
        zcr, low = quality(x, s, d)
        is_v = zcr < 0.12 and low > 0.45
        n_voice += is_v
        print("%8.2f %8.0f %9.1f %7.3f %7.2f %s"
              % (s, d * 1000, pk, zcr, low, "浊音→嗯" if is_v else "其它"))
    print("\n多出 %d 个中，浊音（疑似漏删的「嗯」）%d 个，其它 %d 个"
          % (len(extra), n_voice, len(extra) - n_voice))
    return 0


if __name__ == "__main__":
    sys.exit(main())
