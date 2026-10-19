# -*- coding: utf-8 -*-
"""gruntiso53.py —— 统计「句间孤立低能量鼻音碎片」（我们 vs EL 同口径）。

## 为什么要它

上一轮（z29 后）我判断「残留 en/砰砰 是模型 spurious phonemes，后处理到极限」。
但 EL 同文本对照（d172b8c4efb3/merged.wav）在对应位置也有同样密集的 5ms 鼻音碎片
（EL 98s / 128s）。→ 必须量化：这种碎片到底谁更多？

## 定义（孤立碎片，区别于 tailgrunt53e 的「句末粘连」）

    · 段时长 5~80ms
    · 峰值 −45 ~ −25dB（低能量；排除 −20dB 以上的正常语音）
    · 浊音/鼻音：zcr < 0.15 且 300Hz 以下能量 > 0.45
    · 前间隔 > 100ms 且 后间隔 > 100ms（真正孤立在静音里，不紧贴主语音）

按「个/分钟」对比两个文件。

用法：python3 gruntiso53.py <ours.wav> <el.wav>
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
VTH = -40.0


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
    if len(a) < 16:
        return 1.0, 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(a)))) / 2)
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    fr = np.fft.rfftfreq(len(a), 1.0 / sr)
    low = float(sp[fr < 300].sum() / (sp.sum() + 1e-12))
    return zcr, low


def count_iso(p, mingap=0.100):
    x = load16(p)
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    v = db > VTH
    raw, i = [], 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            raw.append((i, j))
            i = j
        else:
            i += 1
    dur = len(x) / SR
    hits = []
    for k in range(1, len(raw) - 1):
        a, b = raw[k]
        d = (b - a) * FRAME
        if not (0.005 <= d <= 0.080):
            continue
        pk = float(db[a:b].max())
        if not (-45 <= pk <= -25):
            continue
        gap_pre = (a - raw[k - 1][1]) * FRAME
        gap_nxt = (raw[k + 1][0] - b) * FRAME
        if gap_pre < mingap or gap_nxt < mingap:
            continue
        seg = x[int(a * FRAME * SR):int(b * FRAME * SR)]
        zcr, low = voiced_quality(seg)
        if not (zcr < 0.15 and low > 0.45):
            continue
        hits.append((a * FRAME, d * 1000, pk, zcr, low))
    return dur, hits


def main():
    if len(sys.argv) < 3:
        print("用法：gruntiso53.py <ours.wav> <el.wav> [--mingap 秒]")
        return 1
    mingap = 0.100
    if "--mingap" in sys.argv:
        mingap = float(sys.argv[sys.argv.index("--mingap") + 1])
    files = [a for a in sys.argv[1:] if not a.startswith("--")
             and a != str(mingap)]
    for p in files:
        dur, hits = count_iso(p, mingap)
        print("=== %s  %.1fs  门槛=前后静音>%.0fms ==="
              % (os.path.basename(p), dur, mingap * 1000))
        print("孤立低能量鼻音碎片：%d 个 (%.2f/分钟)"
              % (len(hits), len(hits) / dur * 60))
        if hits:
            ds = np.array([h[1] for h in hits])
            pks = np.array([h[2] for h in hits])
            print("  时长 中位=%.0fms p90=%.0fms | 峰值 中位=%.0fdB"
                  % (np.median(ds), np.percentile(ds, 90), np.median(pks)))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
