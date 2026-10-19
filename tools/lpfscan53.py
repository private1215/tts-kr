# -*- coding: utf-8 -*-
"""lpfscan53.py —— 只切极高频(7~8k)，看能否降 click 而几乎不动质心。

## 依据

男女声共同的**唯一剩余差距就是 click**（男 1.37 倍、女 1.59 倍）。
此前证伪：`hfscan53.py` 衰减 >4k 整体 −5dB 只降 click 10%，
却把质心从 544 压到 502（低于 EL，发闷）——因为砍到了 4~6k 有效区。

本脚本改为**只切 7~8kHz**（16k 采样的 Nyquist 附近）。这一段：
    · 能量占比极小（男声 6k-8k 仅 0.19%）
    · 人耳对 7-8k 极不敏感（主要是气息/齿音尾梢）
    · 若 click 主要来自这里的陡峭瞬态，切掉它 click 应明显降、质心几乎不动

用法：python3 lpfscan53.py <in.wav> <el.wav>
"""
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
VTH = -40.0


def load16(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    return x.astype(np.float64)


def lpf(p, f):
    if f >= 8000:
        return p
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-af", "lowpass=f=%d:p=2" % f,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    return t


def metrics(p):
    x = load16(p)
    pk = np.max(np.abs(x))
    xn = x * (0.70 / pk) if pk > 0 else x
    click = float((np.abs(np.diff(xn)) >= 0.30).mean())
    n = int(FRAME * SR)
    nfr = len(x) // n
    fr = x[:nfr * n].reshape(nfr, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    v = db > VTH
    sp = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    mid, cen = [], []
    for i in np.where(v)[0]:
        s = sp[i]
        tot = s.sum() + 1e-12
        mid.append(float(s[(freqs >= 1000) & (freqs <= 3000)].sum() / tot))
        cen.append(float((freqs * s).sum() / tot))
    return dict(click=click * 100,
                mid=float(np.median(mid)) * 100 if mid else 0,
                centroid=float(np.median(cen)) if cen else 0)


def main():
    if len(sys.argv) < 3:
        print("用法：lpfscan53.py <in.wav> <el.wav>")
        return 1
    ours, el = sys.argv[1:3]
    e = metrics(el)
    print("EL 基准：click %.4f%%  中频 %.2f%%  质心 %.0f"
          % (e["click"], e["mid"], e["centroid"]))
    print()
    print("%-12s %10s %8s %8s" % ("低通", "click%", "中频%", "质心Hz"))
    for f in (8000, 7800, 7500, 7000, 6500):
        m = metrics(lpf(ours, f))
        lab = "无" if f >= 8000 else "%d Hz" % f
        print("%-12s %10.4f %8.2f %8.0f" % (lab, m["click"], m["mid"],
                                            m["centroid"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
