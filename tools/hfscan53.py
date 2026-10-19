# -*- coding: utf-8 -*-
"""hfscan53.py —— 扫描高频(>4k)衰减，看能否降低 click 而不牺牲清晰度。

## 依据

优化后我们 vs EL（同编码）：click 仍是 **1.37 倍**，是唯一明显剩余差距。
频段分布显示我们高频略高：3k-8k 合计 4.27% vs EL 3.94%。
click = 归一后相邻样本 |Δ|>=0.30，主要来自陡峭瞬态（高频成分贡献大）。

## 做法

用 firequalizer 在 4k 以上做衰减（5k 处 -G，8k 处 -G），扫描 G，
测 click / 中频 / 质心 / 杂音。只选「click 明显降、中频不明显退」的点。

⚠️ 本机 ffmpeg `equalizer` 滤镜失效，必须用 firequalizer。

用法：python3 hfscan53.py <in.wav> <el.wav>
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


def hf_cut(p, g):
    if g == 0:
        return p
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p, "-af",
                    "firequalizer=gain_entry='entry(3000,0);"
                    "entry(5000,-%g);entry(8000,-%g)'" % (g, g),
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
    mid, cen, noi = [], [], []
    for i in range(len(v)):
        s = sp[i]
        tot = s.sum() + 1e-12
        if v[i]:
            mid.append(float(s[(freqs >= 1000) & (freqs <= 3000)].sum() / tot))
            cen.append(float((freqs * s).sum() / tot))
        else:
            noi.append(float(s[(freqs >= 4000) & (freqs <= 8000)].sum() / tot))
    return dict(click=click * 100,
                mid=float(np.median(mid)) * 100 if mid else 0,
                centroid=float(np.median(cen)) if cen else 0,
                noise=float(np.median(noi)) * 100 if noi else 0)


def main():
    if len(sys.argv) < 3:
        print("用法：hfscan53.py <in.wav> <el.wav>")
        return 1
    ours, el = sys.argv[1:3]
    e = metrics(el)
    print("EL 基准：click %.4f%%  中频 %.2f%%  质心 %.0f"
          % (e["click"], e["mid"], e["centroid"]))
    print()
    print("%-10s %10s %8s %8s %8s" % ("高频衰减", "click%", "中频%",
                                      "质心Hz", "杂音%"))
    for g in (0, 1, 2, 3, 5):
        m = metrics(hf_cut(ours, g))
        mark = ""
        if abs(m["click"] - e["click"]) / e["click"] < 0.15:
            mark = "  <<< click 接近 EL"
        print("%-10s %10.4f %8.2f %8.0f %8.2f%s"
              % ("-%g dB" % g if g else "0", m["click"], m["mid"],
                 m["centroid"], m["noise"], mark))
    return 0


if __name__ == "__main__":
    sys.exit(main())
