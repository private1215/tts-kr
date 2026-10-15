#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""房间底噪**频谱斜率**实验：不改电平，只改噪声的"颜色"。

为什么值得试
------------
女声 flat 现在 +2.89/+2.93σ（两篇一致 = 系统性）。已知：
  · 无底噪的链路产物 flat = 0.0222（+0.46σ，达标）
  · 叠 -62dB 粉噪后 -> +2.89σ
也就是说 **roomtone 一手贡献了约 2.4σ**。

之前降电平（-62 -> -80）确实能降 flat，但 voiced 会一路恶化（-1.6 -> -2.9 -> -3.5），
因为底噪声的是"停顿处的绝对电平"，电平一低停顿就更空。

✅ 另一条路：**保持 -62dB 的 RMS 不变，把频谱做得更陡**（能量更集中到低频）。
   RMS 不变 -> voiced 不受影响；频谱更"有结构" -> flat 下降。
   反过来，男声 flat **偏低**（-2.61σ，比 EL 过净），需要更**平**（更白）的底噪。
   男女方向相反，正好可以分性别配斜率。

用法：
    python rt_slope.py            # 扫斜率
"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/diag")
from probe_g import measure, load_base  # noqa: E402

SR = 24000
F_FLOOR = 60.0
HF_KNEE = 6000.0
HF_EXTRA = 6.0

# 无底噪的链路产物（atempo + asetrate + EQ，还没叠 roomtone）
BASE_F = "/mnt/tts/diag/xf5272/v_swr_eq.wav"


def gen(n, sr, db, slope, knee_roll=HF_EXTRA, rng=None):
    """生成 RMS=db dBFS 的噪声，功率谱按 slope dB/oct 滚降。

    slope=-3 就是现在的粉噪；更负 = 能量更集中到低频 = flat 更低。
    """
    g = rng or np.random.default_rng(20261007)
    m = 1 << int(np.ceil(np.log2(max(int(n), sr))))
    W = np.fft.rfft(g.standard_normal(m))
    f = np.fft.rfftfreq(m, 1.0 / sr)
    ff = np.maximum(f, F_FLOOR)
    amp = (F_FLOOR / ff) ** (abs(slope) / 6.0)        # slope dB/oct
    over = np.log2(np.maximum(f, HF_KNEE) / HF_KNEE)
    amp = amp * 10.0 ** (-(knee_roll / 20.0) * np.maximum(over, 0.0))
    y = np.fft.irfft(W * amp, n=m).astype(np.float32)[:int(n)]
    rms = np.sqrt(np.mean(y ** 2)) + 1e-20
    return (y * (10.0 ** (db / 20.0) / rms)).astype(np.float32)


def main():
    base = load_base("female")
    x, _ = sf.read(BASE_F, dtype="float32")
    print("无底噪链路产物：flat=%.5f (z=%+.2f) voiced=%.4f"
          % (measure(BASE_F)["flat"],
             (measure(BASE_F)["flat"] - base["flat"]["mean"]) / base["flat"]["sd"],
             measure(BASE_F)["voiced"]))
    print()
    print("%-10s %10s %10s %10s" % ("斜率", "flat", "z(flat)", "voiced"))
    m0 = measure(BASE_F)
    for slope in (-3.0, -4.5, -6.0, -9.0, -1.5):
        o = x + gen(len(x), SR, -62.0, slope)
        mx = float(np.max(np.abs(o)))
        if mx > 0.999:
            o = o * (0.999 / mx)
        p = "/mnt/tts/diag/_rt_s%s.wav" % str(slope).replace(".", "_")
        sf.write(p, o.astype("float32"), SR)
        m = measure(p)
        z = (m["flat"] - base["flat"]["mean"]) / base["flat"]["sd"]
        zv = (m["voiced"] - base["voiced"]["mean"]) / base["voiced"]["sd"]
        print("%-10s %10.5f %+10.2f %+10.2f" % ("%.1f dB/oct" % slope, m["flat"], z, zv))
    print()
    print("参考：现役(-62dB, -3dB/oct) flat z=+2.89  voiced z=-1.45")
    os.system("rm -f /mnt/tts/diag/_rt_s*.wav")


if __name__ == "__main__":
    main()
