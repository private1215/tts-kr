# -*- coding: utf-8 -*-
"""女声「滋滋滋」量化：静音段 4-8kHz 能量占比。
🔴 口径：统一降到 16k 再比（16k 源与 24k 源的 4-8k 占比天然不同：
   16k 的 Nyquist=8k，4-8k 是最高频段，占比虚高；
   若统一升到 24k 去比，等于拿不同有效带宽的材料对比 —— 不公平）。（取最安静 20% 帧 -> 拼整段 -> 4~8k 能量占比%）。
   我自己另写的 dB 版口径不同，-83.5 那种数字不能与这里的 11~15 混用。
"""
import sys
import numpy as np, soundfile as sf

SR = 16000

def band(x, lo, hi):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    tot = X.sum()
    m = (f >= lo) & (f < hi)
    return float(X[m].sum() / tot * 100)

def hiss(x):
    n = int(SR * 0.02)
    m = (len(x) // n) * n
    fr = x[:m].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    thr = np.percentile(rms, 20)
    seg = fr[rms <= thr].ravel()
    if len(seg) < SR:
        return float("nan")
    return band(seg, 4000, 8000)

def load(p):
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import subprocess, os, tempfile
        t = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                        "-ar", str(SR), "-ac", "1", t], check=True)
        x, sr = sf.read(t, dtype="float32")
        os.remove(t)
    return x

print(f"{'样本':<34} {'时长':>8} {'静音4-8k%':>11}")
for p in sys.argv[1:]:
    x = load(p)
    print(f"{p.split('/')[-1]:<34} {len(x)/SR:>8.1f} {hiss(x):>11.3f}")
