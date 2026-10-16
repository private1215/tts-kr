# -*- coding: utf-8 -*-
"""简易 F0 中位（自相关）。只为回答「女声到底需不需要变调」这一个二选一问题，
不追求精度：三条同源材料用同一口径比即可。
"""
import sys
import numpy as np, soundfile as sf, subprocess, os, tempfile

SR = 16000
LO, HI = 80.0, 400.0          # 女声搜索范围

def load(p):
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        t = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                        "-ar", str(SR), "-ac", "1", t], check=True)
        x, sr = sf.read(t, dtype="float32")
        os.remove(t)
    return x

def f0_med(x):
    n = int(0.05 * SR)          # 50ms 帧
    hop = n // 2
    m = (len(x) // n) * n
    fr = x[:m].reshape(-1, n) if False else None
    m = ((len(x) - n) // hop) * hop + n
    frames = np.array([x[i:i+n] for i in range(0, m - n + 1, hop)])
    rms = np.sqrt((frames**2).mean(axis=1))
    thr = np.percentile(rms, 50)          # 只取较响的一半（发声帧）
    lo_lag, hi_lag = int(SR/HI), int(SR/LO)
    out = []
    for f in frames[rms >= thr][:3000]:
        f = f - f.mean()
        if np.abs(f).max() < 1e-4:
            continue
        # 归一化自相关
        ac = np.correlate(f, f, mode="full")[len(f)-1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg = ac[lo_lag:hi_lag]
        if len(seg) == 0:
            continue
        k = int(np.argmax(seg))
        if seg[k] < 0.3:         # 相关性太弱 -> 清音，跳过
            continue
        out.append(SR / (lo_lag + k))
    return float(np.median(out)) if out else float("nan")

print(f"{'样本':<28}{'F0(中位Hz)':>12}")
for p in sys.argv[1:]:
    x = load(p)
    print(f"{p.split('/')[-1]:<28}{f0_med(x):>12.1f}")
