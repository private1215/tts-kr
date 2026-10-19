# -*- coding: utf-8 -*-
"""measure53.py —— 全量客观指标对照（我们 vs EL 同文本）。

## 为什么

员工下班、z30 待反馈期间，按「能确定的因素」继续优化：
只动**有 EL 同文本对照、且我们确实更差**的指标。

## 口径（严格按项目铁律）

- click：**先归一到峰值 0.70**、**无条件重采样到 16k**，判 |Δ|>=0.30
- 质心 / 中频(1k-3k)占比：只算发声帧（>-40dB）
- 停顿：-40dB 分段后静音段时长（中位 / 变异CV = MAD*1.4826/中位）
- 杂音：静音段的 4-8kHz 能量占比
- cps：由外部传入字数（同文本，字数来自 full.txt）

用法：python3 measure53.py <ours.wav> <el.wav> [字数]
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
    return x


def measure(p, nchars=None):
    x = load16(p).astype(np.float64)
    dur = len(x) / SR
    # 归一到峰值 0.70（click 铁律）
    pk = np.max(np.abs(x))
    xn = x * (0.70 / pk) if pk > 0 else x
    d = np.abs(np.diff(xn))
    click = float((d >= 0.30).mean())

    # 分帧
    n = int(FRAME * SR)
    nfr = len(x) // n
    fr = x[:nfr * n].reshape(nfr, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > VTH

    # 谱质心 + 中频占比（发声帧）
    sp = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    cents, mids = [], []
    for i in np.where(voiced)[0]:
        s = sp[i]
        tot = s.sum() + 1e-12
        cents.append(float((freqs * s).sum() / tot))
        mids.append(float(s[(freqs >= 1000) & (freqs <= 3000)].sum() / tot))
    centroid = float(np.median(cents)) if cents else 0.0
    mid_ratio = float(np.median(mids)) if mids else 0.0

    # 停顿（静音段）
    v = voiced
    sil, i = [], 0
    while i < len(v):
        if not v[i]:
            j = i
            while j < len(v) and not v[j]:
                j += 1
            sil.append((j - i) * FRAME)
            i = j
        else:
            i += 1
    sil = [s for s in sil if s > 0.02]
    if sil:
        med = float(np.median(sil))
        mad = float(np.median(np.abs(np.array(sil) - med)))
        cv = (mad * 1.4826 / med) if med > 0 else 0.0
    else:
        med, cv = 0.0, 0.0

    # 杂音：静音段的 4-8k 占比
    noise = []
    for i in range(len(v)):
        if not v[i]:
            s = sp[i]
            tot = s.sum() + 1e-12
            noise.append(float(s[(freqs >= 4000) & (freqs <= 8000)].sum()
                               / tot))
    noise_ratio = float(np.median(noise)) if noise else 0.0

    cps = (nchars / dur) if nchars else None
    print("=== %s  %.1fs ===" % (p.split("/")[-1], dur))
    print("  click%%       %.4f" % (click * 100))
    print("  谱质心 Hz    %.0f" % centroid)
    print("  中频1k-3k%%   %.1f" % (mid_ratio * 100))
    print("  停顿中位 s   %.3f   变异CV %.2f   停顿数 %d" % (med, cv, len(sil)))
    print("  静音段杂音%%  %.1f" % (noise_ratio * 100))
    if cps:
        print("  cps          %.2f" % cps)
    print()
    return dict(click=click, centroid=centroid, mid=mid_ratio,
                pause_med=med, pause_cv=cv, n_pause=len(sil),
                noise=noise_ratio, dur=dur)


def main():
    if len(sys.argv) < 3:
        print("用法：measure53.py <ours.wav> <el.wav> [字数]")
        return 1
    nchars = float(sys.argv[3]) if len(sys.argv) > 3 else None
    a = measure(sys.argv[1], nchars)
    b = measure(sys.argv[2], nchars)
    print("=" * 46)
    print("%-16s %10s %10s %8s" % ("指标", "我们", "EL", "我们/EL"))
    for k, lab in [("click", "click%"), ("centroid", "谱质心Hz"),
                   ("mid", "中频1k-3k"), ("pause_med", "停顿中位s"),
                   ("noise", "静音杂音")]:
        r = (a[k] / b[k]) if b[k] else float("nan")
        print("%-16s %10.4f %10.4f %8.2f" % (lab, a[k], b[k], r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
