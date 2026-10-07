# -*- coding: utf-8 -*-
"""音频「开头几秒」体检：定位电流声 / 瞬间停顿。

用户反馈（2026-10-07）：r22-6cdd25 和 r22-8197d9 **开头几秒**有电流声和瞬间停顿。

查什么
------
1. **前导静音**：开头是不是先来一段静音（听感上就是「瞬间停顿」）
2. **工频哼声**：50/100/150 Hz（或 60/120/180）有没有孤立尖峰 —— 电流声典型特征
3. **包络凹陷**：前 10 秒里有没有突然掉下去又回来的洞（= 瞬间停顿）
4. **头尾对比**：头部 5 秒的谱 vs 中段 5 秒的谱，差多少 dB
   （滤波器 IIR 零状态启动 / 响度闭环爬升都会在头部留下痕迹）
5. **8~16 Hz 调制深度**：`envshape` 只放 8~16Hz，如果头部调制过量就是「颤/嗡」

用法：
    venv/bin/python tools/diag_head.py a.mp3 b.mp3 ...
"""
import os
import sys

import numpy as np
import soundfile as sf
import librosa

SR = 24000


def load(p, sr0=SR):
    x, sr = sf.read(p, dtype="float32")
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    if sr != sr0:
        x = librosa.resample(x.astype(np.float64), orig_sr=sr,
                             target_sr=sr0).astype(np.float32)
        sr = sr0
    return x, sr


def env_db(x, sr, win=0.05, hop=0.01):
    W, H = int(win * sr), int(hop * sr)
    n = max((len(x) - W) // H, 1)
    e = np.sqrt(np.array([np.mean(x[i * H:i * H + W] ** 2) for i in range(n)]) + 1e-12)
    return 20 * np.log10(e + 1e-12), H / sr


def spectrum_peaks(x, sr, fmax=400, top=6):
    """低频段的孤立尖峰（工频哼声）。返回 [(Hz, 相对该段中位 dB)]"""
    n = min(len(x), int(8 * sr))
    S = np.abs(np.fft.rfft(x[:n] * np.hanning(n))) + 1e-12
    f = np.fft.rfftfreq(n, 1.0 / sr)
    sel = (f >= 20) & (f <= fmax)
    if sel.sum() < 8:
        return []
    db = 20 * np.log10(S[sel] + 1e-20)
    med = np.median(db)
    idx = np.argsort(db)[::-1][:top]
    return [(float(f[sel][i]), float(db[i] - med)) for i in idx]


def modulation(x, sr, lo=6.0, hi=20.0):
    """包络在 lo~hi Hz 的调制深度（dB）：envshape 只放 8~16Hz。"""
    e, dt = env_db(x, sr)
    e = e - np.median(e)
    n = len(e)
    if n < 64:
        return None
    S = np.abs(np.fft.rfft((e - e.mean()) * np.hanning(n))) + 1e-12
    f = np.fft.rfftfreq(n, dt)
    sel = (f >= lo) & (f <= hi)
    if sel.sum() == 0:
        return None
    return float(20 * np.log10(S[sel].mean() + 1e-20))


def head_report(p, head_s=10.0, mid_at=0.5):
    x, sr = load(p)
    name = os.path.basename(p)
    dur = len(x) / sr
    # 1) 前导静音
    e, dt = env_db(x[:int(min(6.0, dur) * sr)], sr)
    thr = np.percentile(e[e > -120], 90) - 35 if len(e) else -60
    lead = 0.0
    for i, v in enumerate(e):
        if v > thr:
            lead = i * dt
            break
    # 2) 前 10 秒包络凹陷
    hn = int(min(head_s, dur) * sr)
    eh, dth = env_db(x[:hn], sr)
    med = np.median(eh)
    dips = []
    if len(eh) > 20:
        below = eh < (med - 20)
        # 连续 3 帧以上算一个洞
        i = 0
        while i < len(below):
            if below[i]:
                j = i
                while j < len(below) and below[j]:
                    j += 1
                if (j - i) * dth >= 0.12:
                    dips.append((round(i * dth, 2), round((j - i) * dth, 2)))
                i = j
            else:
                i += 1
    # 3) 工频哼声
    hum = spectrum_peaks(x, sr)
    # 4) 头尾谱差
    def band_ltas(y):
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
        fb = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=64, fmin=50, fmax=11000)
        M = fb @ S
        ee = M.mean(axis=0)
        sel = ee > np.percentile(ee, 40)
        p2 = M[:, sel].mean(axis=1)
        return 10 * np.log10(p2 / p2.sum() + 1e-12)
    a = band_ltas(x[:int(min(5.0, dur) * sr)])
    s0 = int(dur * mid_at)
    b = band_ltas(x[s0:s0 + int(min(5.0, dur - s0 - 1) * sr)])
    head_mid = float(np.mean(np.abs(a - b))) if a is not None and b is not None else None
    # 5) 调制深度 头 vs 中
    mh = modulation(x[:int(min(10.0, dur) * sr)], sr)
    mm = modulation(x[s0:s0 + int(min(10.0, dur - s0 - 1) * sr)], sr)
    print(f"\n=== {name}  ({dur:.1f}s) ===")
    print(f"  前导静音        : {lead:.2f} s")
    print(f"  前10s 包络洞(<-20dB): {dips if dips else '无'}")
    print(f"  20~400Hz 尖峰   : " + "  ".join(f"{f:.0f}Hz(+{d:.0f}dB)" for f, d in hum))
    print(f"  |头5s-中5s| 谱差: {('%.2f dB' % head_mid) if head_mid else '-'}")
    print(f"  8~16Hz 调制深度 : 头 {('%.1f' % mh) if mh else '-'} dB / "
          f"中 {('%.1f' % mm) if mm else '-'} dB")
    return dict(name=name, lead=lead, dips=dips, hum=hum,
                head_mid=head_mid, mod_head=mh, mod_mid=mm)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            head_report(p)
        except Exception as ex:
            print(f"\n=== {os.path.basename(p)} ===  ERR {repr(ex)[:160]}")
