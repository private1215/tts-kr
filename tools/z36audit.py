#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""z36audit - 审查 z36 盲听结果：为什么 A/B 听不出、而 vs EL 听得出？

z36 投票（admin，控制题 C = 有效）：
  4 题 v3.6 vs v3.5 对照 → 全 C（听不出区别）
  m_el  v3.6 vs EL 标杆  → B（EL 更干净）
本脚本只测**唯一有判别力的那一组**（m_el 的 A/B 素材），看客观上差在哪，
给下一步定位一个方向，而不是继续在数字零上加码。
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

SR = 24000          # 素材是 24k wav


def load(path):
    out = tempfile.mktemp(suffix=".raw")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    "-ac", "1", "-ar", str(SR), "-c:a", "pcm_f32le",
                    "-f", "f32le", out], check=True)
    y = np.fromfile(out, dtype="<f4").astype(np.float32)
    os.remove(out)
    return y


def band_ratio(y, sr, lo, hi):
    """[lo,hi) 能量占全频段比例（同采样率才可跨样本比）。"""
    S = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    tot = S.sum() + 1e-20
    return float(S[(f >= lo) & (f < hi)].sum() / tot)


def quiet_band(y, sr, lo, hi, thr_db=-45.0):
    """静音段（帧 RMS < thr）里的频段能量占比 = 杂音/底噪指标。"""
    w = int(0.02 * sr)
    n = len(y) // w
    if n < 5:
        return float("nan")
    fr = y[:n * w].reshape(n, w)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    idx = np.where(db < thr_db)[0]
    if len(idx) < 5:
        return float("nan")
    seg = fr[idx].reshape(-1)
    S = np.abs(np.fft.rfft(seg)) ** 2
    f = np.fft.rfftfreq(len(seg), 1.0 / sr)
    return float(S[(f >= lo) & (f < hi)].sum() / (S.sum() + 1e-20))


def click_density(y, peak=0.70):
    """峰值归一到 peak 后，相邻样本 |Δ|>=0.30 的比例（规范口径）。"""
    m = np.abs(y).max() + 1e-20
    z = y * (peak / m)
    d = np.abs(np.diff(z))
    return float((d >= 0.30).mean() * 100.0)


def row(tag, p):
    y = load(p)
    db = 20 * np.log10(np.abs(y) + 1e-12)
    q = db < -45
    print("%-16s dur=%5.1fs  4-8k=%.5f  8-16k=%.6f  静音4-8k=%.5f  "
          "click=%.3f%%  peak=%.1fdB"
          % (tag, len(y) / SR, band_ratio(y, SR, 4000, 8000),
             band_ratio(y, SR, 8000, 16000), quiet_band(y, SR, 4000, 8000),
             click_density(y), db.max()))
    return dict(b48=band_ratio(y, SR, 4000, 8000),
                b816=band_ratio(y, SR, 8000, 16000),
                q48=quiet_band(y, SR, 4000, 8000), ck=click_density(y))


def main():
    fb = "/mnt/tts/fb"
    pairs = [("m_el", "z36_a_m_el.wav", "z36_b_m_el.wav"),
             ("m_noise", "z36_a_m_noise.wav", "z36_b_m_noise.wav"),
             ("f_noise", "z36_a_f_noise.wav", "z36_b_f_noise.wav")]
    for name, fa, fb_ in pairs:
        pa, pb = os.path.join(fb, fa), os.path.join(fb, fb_)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print("缺素材", name)
            continue
        print("\n=== %s  (A=v3.6 / B=%s) ===" % (name, "EL" if name == "m_el" else "v3.5"))
        a = row("A v3.6", pa)
        b = row("B " + ("EL" if name == "m_el" else "v3.5"), pb)
        print("   倍率 A/B:  4-8k %.2f×  8-16k %.2f×  静音4-8k %.2f×  click %.2f×"
              % (a["b48"] / max(b["b48"], 1e-9), a["b816"] / max(b["b816"], 1e-9),
                 a["q48"] / max(b["q48"], 1e-9), a["ck"] / max(b["ck"], 1e-9)))


if __name__ == "__main__":
    main()
