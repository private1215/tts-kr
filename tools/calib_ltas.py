# -*- coding: utf-8 -*-
"""fidelity.py 的口径校准：找出与历史文档一致的 LTAS / 静音占比定义。

校准目标（来自 docs 与 server.py 注释，r20 配置）：
  标杆男 1a6c20 的「音色 LTAS」偏差 ≈ 0.36 dB
  案例静音占比 ≈ 0.188（docs/r18 §7）
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fidelity import measure, ltas_db, load, sent_windows  # noqa: E402

CASE = "/mnt/tts/data/cases/1a6c20a60481/merged.wav"
PROD = "/tmp/eval5work/1a6c20.wav"


def band_dev(a, b, fa, fb, fmax=11000.0, mode="none"):
    ma = (fa >= 50) & (fa <= fmax)
    mb = (fb >= 50) & (fb <= fmax)
    x, y = a[ma].copy(), b[mb].copy()
    if mode == "none":
        pass
    elif mode == "mean11k":
        x = x - x.mean()
        y = y - y.mean()
    elif mode == "meanall":
        x = a - a.mean()
        y = b - b.mean()
        x, y = x[ma], y[mb]
    elif mode == "total":
        pa = 10 ** (a / 10)
        pb = 10 ** (b / 10)
        x = 10 * np.log10(pa[ma] / pa.sum())
        y = 10 * np.log10(pb[mb] / pb.sum())
    elif mode == "meanall_then11k":
        x = (a - a.mean())[ma]
        y = (b - b.mean())[mb]
    if len(x) != len(y):
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
    return float(np.mean(np.abs(x - y)))


def main():
    cm = measure(CASE, "CASE")
    pm = measure(PROD, "PROD")
    ca = np.array(cm["ltas"])
    pa = np.array(pm["ltas"])
    fa = np.array(cm["mel_f"])
    fb = np.array(pm["mel_f"])
    print("mel 频段数", len(ca), "fmax", fa[-1])
    print("\nLTAS 口径扫描（目标 ≈ 0.36 dB）")
    for mode in ["none", "mean11k", "meanall", "meanall_then11k", "total"]:
        print(f"  {mode:<18} {band_dev(ca, pa, fa, fb, mode=mode):.3f} dB")
    print("  分 fmax:")
    for fmax in (8000, 11000, 12000):
        for mode in ("meanall", "total"):
            print(f"    fmax={fmax} {mode:<10} "
                  f"{band_dev(ca, pa, fa, fb, fmax=fmax, mode=mode):.3f} dB")

    # 静音占比口径
    print("\n静音占比（目标：案例 ≈ 0.188）")
    for tag, p in (("CASE", CASE), ("PROD", PROD)):
        x, sr = load(p, sr0=24000)
        segs, cuts, hop = sent_windows(x, sr)
        hop2, win2 = int(0.02 * sr), int(0.05 * sr)
        n2 = (len(x) - win2) // hop2
        idx = np.arange(n2) * hop2
        rms = np.sqrt(np.array([np.mean(x[i:i + win2] ** 2) for i in idx]) + 1e-12)
        db = 20 * np.log10(rms + 1e-12)
        frame_sil = float((db < (np.percentile(db, 25) + 6.0)).mean())
        pause_tot = sum((j - i) for i, j in cuts) / sr
        all_pause = sum((j - i) for i, j in cuts)
        print(f"  {tag}: 帧级VAD {frame_sil:.4f} | cuts总时长占比 "
              f"{pause_tot / (len(x) / sr):.4f} | pauses={len(cuts)} "
              f"dur={len(x)/sr:.1f}")
        # 另一种：直接对 25ms 帧用绝对阈值
        for th in (-40, -45, -50):
            print(f"        绝对阈值 {th} dB -> {float((db < th).mean()):.4f}")


if __name__ == "__main__":
    main()
