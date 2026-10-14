# -*- coding: utf-8 -*-
"""flat 分段诊断：flat 的差距到底来自**语音帧**还是**静音帧**？

## 为什么必须先分清（2026-10-14）
noise_floor.py 显示加 -26dB 粉噪能把 flat 从 -3.85σ 拉到 -1.56σ，
但 -26dB 是能听见的嘶嘶声，不能这么补。
于是要先问：EL 的高 flat(0.0461) 是哪来的？
  假设 A：EL 的**语音帧**更"沙"（声门噪声/气声多） -> 只能改合成端，后处理无解
  假设 B：EL 的**静音帧**是真底噪，我们的是数字静音（flat=0）-> 补 room tone 即可
两个假设的解法完全不同，必须先测。

## 口径
16k / n_fft=2048 / hop=256，逐帧算 RMS 与 spectral_flatness，
按帧能量分位切成「发声帧」(top40%) 与「静音帧」(bottom25%)，分别取均值。
"""
import os
import sys

import librosa
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eq_search6 as E  # noqa: E402

TARGETS = [
    ("EL8197", "/mnt/tts/data/cases/8197d9ac8a35/merged.wav"),
    ("EL1a6c", "/mnt/tts/data/cases/1a6c20a60481/merged.wav"),
    ("base_pw(无EQ)", "base_pw.wav"),
    ("v10", "m_v10.wav"),
    ("v11", "m_v11.wav"),
    ("PSOLA_v2", "m_psola_v2.wav"),
]


def frames(path, dur=120.0):
    y16, _ = E.load16(path)
    n = int(dur * 16000)
    y = y16[:min(len(y16), n)]
    # 🔴 必须传**幅度谱**给 spectral_flatness：E.measure / el_baseline2 用的是
    #    spectral_flatness(y=...)，内部取 np.abs(stft) 即幅度；传功率谱(S**2)
    #    会让 flat 整体小一到两个数量级，跟基线不可比（第一次跑就踩了这个坑）。
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
    flat = librosa.feature.spectral_flatness(S=S).ravel()
    rms = np.sqrt((S ** 2).mean(axis=0))
    db = 20 * np.log10(rms + 1e-12)
    return flat, db


def main():
    print("%-16s %8s | %8s %8s %8s | %8s %8s"
          % ("文件", "flat总", "发声flat", "发声dB", "占比",
             "静音flat", "静音dB"))
    print("-" * 78)
    for name, p in TARGETS:
        p = p if os.path.isabs(p) else os.path.join(HERE, p)
        if not os.path.exists(p):
            print("%-16s 缺失" % name)
            continue
        flat, db = frames(p)
        hi = np.percentile(db, 60)
        lo = np.percentile(db, 25)
        v = flat[db >= hi]
        q = flat[db <= lo]
        print("%-16s %8.4f | %8.4f %8.1f %7.1f%% | %8.4f %8.1f"
              % (name, flat.mean(), v.mean(), db[db >= hi].mean(),
                 100.0 * (db >= hi).mean(), q.mean(), db[db <= lo].mean()))


if __name__ == "__main__":
    main()
