#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EL 案例的频谱到底铺到多宽？决定要不要把处理链从 16k 改成 24k。

如果 EL 24kHz 音频里 >8kHz 的能量占比接近 0，那我们 16k 成品根本没损失，
改 24k 就是白折腾；反之则是砍掉了真实存在的自然质感（气声/齿音）。
"""
import subprocess
import tempfile
import os
import numpy as np
import soundfile as sr_sf
import librosa

FILES = [
    ("EL案例8197", "/mnt/tts/data/cases/8197d9ac8a35/merged.wav"),
    ("EL案例1a6c", "/mnt/tts/data/cases/1a6c20a60481/merged.wav"),
    ("EL_male参考", "/mnt/tts/data/voices/EL_male.wav"),
    ("VoxCPM2 raw(48k)", "/mnt/tts/diag/x8197/vox_male_raw.wav"),
    ("我们成品(16k)", "/mnt/tts/diag/x8197/m_v4.wav"),
]


def bands(path, sr=24000, dur=60.0):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    y, _ = sr_sf.read(t, dtype="float32")
    os.unlink(t)
    y = y[:int(dur * sr)]
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    tot = S.sum() + 1e-12
    out = {}
    for lo, hi, name in [(0, 100, "<100"), (100, 1000, "100-1k"),
                         (1000, 4000, "1k-4k"), (4000, 8000, "4k-8k"),
                         (8000, 11000, "8k-11k"), (11000, 12000, "11k-12k")]:
        m = (f >= lo) & (f < hi)
        out[name] = round(float(S[m].sum() / tot), 5)
    out[">8k合计"] = round(out["8k-11k"] + out["11k-12k"], 5)
    return out


if __name__ == "__main__":
    keys = ["<100", "100-1k", "1k-4k", "4k-8k", "8k-11k", "11k-12k", ">8k合计"]
    print("文件\t" + "\t".join(keys))
    for n, p in FILES:
        try:
            r = bands(p)
            print(n + "\t" + "\t".join(str(r[k]) for k in keys))
        except Exception as e:
            print("FAIL", n, repr(e)[:100])
