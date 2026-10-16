#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L1 快筛：裸模型输出的平坦度是不是本来就低（第30轮）

前面已经排掉了：
  · 频段能量分布（我们 mp3 vs EL mp3 各频段都在 ±13% 内）→ 不是"缺某个频段"
  · 重采样器质量（换 soxr 数值一模一样）            → 不是重采样器问题
  · 两次 atempo（占 5.4%，且男声 R≈1 时只走一次）    → 不是主因

剩下的嫌疑只剩两个：**裸模型本身太"干净"**，或 **mp3 把我们的高频细节压掉**。
（反常现象：EL 过 mp3 平坦度 0.04928→0.04931 一动不动，我们却掉 36%。）

本实验：出一条 30~40s 的**裸模型**男声，量
  裸 wav → 过 mp3 → 再和 EL 同长度片段比。

## 用法
    ./venv/bin/python tools/raw_flat.py [--chars 320]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
DIAG = "/mnt/tts/diag/muffle"
OUT = os.path.join(DIAG, "raw_flat.json")
EL = os.path.join(DIAG, "el24.wav")

try:
    import librosa
    import soundfile as sf
except Exception as e:
    print("缺依赖:", e)
    raise


def flat16(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1",
                          "-ar", "16000", "-f", "f32le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    y = np.frombuffer(raw, dtype=np.float32)
    return float(librosa.feature.spectral_flatness(
        y=y[:60 * 16000], hop_length=256)[0].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=320)
    a = ap.parse_args()

    txt = open("/mnt/tts/data/cases/6cdd25bc75f2/full.txt",
               encoding="utf-8").read().strip()[:a.chars]
    print("文本 %d 字：%s…" % (len(txt), txt[:40]))

    import vox_client
    audio, sr, meta = vox_client.synth(text=txt, voice="male")
    if audio is None:
        print("vox 合成失败:", meta)
        return
    print("裸输出 %.1fs @ %d Hz" % (len(audio) / float(sr), sr))

    d = "/mnt/tts/diag"
    raw_wav = os.path.join(d, "raw_flat_vox.wav")
    raw_mp3 = os.path.join(d, "raw_flat_vox.mp3")
    sf.write(raw_wav, audio, sr)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw_wav,
                    "-codec:a", "libmp3lame", "-b:a", "192k", raw_mp3],
                   check=True)

    # EL 同长度片段
    dur = len(audio) / float(sr)
    el_seg = os.path.join(d, "raw_flat_el.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", EL, "-t", str(dur),
                    "-ar", str(sr), "-ac", "1", "-f", "wav", el_seg],
                   check=True)
    el_mp3 = os.path.join(d, "raw_flat_el.mp3")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", el_seg,
                    "-codec:a", "libmp3lame", "-b:a", "192k", el_mp3],
                   check=True)

    res = {
        "裸模型 wav": flat16(raw_wav),
        "裸模型 mp3": flat16(raw_mp3),
        "EL 同长 wav": flat16(el_seg),
        "EL 同长 mp3": flat16(el_mp3),
    }
    print()
    for k, v in res.items():
        print("%-16s %.5f" % (k, v))
    print("\n裸模型 wav vs EL wav：%.2f 倍" % (res["EL 同长 wav"] / res["裸模型 wav"]))
    print("mp3 造成的损失：我们 %.1f%% / EL %.1f%%"
          % ((1 - res["裸模型 mp3"] / res["裸模型 wav"]) * 100,
             (1 - res["EL 同长 mp3"] / res["EL 同长 wav"]) * 100))

    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("\n→ " + OUT)


if __name__ == "__main__":
    main()
