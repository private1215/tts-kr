# -*- coding: utf-8 -*-
"""pipe_8197_g.py —— **第二文案泛化验证**：EQ-only 链在 8197d9ac8a35 上重跑。

## 为什么
EQ-only 链（eqfix53b +2dB → clickfix53b）的全部结论目前只在
d172b8c4efb3 一条文案上。项目铁律：**新链路必须过第二条文案的泛化**，
否则不接生产。

## 素材
- 我们：`m8197.wav` —— 现役 v3.1 生产链重跑 8197d9ac8a35 全文（4489 字）
  的产物，549.65s @ 44100Hz（生产只出 mp3，这里已转 wav）
- EL  ：`/mnt/tts/data/cases/8197d9ac8a35/merged.wav`

## 验收
  · 中频 0.86 → 接近 1.0（第一条上 +2dB 给 1.01）
  · click 不得明显恶化（第一条上 +2dB 给 0.11×EL）
  · 音色不掉（第一条上现役 0.9331）

🔴 全程 44.1 kHz，最后才压 mp3；click/杂音必须**压完 mp3 后**测。

用法: venv/bin/python -u /mnt/tts/diag/pipe_8197_g.py
"""
import subprocess
import sys

PY = sys.executable
T = "/mnt/tts/projects/qwen3-tts/tools"
D = "/mnt/tts/diag"
EL = "/mnt/tts/data/cases/8197d9ac8a35/merged.wav"
SRC = D + "/m8197.wav"
W0 = D + "/i_a.wav"
W1 = D + "/i_cf.wav"
MP3 = D + "/i_cf.mp3"
CHARS = "4489"


def run(a):
    print("$ " + " ".join(a), flush=True)
    subprocess.run(a, check=True)


def main():
    print("== 1/2 eqfix53b（中频 EQ +2dB，保采样率）==", flush=True)
    run([PY, "-u", T + "/eqfix53b.py", SRC, EL, W0, "--gain", "2"])
    print("== 2/2 clickfix53b（保采样率 th=0.30）==", flush=True)
    run([PY, "-u", T + "/clickfix53b.py", W0, W1, "--th", "0.30",
         "--rounds", "3"])
    run(["ffmpeg", "-y", "-v", "error", "-i", W1, "-codec:a",
         "libmp3lame", "-b:a", "192k", MP3])
    print("\n成品 -> %s" % MP3, flush=True)

    print("\n===== 客观指标 vs EL（压完 mp3 后测）=====", flush=True)
    subprocess.run([PY, "-u", T + "/measure53.py", MP3, EL, CHARS])
    print("\n===== 句末/句首残留（EQ-only 不清理，应与现役同档）=====",
          flush=True)
    subprocess.run([PY, "-u", T + "/tailclean53b2.py", MP3,
                    "/tmp/_i.wav", "--dry"])
    subprocess.run([PY, "-u", T + "/tailclean53c2.py", MP3, "--dry"])
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
