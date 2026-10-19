# -*- coding: utf-8 -*-
"""pipe_1019d.py —— **保采样率版「嗯/尾音」清理链**离线验证（不碰生产）。

链路（顺序不能换，文档 §1.2 铁律）：
  ① tailclean53b2（句末粘连「嗯」）—— 后处理里的 FIR 滤波会「抹」出新碎片，
     所以**频谱类处理必须在前、清理在后**；本链不跑 LTAS（LTAS 收益另议）
  ② tailclean53c2（句首紧贴碎片）
  ③ clickfix53b（已在生产，男声 th=0.30）
全程 44.1 kHz，最后才压 192k mp3。

验证四件事（缺一不可）：
  · 句首碎片 15 → 接近 EL 的 2
  · 句末粘连 56 → 接近 EL 的 36（不是清零）
  · 音色不掉（基线 0.9331 / 现役 0.9336）
  · click 不恶化（现役 0.05×EL）
    🔴 铁律：click / 杂音必须在**压完 mp3 后**测，交付就是 mp3

用法: venv/bin/python -u /mnt/tts/diag/pipe_1019d.py
"""
import subprocess
import sys

PY = sys.executable
T = "/mnt/tts/projects/qwen3-tts/tools"
D = "/mnt/tts/diag"
EL = "/mnt/tts/data/cases/d172b8c4efb3/merged.wav"
SRC = D + "/mcur1019.wav"        # 44.1k，当前生产链产物
W1 = D + "/g_b.wav"
W2 = D + "/g_c.wav"
W3 = D + "/g_cf.wav"
MP3 = D + "/g_cf.mp3"


def run(a):
    print("$ " + " ".join(a), flush=True)
    subprocess.run(a, check=True)


def main():
    print("== 1/3 tailclean53b2（句末粘连「嗯」）==", flush=True)
    run([PY, "-u", T + "/tailclean53b2.py", SRC, W1])
    print("== 2/3 tailclean53c2（句首紧贴碎片）==", flush=True)
    run([PY, "-u", T + "/tailclean53c2.py", W1, W2])
    print("== 3/3 clickfix53b（保采样率 th=0.30）==", flush=True)
    run([PY, "-u", T + "/clickfix53b.py", W2, W3, "--th", "0.30",
         "--rounds", "3"])
    run(["ffmpeg", "-y", "-v", "error", "-i", W3, "-codec:a", "libmp3lame",
         "-b:a", "192k", MP3])
    print("\n成品 -> %s" % MP3, flush=True)

    print("\n===== 清理后残留（目标：贴近 EL，不是清零）=====", flush=True)
    for tag, f in (("清理后", MP3), ("EL", EL), ("清理前", D + "/mcur1019.mp3")):
        print("-- %s --" % tag, flush=True)
        subprocess.run([PY, "-u", T + "/tailclean53b2.py", f,
                        "/tmp/_d.wav", "--dry"])
        subprocess.run([PY, "-u", T + "/tailclean53c2.py", f, "--dry"])

    print("\n===== 客观指标 vs EL（压完 mp3 后测）=====", flush=True)
    subprocess.run([PY, "-u", T + "/measure53.py", MP3, EL, "4594"])


if __name__ == "__main__":
    sys.exit(main() or 0)
