# -*- coding: utf-8 -*-
"""pipe_1019c.py —— **保采样率**版男声优化管线（离线对照，不碰生产）。

与 pipeline53 的差别：LTAS 与 clickfix 都换成保采样率版本
（ltasmatch53b / clickfix53b），成品维持输入采样率（44.1k）。

目的：验证「中频 0.84 → 1.05」这个收益，能否在**不毁音色**的前提下拿到。
53 版拿到了收益但同时把成品降到 16k，压 mp3 后音色 0.9331 → 0.6903，
属于典型的「指标好看、交付变坏」。

用法: venv/bin/python -u /mnt/tts/diag/pipe_1019c.py
"""
import subprocess
import sys

PY = sys.executable
T = "/mnt/tts/projects/qwen3-tts/tools"
D = "/mnt/tts/diag"
EL = "/mnt/tts/data/cases/d172b8c4efb3/merged.wav"
SRC = D + "/mcur1019.wav"          # 44.1k
W1 = D + "/n_ltas.wav"
W2 = D + "/n_b.wav"
W3 = D + "/n_c.wav"
W4 = D + "/n_cf.wav"
MP3 = D + "/n_cf.mp3"


def run(a):
    print("$ " + " ".join(a), flush=True)
    subprocess.run(a, check=True)


def main():
    print("== 1/4 ltasmatch53b（保采样率 LTAS 对齐）==", flush=True)
    run([PY, "-u", T + "/ltasmatch53b.py", SRC, EL, W1])

    print("== 2/4 tailclean53b ==", flush=True)
    run([PY, "-u", T + "/tailclean53b.py", W1, W2])

    print("== 3/4 tailclean53c ==", flush=True)
    run([PY, "-u", T + "/tailclean53c.py", W2, W3])

    print("== 4/4 clickfix53b（保采样率，th=0.30）==", flush=True)
    run([PY, "-u", T + "/clickfix53b.py", W3, W4, "--th", "0.30",
         "--rounds", "3"])

    run(["ffmpeg", "-y", "-v", "error", "-i", W4, "-codec:a", "libmp3lame",
         "-b:a", "192k", MP3])
    print("\n成品 ->", MP3, flush=True)

    for tag, f in (("管线前", D + "/mcur1019.mp3"),
                   ("保采样率管线后", MP3),
                   ("16k 管线后", D + "/mcur1019_pipe.mp3")):
        print("== %s vs EL ==" % tag, flush=True)
        subprocess.run([PY, "-u", T + "/measure53.py", f, EL, "4594"])


if __name__ == "__main__":
    sys.exit(main() or 0)
