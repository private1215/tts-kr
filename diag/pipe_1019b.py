# -*- coding: utf-8 -*-
"""pipe_1019b.py —— 在**当前生产链**男声产物上离线复现 pipeline53，看还灵不灵。

背景：pipeline53（LTAS → 53b → 53c → clickfix）的全部实测都是在**旧的
voxcpm 链路**上做的。现役男声已换成 omnivoice + 极简后处理，基线变了
（click 1.39 → 1.21、杂音 0.92 → 0.98、中频仍是 0.84）。
不重新验一遍就照搬旧结论，是拿旧地图找新路。

本脚本只产出**离线对照产物**，不碰生产代码。测完把数据摆出来由人拍板。

用法: venv/bin/python -u /mnt/tts/diag/pipe_1019b.py
"""
import os
import subprocess
import sys

PY = sys.executable
T = "/mnt/tts/projects/qwen3-tts/tools"
D = "/mnt/tts/diag"
SRC = D + "/mcur1019.mp3"          # 当前生产链成品（192k mp3）
EL = "/mnt/tts/data/cases/d172b8c4efb3/merged.wav"
W0 = D + "/mcur1019.wav"           # 解回 wav 供管线吃
W1 = D + "/mcur1019_ltas.wav"
W2 = D + "/mcur1019_b.wav"
W3 = D + "/mcur1019_c.wav"
W4 = D + "/mcur1019_pipe.wav"
MP3 = D + "/mcur1019_pipe.mp3"


def run(a, **kw):
    print("$ " + " ".join(a), flush=True)
    subprocess.run(a, check=True, **kw)


def main():
    # 0) mp3 -> wav（保持 44100，与生产编码前一致）
    run(["ffmpeg", "-y", "-v", "error", "-i", SRC, "-ac", "1", W0])

    # 1) LTAS 全频段对齐（必须在清理前：firequalizer 是 FIR，会抹出振铃碎片）
    print("== 1/4 ltasmatch53 ==", flush=True)
    run([PY, "-u", T + "/ltasmatch53.py", W0, EL, W1])

    print("== 2/4 tailclean53b（句末粘连）==", flush=True)
    run([PY, "-u", T + "/tailclean53b.py", W1, W2])

    print("== 3/4 tailclean53c（句首紧贴碎片）==", flush=True)
    run([PY, "-u", T + "/tailclean53c.py", W2, W3])

    print("== 4/4 clickfix53（男声工作点 th=0.30）==", flush=True)
    run([PY, "-u", T + "/clickfix53.py", W3, W4, "--th", "0.30",
         "--rounds", "3"])

    # 5) 压回生产同款 mp3 再测（铁律：必须测压完 mp3 后的值）
    run(["ffmpeg", "-y", "-v", "error", "-i", W4, "-codec:a", "libmp3lame",
         "-b:a", "192k", MP3])
    print("\n成品 ->", MP3, flush=True)

    print("== 指标：管线后 vs EL ==", flush=True)
    subprocess.run([PY, "-u", T + "/measure53.py", MP3, EL, "4594"])
    print("== 指标：管线前 vs EL ==", flush=True)
    subprocess.run([PY, "-u", T + "/measure53.py", SRC, EL, "4594"])


if __name__ == "__main__":
    sys.exit(main() or 0)
