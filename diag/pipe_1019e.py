# -*- coding: utf-8 -*-
"""pipe_1019e.py —— 保采样率**完整链**离线验证：LTAS → 53b2 → 53c2 → clickfix53b。

与 pipe_1019d 的差别：**最前面加了 ltasmatch53b**（全频段频谱对齐 EL）。
这是确定性优化（对齐标杆，不依赖听感），用来补男声剩下的 **中频 0.86** 差距
（LTAS 实测能修到 ≈1.05）。

🔴 顺序铁律（文档 §1.2）：**频谱类处理必须在前、清理在后**。
firequalizer 是 FIR，有振铃/时域涂抹，会在干净静音里「抹」出 -44dB/5ms 碎片
（实测句末 2→8、碎片 7→11）。反过来做等于边清边造。

全程 44.1 kHz，最后才压 192k mp3（🔴 绝不能在 16 kHz 上出成品）。

验证四件事：
  · 中频 0.84 → 接近 1.05
  · 音色不掉（现役 0.9331；d 链已得 0.9350）
  · click 不恶化（现役 0.05×EL）
  · 句末/句首残留（本次是全清版，预期过清，数据与 f 链的保守版对照）
🔴 click / 杂音必须**压完 mp3 后**测（交付就是 mp3）。

用法: venv/bin/python -u /mnt/tts/diag/pipe_1019e.py
"""
import subprocess
import sys

PY = sys.executable
T = "/mnt/tts/projects/qwen3-tts/tools"
D = "/mnt/tts/diag"
EL = "/mnt/tts/data/cases/d172b8c4efb3/merged.wav"
SRC = D + "/mcur1019.wav"        # 44.1k，当前生产链产物
W0 = D + "/e_a.wav"              # LTAS 后
W1 = D + "/e_b.wav"              # + 句末清理
W2 = D + "/e_c.wav"              # + 句首清理
W3 = D + "/e_cf.wav"             # + clickfix
MP3 = D + "/e_cf.mp3"


def run(a):
    print("$ " + " ".join(a), flush=True)
    subprocess.run(a, check=True)


def main():
    print("== 0/4 ltasmatch53b（全频段频谱对齐 EL）==", flush=True)
    run([PY, "-u", T + "/ltasmatch53b.py", SRC, EL, W0, "--iters", "3"])
    print("== 1/4 tailclean53b2（句末粘连嗯）==", flush=True)
    run([PY, "-u", T + "/tailclean53b2.py", W0, W1])
    print("== 2/4 tailclean53c2（句首紧贴碎片）==", flush=True)
    run([PY, "-u", T + "/tailclean53c2.py", W1, W2])
    print("== 3/4 clickfix53b（保采样率 th=0.30）==", flush=True)
    run([PY, "-u", T + "/clickfix53b.py", W2, W3, "--th", "0.30",
         "--rounds", "3"])
    run(["ffmpeg", "-y", "-v", "error", "-i", W3, "-codec:a", "libmp3lame",
         "-b:a", "192k", MP3])
    print("\n成品 -> %s" % MP3, flush=True)

    print("\n===== 清理后残留 =====", flush=True)
    for tag, f in (("清理后", MP3), ("EL", EL), ("清理前", D + "/mcur1019.mp3")):
        print("-- %s --" % tag, flush=True)
        subprocess.run([PY, "-u", T + "/tailclean53b2.py", f,
                        "/tmp/_e.wav", "--dry"])
        subprocess.run([PY, "-u", T + "/tailclean53c2.py", f, "--dry"])

    print("\n===== 客观指标 vs EL（压完 mp3 后测）=====", flush=True)
    subprocess.run([PY, "-u", T + "/measure53.py", MP3, EL, "4594"])
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
