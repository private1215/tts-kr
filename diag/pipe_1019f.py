# -*- coding: utf-8 -*-
"""pipe_1019f.py —— 保采样率**窄带中频 EQ**版完整链（对照 pipe_1019e 的全频段 LTAS）。

链路：eqfix53b（只动 1k-3k）→ tailclean53b2 → tailclean53c2 → clickfix53b

## 为什么要有这条（e 链的教训）
pipe_1019e 用**全频段 LTAS** 把中频从 0.84 修到 1.05 ✅，
但 click 从现役的 **0.05×EL 恶化到 1.72×EL** ❌。
click 是**唯一与员工评分单调对应的客观指标**，且现役本来已比 EL 好 20 倍
→ 按铁律「已比 EL 好的不动」，这个代价不能接受。

→ 换 eqfix53b：单条窄带 firequalizer（1k:0 / 1.8k:+G / 3k:0），FIR 涂抹量小得多。
记忆里已验证：「男声中频 EQ +2dB：5.76% → 6.92%（EL 6.86%）」。

## 验收（缺一不可）
  · 中频 0.84 → 接近 1.0（比 e 链的 1.05 可以差一点，但不能掉回 0.84）
  · click **不得比现役 0.05×EL 明显恶化**（这是本链存在的理由）
  · 音色不掉（现役 0.9331；d 链 0.9350）
🔴 click / 杂音必须**压完 mp3 后**测。全程 44.1 kHz，最后才压 mp3。

用法: venv/bin/python -u /mnt/tts/diag/pipe_1019f.py
"""
import subprocess
import sys

PY = sys.executable
T = "/mnt/tts/projects/qwen3-tts/tools"
D = "/mnt/tts/diag"
EL = "/mnt/tts/data/cases/d172b8c4efb3/merged.wav"
SRC = D + "/mcur1019.wav"        # 44.1k，当前生产链产物（v3.1，click 0.05×）
W0 = D + "/f_a.wav"              # 中频 EQ 后
W1 = D + "/f_b.wav"              # + 句末清理
W2 = D + "/f_c.wav"              # + 句首清理
W3 = D + "/f_cf.wav"             # + clickfix
MP3 = D + "/f_cf.mp3"


def run(a):
    print("$ " + " ".join(a), flush=True)
    subprocess.run(a, check=True)


def main():
    print("== 0/4 eqfix53b（窄带中频 EQ，自动选增益）==", flush=True)
    run([PY, "-u", T + "/eqfix53b.py", SRC, EL, W0])
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
                        "/tmp/_f.wav", "--dry"])
        subprocess.run([PY, "-u", T + "/tailclean53c2.py", f, "--dry"])

    print("\n===== 客观指标 vs EL（压完 mp3 后测）=====", flush=True)
    subprocess.run([PY, "-u", T + "/measure53.py", MP3, EL, "4594"])
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
