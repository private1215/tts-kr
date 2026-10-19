# -*- coding: utf-8 -*-
"""pipe_1019g.py —— 定中频 EQ 增益：**不含句末清理**的纯净版对照。

## 为什么单独跑
句末清理（53b2/53c2）有**过清**争议（残留 0.75/分 vs EL 3.89/分），
要等员工裁决。而**中频 EQ 是确定性收益**（对齐标杆频谱，不依赖听感）。
→ 生产上应该**先只上 EQ**，清理等票。所以需要 EQ-only 的干净数据。

本脚本跑两条 EQ-only 链：
    g: eqfix53b(--gain 2) → clickfix53b
    h: eqfix53b(--gain 3) → clickfix53b

## 为什么 +2 也要测（而不是直接用 f 链自动选的 +3）
eqfix53b 的自动选器用**压 mp3 前、16k 口径**的 click 做惩罚，选了 +3dB。
但项目铁律要求 **click 必须测压完 mp3 后的值** —— 那个惩罚口径本身就是错的。
按「对齐标杆」原则：
    +2dB → 中频 6.91% vs EL 6.86%  = **1.007× 最贴合**
    +3dB → 中频 7.64%              = 1.11× 过冲
→ 需要实测两条的最终 click / 音色来定。

🔴 全程 44.1 kHz，最后才压 192k mp3。

用法: venv/bin/python -u /mnt/tts/diag/pipe_1019g.py
"""
import subprocess
import sys

PY = sys.executable
T = "/mnt/tts/projects/qwen3-tts/tools"
D = "/mnt/tts/diag"
EL = "/mnt/tts/data/cases/d172b8c4efb3/merged.wav"
SRC = D + "/mcur1019.wav"


def run(a):
    print("$ " + " ".join(a), flush=True)
    subprocess.run(a, check=True)


def chain(tag, gain):
    w0 = "%s/%s_a.wav" % (D, tag)
    w1 = "%s/%s_cf.wav" % (D, tag)
    mp3 = "%s/%s_cf.mp3" % (D, tag)
    print("\n========== %s 链：中频 EQ +%g dB → clickfix53b =========="
          % (tag, gain), flush=True)
    run([PY, "-u", T + "/eqfix53b.py", SRC, EL, w0, "--gain", str(gain)])
    run([PY, "-u", T + "/clickfix53b.py", w0, w1, "--th", "0.30",
         "--rounds", "3"])
    run(["ffmpeg", "-y", "-v", "error", "-i", w1, "-codec:a",
         "libmp3lame", "-b:a", "192k", mp3])
    print("\n-- %s 客观指标 vs EL（压完 mp3 后测）--" % tag, flush=True)
    subprocess.run([PY, "-u", T + "/measure53.py", mp3, EL, "4594"])
    print("-- %s 句末/句首残留（EQ-only 不应改变清理前基线）--" % tag,
          flush=True)
    subprocess.run([PY, "-u", T + "/tailclean53b2.py", mp3,
                    "/tmp/_%s.wav" % tag, "--dry"])
    subprocess.run([PY, "-u", T + "/tailclean53c2.py", mp3, "--dry"])
    return mp3


def main():
    g = chain("g", 2)
    h = chain("h", 3)
    print("\n成品：%s / %s" % (g, h), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
