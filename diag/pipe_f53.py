# -*- coding: utf-8 -*-
"""pipe_f53.py —— 女声要不要接窄带中频 EQ + clickfix 的**同源对照**。

## 背景（2026-10-19 权威基线， female_full.txt 3415字 / 459.18s）
  女声 click **0.49×EL**（已比 EL 好 → 按铁律不能动）
  女声 中频 **0.89**（真实缺口；男声同款 EQ 把 0.84 修到 1.01，零音色代价）
  女声 质心 1.01 / 杂音 0.22 / cps 0.99 / 音色 0.8172

女声当初只证伪了「**全频段** LTAS」（误差 0.0339 且 click 恶化 19%），
**窄带 firequalizer 从没在女声上试过**。男声的成功不能直接外推到女声
（女声是 omnivoice 模型上限问题，音色 0.8172 是全场最短板）。

## 设计（全部从同一条女声成品解出的 wav 出发）
    f0: 原样 → mp3                       （基线）
    f1: EQ +1dB → mp3
    f2: EQ +2dB → mp3
    f3: EQ +2dB → clickfix th=0.26 → mp3
    f4: clickfix th=0.26 → mp3           （隔离 clickfix 对女声的作用）

判据（铁律）：
  - 中频要**升到 ≥0.97** 才算修到位；
  - click **不能比 EL 差**（>1.0× 就不接 EQ）；
  - 音色**不能掉**（女声音色是最大短板，掉 0.005 都算亏）。
🔴 全程保持原生 44.1 kHz，最后才压 192k mp3。
"""
import os
import subprocess
import sys

PY = sys.executable
T = "/mnt/tts/projects/qwen3-tts/tools"
D = "/mnt/tts/diag"
ELF = "/mnt/tts/diag/qwen_fullf_EL_female.wav"
SRC = D + "/f53_src.wav"          # 女声成品解出的 wav（44.1k）
CHARS = "3415"


def run(a):
    print("$ " + " ".join(a), flush=True)
    subprocess.run(a, check=True)


def to_mp3(w, mp3):
    run(["ffmpeg", "-y", "-v", "error", "-i", w, "-codec:a",
         "libmp3lame", "-b:a", "192k", "-ar", "44100", mp3])


def measure(mp3, tag):
    print("-- %s 客观指标 vs EL 女声（压完 mp3 后测）--" % tag, flush=True)
    subprocess.run([PY, "-u", T + "/measure53.py", mp3, ELF, CHARS])


def chain(tag, eq=None, th=None):
    print("\n" + "=" * 68, flush=True)
    print("%s 链：%s → %s"
          % (tag, "原样" if eq is None else "EQ %+gdB" % eq,
             "原样" if th is None else "clickfix th=%.2f" % th), flush=True)
    print("=" * 68, flush=True)
    w = SRC
    if eq is not None:
        w = "%s/%s_eq.wav" % (D, tag)
        run([PY, "-u", T + "/eqfix53b.py", SRC, ELF, w, "--gain", str(eq)])
    if th is not None:
        o = "%s/%s_cf.wav" % (D, tag)
        run([PY, "-u", T + "/clickfix53b.py", w, o, "--th", str(th),
             "--rounds", "3"])
        w = o
    mp3 = "%s/%s.mp3" % (D, tag)
    to_mp3(w, mp3)
    measure(mp3, tag)
    return mp3


def main():
    if not os.path.exists(SRC):
        run(["ffmpeg", "-y", "-v", "error", "-i",
             "/mnt/tts/output/l3f53/l3f53-1.mp3", "-ar", "44100", "-ac", "1",
             SRC])
    mp3s = []
    f0 = "%s/f0.mp3" % D
    to_mp3(SRC, f0)
    measure(f0, "f0 原样（基线）")
    mp3s.append(f0)
    mp3s.append(chain("f1", eq=1))
    mp3s.append(chain("f2", eq=2))
    mp3s.append(chain("f4", eq=None, th=0.26))
    mp3s.append(chain("f3", eq=2, th=0.26))

    print("\n" + "=" * 68, flush=True)
    print("音色对照（campplus，cosy venv）—— 女声音色是最大短板，重点看", flush=True)
    print("=" * 68, flush=True)
    args = []
    for m in mp3s:
        args += [m, ELF]
    subprocess.run(["/mnt/tts/projects/cosy/venv/bin/python", "-u",
                    T + "/spkcmp53.py"] + args, check=False)
    print("\n成品：%s" % " ".join(mp3s), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
