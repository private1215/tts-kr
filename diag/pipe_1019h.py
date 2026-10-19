# -*- coding: utf-8 -*-
"""pipe_1019h.py —— clickfix53b 音色代价的**严格同源对照**。

## 为什么要重跑
上一轮留下一个疑点：纯 fix_chain 音色 **0.9583**，加 clickfix 后 **0.9329**，
差 0.025。但这个 0.9583 是在**另一个测量上下文**里得到的，而且与 v3.1 当时的
「clickfix 零损失（0.9331→0.9336）」直接矛盾。可能是：
  (a) clickfix 真的吃音色；
  (b) 两次测的**不是同一条音频**（不同 job / 不同长度 / 不同归一化）。
→ 必须同源重测，否则会基于一个错数字改错参数。

## 设计
全部从**同一个** `mcur1019.wav`（纯 fix_chain 输出，44.1k，560.98s）出发：

    a: 原样 → mp3                     （基线，无 EQ 无 clickfix）
    b: clickfix th=0.30 → mp3         （隔离 clickfix 代价）
    c: EQ +2dB → mp3                  （隔离 EQ 代价）
    d: EQ +2dB → clickfix th=0.30 → mp3  （= 现役生产链）
    e: clickfix th=0.40 → mp3         （阈值上探：少改样本）
    f: clickfix th=0.50 → mp3
    g: EQ +2dB → clickfix th=0.50 → mp3

每组同时测：**音色（campplus，压完 mp3 后）** + click（压完 mp3 后）。
EL 基线 = cases/d172b8c4efb3/merged.wav。

🔴 全程 44.1 kHz，最后才压 192k mp3（16k 上压 mp3 会毁音色，铁律）。
"""
import os
import subprocess
import sys

PY = sys.executable
T = "/mnt/tts/projects/qwen3-tts/tools"
D = "/mnt/tts/diag"
EL = "/mnt/tts/data/cases/d172b8c4efb3/merged.wav"
SRC = D + "/mcur1019.wav"
CHARS = "4594"


def run(a):
    print("$ " + " ".join(a), flush=True)
    subprocess.run(a, check=True)


def to_mp3(w, mp3):
    run(["ffmpeg", "-y", "-v", "error", "-i", w, "-codec:a",
         "libmp3lame", "-b:a", "192k", "-ar", "44100", mp3])


def chain(tag, eq=None, th=None):
    """eq: None 或 dB 数；th: None 或 clickfix 阈值。"""
    print("\n" + "=" * 68, flush=True)
    print("%s 链：%s → %s → mp3"
          % (tag,
             "原样" if eq is None else "EQ %+gdB" % eq,
             "原样" if th is None else "clickfix th=%.2f" % th), flush=True)
    print("=" * 68, flush=True)
    w = SRC
    steps = []
    if eq is not None:
        w = "%s/%s_eq.wav" % (D, tag)
        run([PY, "-u", T + "/eqfix53b.py", SRC, EL, w,
             "--gain", str(eq)])
        steps.append(w)
    if th is not None:
        o = "%s/%s_cf.wav" % (D, tag)
        run([PY, "-u", T + "/clickfix53b.py", w, o,
             "--th", str(th), "--rounds", "3"])
        w = o
        steps.append(o)
    mp3 = "%s/%s.mp3" % (D, tag)
    to_mp3(w, mp3)
    print("-- %s 客观指标 vs EL（压完 mp3 后测）--" % tag, flush=True)
    subprocess.run([PY, "-u", T + "/measure53.py", mp3, EL, CHARS])
    for s in steps:
        if s.startswith(D + "/" + tag):
            try:
                os.remove(s)
            except OSError:
                pass
    return mp3


def main():
    print("源：%s（纯 fix_chain 输出，44.1k）" % SRC, flush=True)
    print("EL：%s" % EL, flush=True)
    mp3s = []
    # a 原样
    a = "%s/a.mp3" % D
    to_mp3(SRC, a)
    print("\n-- a 原样（基线）--", flush=True)
    subprocess.run([PY, "-u", T + "/measure53.py", a, EL, CHARS])
    mp3s.append(a)
    mp3s.append(chain("b", eq=None, th=0.30))
    mp3s.append(chain("c", eq=2, th=None))
    mp3s.append(chain("d", eq=2, th=0.30))
    mp3s.append(chain("e", eq=None, th=0.40))
    mp3s.append(chain("f", eq=None, th=0.50))
    mp3s.append(chain("g", eq=2, th=0.50))

    print("\n" + "=" * 68, flush=True)
    print("音色对照（campplus，cosy venv，多窗平均）", flush=True)
    print("=" * 68, flush=True)
    subprocess.run(["/mnt/tts/projects/cosy/venv/bin/python", "-u",
                    T + "/spkcmp53.py"] + mp3s + [EL], check=False)
    print("\n成品：%s" % " ".join(mp3s), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
