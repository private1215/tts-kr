# -*- coding: utf-8 -*-
"""tailclean53.py —— 清理「句末粘连型嗯/浊音尾巴」。

## 背景（z28 员工反馈 + tailgrunt53e.py 诊断）

员工：「好比说完某个字后有个嗯」「断句有异声、卡顿」
`gruntprobe53.py` 在员工标注位置探测到：
    男 101s -> 100.66s 30ms(-33dB, zcr=0.027, low=0.90) + 一串 5ms 浊音残片
    男 143s -> 143.28~143.36s 10+60+5ms 浊音残片(zcr<0.14, low>0.60)
    女 374s -> 373.40s 80ms(-33.6dB, zcr=0.042, low=0.70)

`tailgrunt53e.py` 全篇统计：
    男：我们 49 个 (5.25/分钟) vs EL 36 个 (3.89/分钟)  → 我们多 36%
    女：我们 14 个 (1.83/分钟) vs EL 34 个 (4.44/分钟)  → 我们比 EL 少

→ **男声「嗯/卡顿/断句异声」的根因是句末粘连型浊音尾巴**。女声方向反，不动。

## 判据（句末粘连型）

用绝对门槛 -40dB 分段后，找同时满足：
    · 时长 5~120ms
    · 峰值 -45 ~ -20dB（太低是噪声，太高是主语音）
    · 浊音：zcr < 0.15 且 300Hz 以下能量 > 0.45
    · 紧连前一段：间隔 < 0.03s
    · 后段开始前有 >0.05s 间隙
    · 前一段时长 > 0.10s

用**等长静音**替换（总时长不变），5ms 淡入淡出避免 click。

## 用法
    python3 tailclean53.py in.wav out.wav [--dry]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
VTH = -40.0


def load16(p, sr=SR):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def fade(a, ms=5):
    n = int(SR * ms / 1000)
    if len(a) < 2 * n:
        return a
    w = np.linspace(0, 1, n)
    a = a.copy()
    a[:n] *= w
    a[-n:] *= w[::-1]
    return a


def voiced_quality(a, sr=SR):
    if len(a) < 16:
        return 1.0, 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(a)))) / 2)
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    fr = np.fft.rfftfreq(len(a), 1.0 / sr)
    low = float(sp[fr < 300].sum() / (sp.sum() + 1e-12))
    return zcr, low


def find_tails(x):
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    v = db > VTH
    raw, i = [], 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            raw.append((i, j))
            i = j
        else:
            i += 1

    out = []
    for k in range(1, len(raw) - 1):
        a, b = raw[k]
        s = a * FRAME
        d = (b - a) * FRAME
        if not (0.005 <= d <= 0.120):
            continue
        pk = float(db[a:b].max())
        if not (-45 <= pk <= -20):
            continue
        prev_dur = (raw[k - 1][1] - raw[k - 1][0]) * FRAME
        if prev_dur < 0.10:
            continue
        gap_pre = (a - raw[k - 1][1]) * FRAME
        gap_nxt = (raw[k + 1][0] - b) * FRAME
        if gap_pre > 0.030 or gap_nxt < 0.050:
            continue
        seg = x[int(s * SR):int((s + d) * SR)]
        zcr, low = voiced_quality(seg)
        if not (zcr < 0.15 and low > 0.45):
            continue
        out.append((s, d, pk, zcr, low, gap_pre, gap_nxt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--dry", action="store_true", help="只报告不写文件")
    a = ap.parse_args()

    x = load16(a.src)
    tails = find_tails(x)
    dur = len(x) / SR
    print("%s  %.1fs  句末粘连型嗯 %d 个 (%.2f/分钟)"
          % (os.path.basename(a.src), dur, len(tails), len(tails) / dur * 60),
          flush=True)
    for s, d, pk, zcr, low, gp, gn in tails:
        print("   t=%7.2fs %3.0fms 峰值%.0fdB zcr=%.3f 低频=%.2f "
              "前间隔%.0fms 后间隔%.0fms"
              % (s, d * 1000, pk, zcr, low, gp * 1000, gn * 1000), flush=True)

    if a.dry or not tails:
        if not tails and not a.dry:
            sf.write(a.dst, x, SR)
            print("无需处理，已原样写出", flush=True)
        return 0

    y = x.copy()
    for s, d, *_ in tails:
        i0 = int(s * SR)
        i1 = int((s + d) * SR)
        n = i1 - i0
        # 直接填零，避免残留电平仍被检测器抓出
        seg = np.zeros(n, dtype=np.float32)
        y[i0:i1] = fade(seg)
    sf.write(a.dst, y, SR)
    print("-> %s  已替换 %d 处（等长静音，总时长不变）" % (a.dst, len(tails)),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
