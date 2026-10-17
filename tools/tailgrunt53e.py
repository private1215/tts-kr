# -*- coding: utf-8 -*-
"""tailgrunt53e.py —— 检测「句末粘连型嗯」（用户 2026-10-17 指出，被我不该忘掉的那句）。

## 重新解读员工反馈

员工说：「好比说完某个字后有个嗯」
→ **「嗯」紧跟在语音段后面，不是孤立在静音中。**

z28 员工给的位置（换算为绝对秒）：
    男：85 / 101 / 129 / 143
    女：350 / 363 / 374 / 387 / 400 / 408

`gruntprobe53.py` 用 -40/-50dB 门槛在**原始文件**探测，发现：
    男 101s -> 100.66s 30ms(-33dB, zcr=0.027, 低频=0.90) + 一串 5ms 浊音残片
    男 143s -> 143.28~143.36s 10+60+5ms 浊音残片(zcr<0.14, 低频>0.60)
    女 374s -> 373.40s 80ms(-33.6dB, zcr=0.042, 低频=0.70)

这些都是**前一个主语音段刚结束、句末粘连的浊音尾巴**。
v1~v4 判据要求「前后静音 >0.08s」→ 永远抓不到它们。

## 判据

用**绝对低门槛**（-40dB）分段后，找满足以下条件的短段：
    · 时长 5~120ms
    · 峰值 -45 ~ -20dB（太低是噪声，太高是主语音）
    · 浊音：zcr < 0.15 且 300Hz 以下能量 > 0.45
    · 紧连前一段：间隔 < 0.03s（说明是句末尾巴）
    · 后段开始前有 >0.05s 间隙（确实是句末）
    · 前一段时长 > 0.10s（有主语音）

与 EL 标杆同口径对比，只有「我们显著多于 EL」的区间才动手。

用法：python3 tailgrunt53e.py <ours> <el> [更多成对文件]
"""
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


def voiced_quality(a, sr=SR):
    if len(a) < 16:
        return 1.0, 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(a)))) / 2)
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    fr = np.fft.rfftfreq(len(a), 1.0 / sr)
    low = float(sp[fr < 300].sum() / (sp.sum() + 1e-12))
    return zcr, low


def analyse(p, dump=0):
    x = load16(p)
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

    dur = len(x) / SR
    rows = []
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
        rows.append(dict(t=s, d=d * 1000, pk=pk, zcr=zcr, low=low,
                         gpre=gap_pre * 1000, gnxt=gap_nxt * 1000,
                         prev_dur=prev_dur * 1000))
    return dur, rows


def report(p, dump=0):
    dur, rows = analyse(p, dump)
    print("=== %s  %.1fs ===" % (os.path.basename(p), dur))
    print("句末粘连型嗯（后接静音的浊音尾巴）：%d 个 (%.2f/分钟)"
          % (len(rows), len(rows) / dur * 60))
    if rows:
        ds = np.array([r["d"] for r in rows])
        pks = np.array([r["pk"] for r in rows])
        print("  时长 中位=%.0fms p90=%.0fms | 峰值 中位=%.0fdB p10=%.0fdB"
              % (np.median(ds), np.percentile(ds, 90),
                 np.median(pks), np.percentile(pks, 10)))
    if dump:
        for r in rows[:dump]:
            print("  t=%7.2fs %3.0fms 峰值%.0fdB zcr=%.3f 低频=%.2f "
                  "前间隔%.0fms 后间隔%.0fms"
                  % (r["t"], r["d"], r["pk"], r["zcr"], r["low"],
                     r["gpre"], r["gnxt"]))
    print()
    return dur, rows


def main():
    fs = sys.argv[1:]
    if not fs:
        print("用法：tailgrunt53e.py <文件1> [文件2] ...")
        return 1
    for p in fs:
        report(p, dump=10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
