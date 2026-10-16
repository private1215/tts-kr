#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""员工新点名 4 处，直接在**他实际听到的那几段 fb 素材**上查。

员工（第44轮）：
  第1题 6 秒「额」   题1 片内 6s  = 绝对 96.0s
  第2题 1 秒「啊」                 = 131.0s
  第2题 15 秒「啊」                = 145.0s
  第3题 / 第4题 10 秒「啊」        = 170.0s

🔴 上一轮教训：我把根因判成「停顿下沉」，改了 13dB，但员工点名的时间点
   一个新点都没变（还多了 145s）→ 「停顿下陷」不是他听到的那个东西，至少不是主因。
   这次先把「短音」本身找出来，不再靠位置相关推因果。

判据（沿用 apply_bump 口径）：孤立短发声段 = 60~400ms 的段，
电平比前后各 400ms 内**其他发声段**中位高 >8dB。
"""
import os
import subprocess

import numpy as np
import soundfile as sf

SR = 24000
HOP = 0.005
TMP = "/mnt/tts/diag/_z45"
os.makedirs(TMP, exist_ok=True)
FB = "/mnt/tts/fb"
EL = "/mnt/tts/data/cases/52727ec43154/merged.wav"

# (题号, 片内秒, 绝对秒, 片起点, A名, B名)
MARKS = [
    (1, 6.0, 96.0, 90.0, "z44_a_mid.mp3", "z44_b_mid.mp3"),
    (2, 1.0, 131.0, 130.0, "z44_a_lo.mp3", "z44_b_lo.mp3"),
    (2, 15.0, 145.0, 130.0, "z44_a_lo.mp3", "z44_b_lo.mp3"),
    (3, 10.0, 170.0, 160.0, "z44_a_tl.mp3", "z44_b_tl.mp3"),
]


def load(src, tag):
    o = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1",
                    "-ar", str(SR), o], check=True)
    return sf.read(o, dtype="float32")[0]


def env_db(y):
    h = int(HOP * SR)
    n = len(y) // h
    m = y[:n * h].astype(np.float64).reshape(n, h)
    return 20.0 * np.log10(np.sqrt((m ** 2).mean(axis=1)) + 1e-12)


def runs(mask):
    d = np.diff(mask.astype(np.int8))
    st = list(np.where(d == 1)[0] + 1)
    en = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        st.insert(0, 0)
    if mask[-1]:
        en.append(len(mask))
    return list(zip(st, en))


def blobs(db, voiced_db=-42.0, minms=60.0, maxms=400.0, ctx_ms=400.0,
          bump_db=8.0):
    out = []
    for s, e in runs(db > voiced_db):
        ms = (e - s) * HOP * 1000.0
        if ms < minms or ms > maxms:
            continue
        c0 = max(0, s - int(ctx_ms / 1000.0 / HOP))
        c1 = min(len(db), e + int(ctx_ms / 1000.0 / HOP))
        ctx = np.concatenate([db[c0:s], db[e:c1]])
        ctx = ctx[ctx > voiced_db]
        if ctx.size < 3:
            continue
        lv = float(np.median(db[s:e]))
        cm = float(np.median(ctx))
        if lv - cm > bump_db:
            out.append((s * HOP, ms, lv, cm, lv - cm))
    return out


def bar(db, a, b, width=64):
    seg = db[int(a / HOP):int(b / HOP)]
    if seg.size == 0:
        return "(空)"
    q = np.clip((seg + 50.0) / 38.0, 0, 1)
    lv = " .:-=+*#%@"
    idx = np.clip((q * (len(lv) - 1)).astype(int), 0, len(lv) - 1)
    step = max(1, len(idx) // width)
    return "".join(lv[i] for i in idx[::step][:width])


print("### 员工实际听到的素材：A=现役 / B=句间地板抬起 / EL=同秒原声")
print("### 横条 = 5ms 包络（左 -50dB … 右 -12dB），一行 3 秒\n")
for qn, t, abs_t, off, fa, fb in MARKS:
    print("=" * 78)
    print("第%d题 片内 %.0fs（绝对 %.0fs）" % (qn, t, abs_t))
    for nm, p in (("A 现役", os.path.join(FB, fa)),
                  ("B 修后", os.path.join(FB, fb)),
                  ("EL 原声", EL)):
        y = load(p, "%s_%d_%d" % (nm[:2], qn, int(t)))
        db = env_db(y)
        lo, hi = max(0.0, t - 1.5), min(len(db) * HOP, t + 1.5)
        print("  %-7s [%5.1f~%5.1fs] %s" % (nm, lo, hi, bar(db, lo, hi)))
        near = [x for x in blobs(db) if abs(x[0] - t) < 1.5]
        if near:
            for st, ms, lv, cm, dp in near:
                print("            └ 孤立短音 @%.2fs %.0fms  电平%.1f 上下文%.1f (高%+.1fdB)"
                      % (st, ms, lv, cm, dp))
        else:
            print("            └ ±1.5s 内无孤立短音(>8dB)")
    print()
print("DONE")
