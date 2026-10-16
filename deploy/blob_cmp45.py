#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同文案同跨度（前 198.6s）：我方 vs EL 的「空洞 / 孤立短音」定量对照。

为什么必须重算：
  上一轮的 EL 基线用的是 `ab/c_el_up.wav`（**EL 上搬 1.094 的变调版**，包络被平滑），
  得出「EL 顶起 8.7 处/分、我们 25.1 = 2.9 倍」→ 据此做了 z43「压顶起」，z43 三票全 D/C。
  真基线必须是 `cases/52727ec43154/merged.wav`。这次两边都在**同一段文案的同一跨度**上量。

两个指标：
  A「句中空洞」= ≥80ms、电平 <−45dB 的段，且前后各 300ms 内都有语音（>−35dB）
                —— 即夹在语音中间的洞，不是句末停顿。
  B「孤立短音」= 60~400ms 的发声段，电平比前后各 400ms 内其他发声段中位高 >8dB。
"""
import os
import subprocess

import numpy as np
import soundfile as sf

SR = 24000
HOP = 0.005
TMP = "/mnt/tts/diag/_blobcmp"
os.makedirs(TMP, exist_ok=True)
SPAN = 198.6
A = "/mnt/tts/diag/xf5272/prod_before_silfloor.mp3"
B = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
EL = "/mnt/tts/data/cases/52727ec43154/merged.wav"


def load(src, tag, dur=None):
    o = os.path.join(TMP, "%s.wav" % tag)
    c = ["ffmpeg", "-y", "-v", "error"]
    if dur:
        c += ["-t", "%.2f" % dur]
    c += ["-i", src, "-ac", "1", "-ar", str(SR), o]
    subprocess.run(c, check=True)
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


def holes(db, quiet_db=-45.0, minms=80.0, ctx_ms=300.0, spk_db=-35.0):
    n = int(ctx_ms / 1000.0 / HOP)
    out = []
    for s, e in runs(db < quiet_db):
        if (e - s) * HOP * 1000.0 < minms:
            continue
        a0, a1 = max(0, s - n), s
        b0, b1 = e, min(len(db), e + n)
        if a1 - a0 < 3 or b1 - b0 < 3:
            continue
        if np.median(db[a0:a1]) > spk_db and np.median(db[b0:b1]) > spk_db:
            out.append((s * HOP, (e - s) * HOP, float(db[s:e].min())))
    return out


def blobs(db, voiced_db=-42.0, minms=60.0, maxms=400.0, ctx_ms=400.0,
          bump_db=8.0):
    n = int(ctx_ms / 1000.0 / HOP)
    out = []
    for s, e in runs(db > voiced_db):
        ms = (e - s) * HOP * 1000.0
        if ms < minms or ms > maxms:
            continue
        c0, c1 = max(0, s - n), min(len(db), e + n)
        ctx = np.concatenate([db[c0:s], db[e:c1]])
        ctx = ctx[ctx > voiced_db]
        if ctx.size < 3:
            continue
        lv, cm = float(np.median(db[s:e])), float(np.median(ctx))
        if lv - cm > bump_db:
            out.append((s * HOP, ms, lv, cm, lv - cm))
    return out


def report(name, y):
    db = env_db(y)
    dur = len(db) * HOP
    hs = holes(db)
    bs = blobs(db)
    hl = [(t, ms, d) for t, ms, d in hs]
    print("== %-12s %.1fs ==" % (name, dur))
    print("   句中空洞   %3d 段  %.2f 处/分  合计 %.1fs（占 %.2f%%）  最深 %.0fdB"
          % (len(hl), len(hl) / dur * 60, sum(x[1] for x in hl) * 1.0,
             100.0 * sum(x[1] for x in hl) / dur,
             min([x[2] for x in hl]) if hl else 0))
    print("   孤立短音   %3d 段  %.2f 处/分  （>8dB）  最大顶起 %+.1fdB"
          % (len(bs), len(bs) / dur * 60,
             max([x[4] for x in bs]) if bs else 0))
    if hl:
        print("   最长的 5 个洞：%s" % "  ".join(
            "%.1fs %.0fms(%.0fdB)" % (t, ms, d)
            for t, ms, d in sorted(hl, key=lambda x: -x[1])[:5]))
    if bs:
        print("   最顶的 5 处：  %s" % "  ".join(
            "%.1fs %.0fms(%+.1f)" % (t, ms, dp)
            for t, ms, _, _, dp in sorted(bs, key=lambda x: -x[4])[:5]))
    return hl, bs


print("### 同一段文案的同一跨度（前 198.6s）###\n")
ha, ba = report("我们·修前", load(A, "A", SPAN))
hb, bb = report("我们·修后", load(B, "B", SPAN))
he, be = report("EL 女声原声", load(EL, "EL", SPAN))

print("\n### 员工点名的 4 处所在的 3 秒窗内，各有几个「孤立短音」 ###")
for nm, bs in (("我们·修前", ba), ("我们·修后", bb), ("EL 原声", be)):
    for t in (96.0, 131.0, 145.0, 170.0):
        n = [x for x in bs if abs(x[0] - t) < 1.5]
        print("  %-10s %5.0fs -> %s" % (nm, t,
              " / ".join("%.2fs %.0fms(%+.1fdB)" % (x[0], x[1], x[4]) for x in n) or "无"))
    print()
