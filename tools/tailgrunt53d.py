# -*- coding: utf-8 -*-
"""tailgrunt53d.py —— 诊断「粘连型嗯」：说完某个字后紧跟的「嗯」（无静音间隔）。

## 背景

v1~v3 判据都要求「前后静音 > 0.08s」，只能抓**孤立**的「嗯」。
z28 员工反馈：
> 「整体方向是对的，就是好比说完某个字后有个嗯，现在清理了一大半，
>   还有一小半没清理干净」

→ 「一大半」= 孤立的「嗯」，v1~v3 已经清掉。
→ 「一小半」= **紧贴发言、中间没有静音的「嗯」**，判据抓不到。

## 做法

对每个发声段，若其**后面紧跟长停顿**（> TAIL_SIL），取段尾最多 300ms
作为候选，测：
    · zcr（过零率）：鼻浊音「嗯」极低（<0.10）
    · low（300Hz 以下能量占比）：鼻浊音极高（>0.55）
    · dip（尾部与主段之间的能量凹陷深度 dB）：「嗯」是附加的，中间有凹陷
    · 尾部时长

与 EL 标杆同口径对比，只有「我们显著多于 EL」的区间才动手。

用法：python3 tailgrunt53d.py <ours.wav> <el.wav> [--dump N]
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
TAIL_SIL = 0.25      # 段后静音超过这么长，才算「句末」
MERGE_GAP = 0.15     # 发声段间隔小于这么长就合并（避免碎片）
TAIL_MAX = 0.30      # 尾部候选最长这么长


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
    if len(a) < 40:
        return 1.0, 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(a)))) / 2)
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    fr = np.fft.rfftfreq(len(a), 1.0 / sr)
    low = float(sp[fr < 300].sum() / (sp.sum() + 1e-12))
    return zcr, low


def frames(x):
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    return db


def segments(db):
    v = db > np.percentile(db, 35)
    segs, i = [], 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            segs.append([i, j])
            i = j
        else:
            i += 1
    # 合并间隔 < MERGE_GAP 的相邻段
    out = []
    for s, e in segs:
        if out and (s - out[-1][1]) * FRAME < MERGE_GAP:
            out[-1][1] = e
        else:
            out.append([s, e])
    return out


def analyse(p, dump=0, tag=""):
    x = load16(p)
    db = frames(x)
    segs = segments(db)
    dur = len(x) / SR
    rows = []
    for k, (a, b) in enumerate(segs):
        ns = segs[k + 1][0] * FRAME if k + 1 < len(segs) else dur
        tail_sil = ns - b * FRAME
        if tail_sil < TAIL_SIL:
            continue
        s0, s1 = a * FRAME, b * FRAME
        seg_len = s1 - s0
        if seg_len < 0.15:
            continue
        # 尾部候选：段末最多 TAIL_MAX
        tlen = min(TAIL_MAX, seg_len * 0.6)
        ts, te = s1 - tlen, s1
        a_t = x[int(ts * SR):int(te * SR)]
        zcr, low = voiced_quality(a_t)
        # 凹陷：尾部区间内 db 的极小值 vs 主段峰值
        fi, fj = int(ts / FRAME), int(te / FRAME)
        main_pk = float(db[int(s0 / FRAME):max(int(s0 / FRAME) + 1, fi)].max())
        tail_db = db[fi:fj]
        if len(tail_db) < 3:
            continue
        dip = main_pk - float(tail_db.min())
        tail_pk = float(tail_db.max())
        rows.append(dict(t=s0, seg=seg_len * 1000, tail=tlen * 1000,
                         sil=tail_sil * 1000, zcr=zcr, low=low,
                         dip=dip, tpk=tail_pk, mpk=main_pk))
    return x, dur, rows


def report(p, dump=0, tag=""):
    x, dur, rows = analyse(p, dump, tag)
    print("=== %s  %s  %.1fs ===" % (tag, os.path.basename(p), dur))
    print("句末候选（段后静音>%dms）共 %d 个，%.2f/分钟"
          % (TAIL_SIL * 1000, len(rows), len(rows) / dur * 60))
    # 鼻浊音强判据
    nasal = [r for r in rows if r["zcr"] < 0.10 and r["low"] > 0.55]
    print("  强鼻音(zcr<0.10 且 低频>0.55)：%d 个 (%.2f/分钟)"
          % (len(nasal), len(nasal) / dur * 60))
    nasal2 = [r for r in rows
              if r["zcr"] < 0.10 and r["low"] > 0.55 and r["dip"] > 6]
    print("  强鼻音+有凹陷(dip>6dB)：        %d 个 (%.2f/分钟)"
          % (len(nasal2), len(nasal2) / dur * 60))
    if rows:
        z = np.array([r["zcr"] for r in rows])
        l = np.array([r["low"] for r in rows])
        d = np.array([r["dip"] for r in rows])
        print("  分位：zcr p10=%.3f p25=%.3f 中位=%.3f | "
              "low p75=%.2f p90=%.2f 中位=%.2f | dip p75=%.1f p90=%.1f"
              % (np.percentile(z, 10), np.percentile(z, 25), np.median(z),
                 np.percentile(l, 75), np.percentile(l, 90), np.median(l),
                 np.percentile(d, 75), np.percentile(d, 90)))
    if dump:
        print("  --- 强鼻音样本（前 %d 个）---" % dump)
        for r in nasal2[:dump]:
            print("    t=%7.2fs 段%4.0fms 尾%3.0fms 后静音%4.0fms "
                  "zcr=%.3f 低频=%.2f dip=%.1fdB"
                  % (r["t"], r["seg"], r["tail"], r["sil"],
                     r["zcr"], r["low"], r["dip"]))
    print()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dump", type=int, default=0)
    a = ap.parse_args()
    for p in a.files:
        report(p, a.dump, tag=os.path.basename(p)[:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
