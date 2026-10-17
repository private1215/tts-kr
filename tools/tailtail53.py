# -*- coding: utf-8 -*-
"""tailtail53.py —— 检测「黏在字后面的尾音嗯」。

## 为什么单独做这个工具
员工 z27/z28 反馈：「好比说完某个字后有个嗯」。
gruntclean53.py 的判据要求**前后都有 >0.08s 静音**，而「字+嗯」是连着的
（前面没有静音），所以这类「嗯」根本进不了候选——这正是「清了一大半，
还剩一小半」的原因：清掉的是孤立嗯，剩的是黏在字后面的尾音嗯。

## 做法
对每个发声段（后接静音 >= 0.15s），看它的**尾部拖尾**：
    · 段内峰值 pk；从段末尾往前回溯，找连续「低于 pk-DIP_DB 但高于 FLOOR_DB」
      的帧，这段就是拖尾。
    · 拖尾长度 40~300ms 且拖尾是**浊音**（zcr 低、300Hz 以下能量占比高）
      → 判定为尾音「嗯」候选。
    · 关键对比量：拖尾长度 / 段主体长度、拖尾平均能量相对 pk 的 dB 差。

在 EL 标杆上跑同样的测量做对照：如果 EL 也有拖尾，就比长度和能量差，
只清「我们显著更长/更平」的那部分，绝不清零。

用法：
    python3 tailtail53.py A.wav B.wav ... [--pause 0.15] [--dip 8] [--floor 45]
                          [--min-ms 40] [--max-ms 300] [--show 20]
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005


def load16(p, sr=SR):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def zcr_low(a, sr=SR):
    if len(a) < 80:
        return 1.0, 0.0
    z = float(np.mean(np.abs(np.diff(np.sign(a)))) / 2)
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    fr = np.fft.rfftfreq(len(a), 1.0 / sr)
    return z, float(sp[fr < 300].sum() / (sp.sum() + 1e-12))


def segs_of(x, thr_pct=35):
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    v = db > np.percentile(db, thr_pct)
    segs, i = [], 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            segs.append((i, j, db))
            i = j
        else:
            i += 1
    return db, segs


def tails(x, pause=0.15, dip=8.0, floor=45.0, min_ms=40, max_ms=300):
    db, segs = segs_of(x)
    out = []
    for k, (i, j, _) in enumerate(segs):
        # 该段后接的静音长度
        nxt = segs[k + 1][0] if k + 1 < len(segs) else len(db)
        if (nxt - j) * FRAME < pause:
            continue
        body = db[i:j]
        if len(body) < 8:
            continue
        pk = float(body.max())
        # 从末尾往前回溯：连续低于 pk-dip 但高于 pk-floor 的帧
        t = j - 1
        while t > i and db[t] < pk - dip and db[t] > pk - floor:
            t -= 1
        L = (j - 1 - t) * FRAME * 1000  # ms
        if not (min_ms <= L <= max_ms):
            continue
        if (j - t) >= (j - i):  # 整段都是拖尾 = 不是「字+嗯」
            continue
        s = t * FRAME
        d = (j - t) * FRAME
        a = x[int(s * SR):int(j * SR)]
        z, lo = zcr_low(a)
        body_ms = (j - i) * FRAME * 1000
        out.append(dict(t=s, tail_ms=L, body_ms=body_ms,
                        d_db=float(db[t:j].mean() - pk),
                        zcr=z, low=lo))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--pause", type=float, default=0.15)
    ap.add_argument("--dip", type=float, default=8.0)
    ap.add_argument("--floor", type=float, default=45.0)
    ap.add_argument("--min-ms", type=float, default=40.0)
    ap.add_argument("--max-ms", type=float, default=300.0)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--zcr", type=float, default=0.12)
    ap.add_argument("--low", type=float, default=0.45)
    a = ap.parse_args()

    print("尾部拖尾（字后的嗯）：pause>=%.2fs  dip=%.0fdB  floor=%.0fdB  "
          "长度%.0f~%.0fms" % (a.pause, a.dip, a.floor, a.min_ms, a.max_ms))
    print("%-34s %6s %6s %8s %8s %8s %8s %8s"
          % ("文件", "候选", "浊音", "个/分", "尾长ms", "主体ms", "ΔdB", "低频"))
    for p in a.files:
        if not os.path.exists(p):
            print("%-34s 缺失" % os.path.basename(p))
            continue
        x = load16(p)
        dur = len(x) / SR
        tt = tails(x, a.pause, a.dip, a.floor, a.min_ms, a.max_ms)
        vd = [r for r in tt if r["zcr"] < a.zcr and r["low"] > a.low]
        med_t = np.median([r["tail_ms"] for r in vd]) if vd else 0
        med_b = np.median([r["body_ms"] for r in vd]) if vd else 0
        med_d = np.median([r["d_db"] for r in vd]) if vd else 0
        med_l = np.median([r["low"] for r in vd]) if vd else 0
        print("%-34s %6d %6d %8.2f %8.0f %8.0f %8.1f %8.2f"
              % (os.path.basename(p)[:34], len(tt), len(vd), len(vd) / dur * 60,
                 med_t, med_b, med_d, med_l))
        for r in vd[:a.show]:
            print("      t=%7.2fs 尾%3.0fms/主体%4.0fms Δ%5.1fdB "
                  "zcr=%.3f 低频=%.2f"
                  % (r["t"], r["tail_ms"], r["body_ms"], r["d_db"],
                     r["zcr"], r["low"]))


if __name__ == "__main__":
    main()
