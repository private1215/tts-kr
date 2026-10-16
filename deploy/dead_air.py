#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「断句有异声」真身核查：停顿里的**死寂段**（能量悬崖）

## 线索（tiny_asr.py 实测，成品 95.5s）
    95.50  -22 dB   正常语音（있는데 的尾音节）
    95.73  -72 dB   160ms 内掉 50 dB
    95.80  -72 dB   死寂 200ms
    95.90  -34 dB   突然起音
真人的句间不会这样：真人有连续的房间底噪，衰减和起音都是"有过渡"的。
我们这一刀切 + 死寂 + 陡起，人耳就听成「断句处突然蹦出啊/额」。

## 本脚本要回答
1. 死寂段（相对自身底噪 −15 / −20 dB 以下、≥100ms）有多少、多长？
2. 是**模型原生**还是**后处理**造成？→ 裸推理 vs 成品
3. 真人原声有没有？→ 目标值

⚠️ 用**相对阈值**（各自 p10 底噪往下 N dB），不然后处理改了整体响度就没法比。
🔴 已知坑：全局抬底噪会让相对阈值跟着抬（2026-10-15 已证伪），所以这里
   只做**测量**，不提修复方案。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python dead_air.py
"""
import os
import subprocess

import numpy as np
import soundfile as sf

TMP = "/mnt/tts/diag/_deadair"
os.makedirs(TMP, exist_ok=True)
HOP = 0.005

FILES = [
    ("ours_raw", "/mnt/tts/diag/xf5272/fem_raw.wav", "我们·裸推理"),
    ("ours_eq",  "/mnt/tts/diag/xf5272/fem_witheq.wav", "我们·后处理+EQ"),
    ("ours_mp3", "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3", "我们·现役成品"),
    ("human",    "/mnt/tts/data/cases/52727ec43154/merged.wav", "真人原声"),
]
MINLEN = 0.10


def load16(path, tag):
    out = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ac", "1", "-ar", "16000", out], check=True)
    return sf.read(out, dtype="float32")[0], 16000


def env(y, sr):
    hop = int(HOP * sr)
    n = len(y) // hop
    e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    return 20 * np.log10(e + 1e-12)


def runs(mask):
    d = np.diff(mask.astype(np.int8))
    st = list(np.where(d == 1)[0] + 1)
    en = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        st.insert(0, 0)
    if mask[-1]:
        en.append(len(mask))
    m = min(len(st), len(en))
    return [(st[i], en[i]) for i in range(m)]


def analyse(tag, path, desc):
    y, sr = load16(path, tag)
    dur = len(y) / sr
    db = env(y, sr)
    floor = np.percentile(db, 10)
    voiced = db[db > floor + 12].mean()
    print("\n=== %-9s %s" % (tag, desc))
    print("    时长 %.1fs  p10底噪 %.1fdB  发声均值 %.1fdB  动态范围 %.0fdB"
          % (dur, floor, voiced, voiced - floor))
    for off in (15.0, 20.0):
        thr = floor - off
        seg = [(s, e) for s, e in runs(db < thr) if (e - s) * HOP >= MINLEN]
        tot = sum((e - s) * HOP for s, e in seg)
        # 落差：死寂段前 40ms 均值 − 死寂段电平
        drops = []
        for s, e in seg:
            pre = db[max(0, s - 8):s]
            if len(pre):
                drops.append(float(pre.mean() - db[s:e].mean()))
        print("    [底噪−%.0fdB] 死寂段 %2d 处 共 %5.1fs (%.1f%%)  中位时长 %.0fms  前落差 %.0fdB"
              % (off, len(seg), tot, 100.0 * tot / dur,
                 np.median([(e - s) * HOP for s, e in seg]) * 1000 if seg else 0,
                 np.median(drops) if drops else 0))
        if off == 20.0 and len(seg) <= 14:
            for s, e in seg:
                print("        %7.2f  %.0fms  %.0fdB" % (s * HOP, (e - s) * HOP * 1000, db[s:e].mean()))
    return dict(tag=tag, floor=floor, voiced=voiced, dur=dur)


def main():
    import sys
    extra = []
    for a in sys.argv[1:]:
        if "=" in a:
            t, p = a.split("=", 1)
            extra.append((t, p, t))
    res = []
    for tag, path, desc in FILES + extra:
        if not os.path.exists(path):
            print("!! 缺 %s" % path)
            continue
        res.append(analyse(tag, path, desc))
    print("\n=== 密度汇总（处/分钟，越低越自然）===")
    for r in res:
        print("  %-10s %.1fs" % (r["tag"], r["dur"]))
    print("\nDEADAIR_DONE")


if __name__ == "__main__":
    main()
