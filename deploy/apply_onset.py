#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第四个修复尝试：**元音起音柔化**（针对「突然出来短音」）

## 员工 z9 实测（两票，控制题都答对，全部有效）
    디아      ：4/4 选 C「声音断了一下 / 卡了一下」
    24124112  ：4/4 选 B「突然多出一个短音」，自由文本「全部突然 出来 短音」
🔴 关键是**4 段全中**，包括我特意放的「无点名、无死寂」隐藏对照段 s4
   → 这是**全篇普遍现象**，不是 95/99/112/170 那几个点。
   → 也解释了为什么「填死寂」（只改 0.8% 时长）不可能有效。

## 客观对应
attack_ab.py：句首 attack 我们 **157.5ms** vs 真人 **290ms**（我们起音快一倍 = 更"蹦"）
kill 掉的三条路线（都别重踩）：
  - 软噪声门压拖尾：release 117→152ms，整体电平掉 2.6dB
  - 填房间底噪：死寂达标但 Σz² 32.1→34.7、平坦度 +3.83σ→−4.30σ
  - 句末淡出：要么改 0 段，改 397 段才 117→82.5ms

## 做法
只柔化**元音起音**（辅音起音不动，否则加重"吃字"——员工已在抱怨吃字）：
  停顿(≥80ms) 之后的起音，取前 15ms 算 ≥4kHz 能量占比，
  占比低 = 没有辅音爆音 = 元音起音 → 对它做 40ms 淡入（0.28 → 1.0）。
  辅音起音保持原样，塞音/擦音的清晰度不受影响。
🔴 时长严格不变。

## 闸门
attack 157ms → 220~260ms（真人 290ms）；Σz² 不许比 32.1 差；音色不许掉。
"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

HOP = 0.005
SR = 24000
TMP = "/mnt/tts/diag/_onset"


def load(path, tag="x", sr=SR):
    os.makedirs(TMP, exist_ok=True)
    out = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ac", "1", "-ar", str(sr), out], check=True)
    return sf.read(out, dtype="float32")


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


def hf_ratio(seg, sr):
    """前 15ms 里 ≥4kHz 能量占比：高 = 辅音爆音，低 = 元音"""
    n = int(0.015 * sr)
    seg = seg[:n]
    if len(seg) < 32:
        return 0.0
    w = np.hanning(len(seg))
    S = np.abs(np.fft.rfft(seg * w)) ** 2
    f = np.fft.rfftfreq(len(seg), 1.0 / sr)
    tot = S.sum() + 1e-12
    return float(S[f >= 4000].sum() / tot)


def attack_ms(path, tag):
    y, sr = load(path, tag)
    db = env(y, sr)
    floor = np.percentile(db, 10)
    voiced = db[db > floor + 12].mean()
    lo, hi = floor + 3, voiced - 3
    on = db > floor + 6
    seg = runs(on)
    atk = []
    for k in range(len(seg) - 1):
        if (seg[k + 1][0] - seg[k][1]) * HOP < 0.25:
            continue
        ge = seg[k + 1][0]

        def cr(i0, i1, lvl):
            for i in range(i0, min(i1, len(db))):
                if db[i] >= lvl:
                    return i
            return None
        i = cr(ge, ge + 120, lo)
        j = cr(i if i is not None else ge, ge + 120, hi) if i is not None else None
        if i is not None and j is not None:
            atk.append((j - i) * HOP * 1000)
    return float(np.median(atk)) if atk else -1.0, len(atk), floor, voiced


def process(src, dst, fade_ms=40.0, hf_thr=0.15, gap_ms=80.0, start_gain=0.28):
    y, sr = load(src, "src")
    db = env(y, sr)
    floor = np.percentile(db, 10)
    on = db > floor + 6
    seg = runs(on)
    faden = int(fade_ms / 1000.0 * sr)
    n_v = n_c = 0
    for k in range(len(seg) - 1):
        if (seg[k + 1][0] - seg[k][1]) * HOP * 1000 < gap_ms:
            continue
        a = int(seg[k + 1][0] * HOP * sr)
        if a + faden > len(y):
            continue
        r = hf_ratio(y[a:a + faden], sr)
        if r > hf_thr:          # 有辅音爆音 → 不动，保清晰度
            n_c += 1
            continue
        w = np.linspace(start_gain, 1.0, faden)
        y[a:a + faden] = y[a:a + faden] * w
        n_v += 1
    pk = float(np.abs(y).max())
    if pk > 0.99:
        y = y * (0.99 / pk)
    sf.write(os.path.join(TMP, "o.wav"), y, sr)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(TMP, "o.wav"),
                    "-b:a", "192k", dst], check=True)
    return n_v, n_c


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
    a0, n0, f0, v0 = attack_ms(src, "before")
    print("原状   attack %.1f ms (n=%d)  底噪 %.1f  发声均值 %.1f" % (a0, n0, f0, v0))
    print("目标   真人 290ms；员工反馈「全部突然出来短音」")
    for fade, hf, g in [(40.0, 0.15, 0.28), (60.0, 0.15, 0.28),
                        (40.0, 0.25, 0.28), (40.0, 0.15, 0.45)]:
        dst = "/mnt/tts/diag/_onset/o_%g_%g_%g.mp3" % (fade, hf, g)
        nv, nc = process(src, dst, fade, hf, start_gain=g)
        a, n, f, v = attack_ms(dst, "after")
        print("fade=%-4g hf<=%-4g gain0=%-4g  attack %6.1f ms (n=%2d)  "
              "柔化元音起音 %3d 处 / 保留辅音 %3d 处  %s"
              % (fade, hf, g, a, n, nv, nc,
                 "OK" if 200 <= a <= 280 else ("过头" if a > 280 else "不够")))
    print("\nONSET_DONE")


if __name__ == "__main__":
    main()
