#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三个修复尝试：句末**快速收尾**到底噪（不是淡到 0）

## 唯一的显著残差
attack_ab.py：句末 release 我们 **117.5ms** vs 真人 **22.5ms**（慢 5 倍）。
其他都排除了：
  - 短音数量 29.6 vs 真人 28.9 个/分（一样）
  - CER 0.038、转写少 6 音节零 insert（不是多念字）← full_cer.py
  - 短音 F0 相对上下文**偏低** −37~−216Hz（自然句末降调，不是拔高）← f0_blip.py

真人是干脆闭嘴，我们是元音自然衰减。这条衰减尾巴落在断句处 = 员工听到的"啊——"。

## 做法（与前两次失败的区别）
❌ apply_tailcut.py（软噪声门压低电平）：release 反而 117→152ms，
   且整体电平掉 2.6dB 会加重吃字。
❌ apply_roomfloor.py（填房间底噪）：死寂 3.32→0.30 处/分有效，
   但 Σz² 32.1→34.7、平坦度 +3.83σ→−4.30σ 翻向 → 回退。

✅ 本版：只把**尾巴末端**（能量已降到 发声均值−BELOW dB 之后）在
   FADE_MS 内线性收到底噪电平。**不动 −BELOW dB 以上的主体**，
   因此不会吃掉字；收到底噪而非 0，因此不制造新的死寂。

## 闸门
release 应 117ms → 40~70ms；Σz² 不许比 32.1 差；音色不许掉。
"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

HOP = 0.005
TMP = "/mnt/tts/diag/_tailfade"


def load16(path, tag="x"):
    os.makedirs(TMP, exist_ok=True)
    out = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ac", "1", "-ar", "16000", out], check=True)
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


def release_ms(path, tag):
    y, sr = load16(path, tag)
    db = env(y, sr)
    floor = np.percentile(db, 10)
    voiced = db[db > floor + 12].mean()
    lo, hi = floor + 3, voiced - 3
    on = db > floor + 6
    seg = runs(on)
    rel = []
    for k in range(len(seg) - 1):
        if (seg[k + 1][0] - seg[k][1]) * HOP < 0.25:
            continue
        pe = seg[k][1]

        def cr(i0, i1, lvl, up):
            for i in range(i0, min(i1, len(db))):
                if (db[i] >= lvl) if up else (db[i] <= lvl):
                    return i
            return None
        i2 = cr(max(pe - 120, seg[k][0]), pe, hi, False)
        j2 = cr(i2 if i2 is not None else pe, min(pe + 130, len(db)), lo, False) if i2 is not None else None
        if i2 is not None and j2 is not None:
            rel.append((j2 - i2) * HOP * 1000)
    return float(np.median(rel)) if rel else -1.0, len(rel), floor, voiced


def process(src, dst, below=20.0, fade_ms=50.0):
    y, sr = load16(src, "src")
    db = env(y, sr)
    floor = np.percentile(db, 10)
    voiced = db[db > floor + 12].mean()
    tail_db = voiced - below
    floor_amp = 10 ** (floor / 20.0)

    on = db > floor + 6
    seg = runs(on)
    nf = 0
    for s, e in seg[:-1]:
        # 从段末往前找"尾巴起点"：能量已降到 tail_db 以下且持续到段末
        i = e - 1
        while i > s and db[i] < tail_db:
            i -= 1
        if i <= s:
            continue
        a = int(i * HOP * sr)
        b = int(e * HOP * sr)
        n = b - a
        if n < int(0.01 * sr):
            continue
        faden = min(int(fade_ms / 1000.0 * sr), n)
        seg_y = y[a:b]
        amp = np.abs(seg_y) + 1e-12
        target = np.minimum(amp, np.maximum(floor_amp, amp * 0.0))
        # 线性：起点保持原样，faden 之后收到 floor_amp
        g = np.ones(n)
        g[n - faden:] = np.linspace(1.0, 0.0, faden)
        new = amp * g
        new = np.maximum(new, floor_amp * 0.9)
        y[a:b] = np.sign(seg_y) * new
        nf += 1

    pk = float(np.abs(y).max())
    if pk > 0.99:
        y = y * (0.99 / pk)
    sf.write(os.path.join(TMP, "o.wav"), y, sr)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(TMP, "o.wav"),
                    "-b:a", "192k", dst], check=True)
    return nf


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
    r0, n0, f0, v0 = release_ms(src, "before")
    print("原状      release %.1f ms (n=%d)  底噪 %.1f  发声均值 %.1f" % (r0, n0, f0, v0))
    for below, fade in [(20.0, 50.0), (15.0, 50.0), (20.0, 30.0), (25.0, 40.0)]:
        dst = "/mnt/tts/diag/_tailfade/t_%g_%g.mp3" % (below, fade)
        nf = process(src, dst, below, fade)
        r, n, f, v = release_ms(dst, "after")
        print("below=%-4g fade=%-4gms  release %6.1f ms (n=%2d) 底噪 %.1f 发声 %.1f  改%d段  %s"
              % (below, fade, r, n, f, v, nf,
                 "OK" if 40 <= r <= 70 else ("偏硬" if r < 40 else "不够")))
    print("\nTAILFADE_DONE  目标：真人 22.5ms，我们原状 117.5ms")


if __name__ == "__main__":
    main()
