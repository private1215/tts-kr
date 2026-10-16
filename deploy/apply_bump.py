#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第五个修复尝试：压掉「局部突然顶起来的短音」（员工点名的"额"）

## 员工最新定位（z10 三票，控制题全对，全部有效）
    디아 / 이민정 ：soft/hard/字清楚 全选「差不多」
    24124112     ：全选「两段都像机器」
    → **元音起音柔化证伪**（第四次失败，见 apply_onset.py 头）。
    用户转述：「1-3 题 一个语音 一个问题 5秒 特别短的『额』音 存在」
    z42 三题同源（成品 90~110s），5 秒 = **成品 95s** —— 正是 blip_find 命中的 95.50s。

## 95.5s 到底长什么样（tiny_asr.py 10ms 包络）
    95.10~95.45  -35dB   轻
    **95.50~95.65  -22dB   突然顶起 13dB，稳 200ms**   ← 员工听到的"额"
    95.70~95.90  -72dB   死寂
    95.90        -34dB   重新起音
    转写：95.14「있는데」→ 95.92「이게」
    🔴 whisper 把这段并进「있는데」了，所以 CER 抓不到（1060 vs 1066 音节、零 insert）。
    但「있는데」3 音节占 780ms = 260ms/音节，而全篇 5.37 音节/秒 = 186ms/音节
    → **偏长 40%，里面确实塞了额外的东西**。

## 判据（可自动化）
一个发声段，若自身电平比**前后各 400ms 内其他发声段**的中位电平高 >BUMP_DB，
且时长在 100~350ms → 判为"局部顶起"，压到「上下文 +3dB」。

## 用法
/mnt/tts/projects/qwen3-tts/venv/bin/python apply_bump.py [src] [dst]
      BUMP_DB=8 MAXLEN=0.35 TARGET_DB=3
"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

HOP = 0.005
SR = 24000
TMP = "/mnt/tts/diag/_bump"


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


def find_bumps(y, sr, bump_db=8.0, minlen=0.10, maxlen=0.35, ctx_win=0.40):
    db = env(y, sr)
    floor = np.percentile(db, 10)
    voiced = db[db > floor + 12].mean()
    seg = runs(db > floor + 6)
    lvls = []
    for s, e in seg:
        a, b = int(s * HOP * sr), int(e * HOP * sr)
        lvls.append(20 * np.log10(np.sqrt((y[a:b] ** 2).mean()) + 1e-12))
    out = []
    for i, (s, e) in enumerate(seg):
        ln = (e - s) * HOP
        if not (minlen <= ln <= maxlen):
            continue
        ctx = []
        for j, (s2, e2) in enumerate(seg):
            if j == i:
                continue
            # 前 400ms 内
            if 0 < (s - e2) * HOP <= ctx_win:
                ctx.append(lvls[j])
            # 后 400ms 内
            if 0 < (s2 - e) * HOP <= ctx_win:
                ctx.append(lvls[j])
        if len(ctx) < 2:
            continue
        cm = float(np.median(ctx))
        if lvls[i] - cm >= bump_db:
            out.append((s * HOP, ln, lvls[i], cm))
    return out, floor, voiced, len(seg)


def apply_bump(src, dst, bump_db=8.0, maxlen=0.35, target_db=3.0):
    y, sr = load(src, "src")
    bumps, floor, voiced, _ = find_bumps(y, sr, bump_db, maxlen=maxlen)
    n = 0
    for t, ln, lvl, ctx in bumps:
        a, b = int(t * sr), int((t + ln) * sr)
        want = ctx + target_db
        g = 10 ** ((want - lvl) / 20.0)
        if g >= 0.999:
            continue
        seg_y = y[a:b].copy()
        m = len(seg_y)
        fw = min(int(0.02 * sr), m // 2)
        w = np.ones(m)
        if fw > 1:
            w[:fw] = np.linspace(1.0, g, fw)
            w[-fw:] = np.linspace(g, 1.0, fw)
            w[fw:m - fw] = g
        else:
            w[:] = g
        y[a:b] = seg_y * w
        n += 1
    pk = float(np.abs(y).max())
    if pk > 0.99:
        y = y * (0.99 / pk)
    sf.write(os.path.join(TMP, "o.wav"), y, sr)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(TMP, "o.wav"),
                    "-b:a", "192k", dst], check=True)
    return n, bumps


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
    y, sr = load(src, "probe")
    bumps, floor, voiced, nseg = find_bumps(y, sr)
    print("=== 局部顶起的短音（>8dB 且 100~350ms）===")
    print("底噪 %.1fdB  发声均值 %.1fdB  发声段 %d 个  命中 %d 处"
          % (floor, voiced, nseg, len(bumps)))
    MARK = [60.0, 95.5, 98.0, 99.0, 112.0, 113.0, 170.0]
    for t, ln, lvl, ctx in bumps:
        flag = "  <<<员工点名附近" if any(abs(t - m) < 3.0 for m in MARK) else ""
        print("  %7.2fs  %5.0fms  %.1fdB (上下文 %.1fdB, 高出 %.1fdB)%s"
              % (t, ln * 1000, lvl, ctx, lvl - ctx, flag))
    if not bumps:
        print("  无命中 —— 判据需要放宽")

    print("\n=== 压制档位对照（压到 上下文+N dB）===")
    for tgt in (3.0, 6.0):
        dst = "/mnt/tts/diag/_bump/b_t%g.mp3" % tgt
        n, _ = apply_bump(src, dst, 8.0, 0.35, tgt)
        print("  压到 上下文+%gdB -> 处理 %d 处 -> %s" % (tgt, n, dst))
    print("\nBUMP_DONE")


if __name__ == "__main__":
    main()
