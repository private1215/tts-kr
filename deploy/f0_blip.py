#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证新假设：员工听到的「啊/额」短音，是不是**音高突然拔高**

## 前提（已确认）
- CER=0.038，转写比文案**少 6 音节**、零 insert → **不是多念了字**（full_cer.py）
- 停顿中的孤立短发声段密度与真人一样（29.6 vs 28.9 个/分）→ 不是多了东西
- 死寂段是 EL 的 4.4 倍，但男声更严重而员工没提 → 死寂不是主感知点

→ 那"异声"只能是**这些音本身发音不对**。线索：blip_feat 量到 12 处短音的
  F0 跨度 176~349Hz，而女声正常说话 F0 应该稳定在 200~230。
  **一段 200ms 的孤立元音，F0 比上下文高出一大截，听起来就是"尖着蹦出来的啊"**。

## 本脚本
对每处短音，量它自身 F0 中位 vs **前后各 1.5s 上下文**的 F0 中位，看 Δ。
同时给全篇 F0 分布做参照。
若 Δ 普遍为正且幅度可观（>20Hz / >10%）→ 假设成立，可做局部降调修复。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python f0_blip.py
"""
import os
import subprocess

import numpy as np
import librosa
import soundfile as sf

SRC = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
TMP = "/mnt/tts/diag/_f0blip"
os.makedirs(TMP, exist_ok=True)

HITS = [(8.86, 0.225), (20.78, 0.130), (35.11, 0.080), (66.02, 0.095),
        (69.48, 0.095), (78.98, 0.185), (85.38, 0.205), (95.50, 0.200),
        (135.00, 0.115), (146.74, 0.160), (185.00, 0.215), (189.34, 0.105)]
MARK = [95.5, 99.0, 112.0, 170.0]
CTX = 1.5


def load16():
    out = os.path.join(TMP, "b.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", SRC,
                    "-ac", "1", "-ar", "16000", out], check=True)
    return sf.read(out, dtype="float32")


def f0_med(seg, sr):
    try:
        f, _, _ = librosa.pyin(seg, fmin=80, fmax=500, sr=sr,
                               frame_length=1024, hop_length=256)
        v = f[np.isfinite(f)]
        return float(np.median(v)) if len(v) >= 3 else float("nan"), len(v)
    except Exception:
        return float("nan"), 0


def voiced_ctx(y, sr, t0, t1):
    """取 [t0,t1) 里能量较高的部分（跳过静音），避免把静音算进 F0"""
    a, b = int(t0 * sr), int(t1 * sr)
    seg = y[max(0, a):min(len(y), b)]
    if len(seg) < sr * 0.3:
        return np.array([])
    hop = 256
    e = np.array([np.sqrt((seg[i:i + hop] ** 2).mean())
                  for i in range(0, len(seg) - hop, hop)])
    if not len(e):
        return np.array([])
    thr = np.percentile(e, 60)
    idx = np.where(e > thr)[0]
    if not len(idx):
        return np.array([])
    s, t = idx[0] * hop, (idx[-1] + 1) * hop
    return seg[s:min(t, len(seg))]


def main():
    y, sr = load16()
    # 全篇 F0 分布
    fall, _, _ = librosa.pyin(y, fmin=80, fmax=500, sr=sr, hop_length=256)
    v = fall[np.isfinite(fall)]
    print("全篇 F0：中位 %.1f Hz  p10 %.1f  p90 %.1f  (n=%d)"
          % (np.median(v), np.percentile(v, 10), np.percentile(v, 90), len(v)))
    print("\n%-8s %-7s %-9s %-9s %-9s %s"
          % ("起点", "时长", "短音F0", "上文F0", "下文F0", "判定"))
    ups = 0
    tot = 0
    for t, ln in HITS:
        a, b = int(t * sr), int((t + ln) * sr)
        f0, n = f0_med(y[a:b], sr)
        pre = voiced_ctx(y, sr, t - CTX, t - 0.05)
        post = voiced_ctx(y, sr, t + ln + 0.05, t + ln + CTX)
        fp, _ = f0_med(pre, sr) if len(pre) else (float("nan"), 0)
        fq, _ = f0_med(post, sr) if len(post) else (float("nan"), 0)
        ctx = np.nanmedian([fp, fq])
        d = f0 - ctx if np.isfinite(f0) and np.isfinite(ctx) else float("nan")
        if np.isfinite(d):
            tot += 1
            if d > 20:
                ups += 1
        verdict = ("偏高 +%.0fHz" % d) if np.isfinite(d) and d > 20 else \
                  ("偏低 %.0fHz" % d) if np.isfinite(d) and d < -20 else \
                  ("接近 %+.0fHz" % d if np.isfinite(d) else "量不出")
        flag = "  <<<员工点名" if any(abs(t - m) < 1.5 for m in MARK) else ""
        print("%-8.2f %-7.3f %-9.1f %-9.1f %-9.1f %s%s"
              % (t, ln, f0, fp, fq, verdict, flag))
    print("\n相对上下文偏高 >20Hz 的：%d / %d" % (ups, tot))
    print("F0BLIP_DONE")


if __name__ == "__main__":
    main()
