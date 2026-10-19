# -*- coding: utf-8 -*-
"""eqscan53.py —— 扫描中频 EQ 增益，找最接近 EL 且不引入嘶声的工作点。

## 依据（measure53.py 同文本 EL 对照）

    中频1k-3k占比：我们 5.76% vs EL 6.86%  → 我们少 16%（发闷）
    click%      ：我们 0.0172 vs EL 0.0124 → 我们多 38%（EQ 不应恶化它）

记忆：男声「音色发闷」与员工 z21 选「机器音」吻合。中频不足是后处理可修项。

## 做法

用 ffmpeg 静态 peak EQ（中心 1.8k，带宽 1.6k）扫描增益 0~5dB，
测每个工作点的：中频占比 / 谱质心 / click% / 静音段高频(4-8k)杂音。
选「中频最接近 EL 6.86%，且杂音、click 不明显恶化」的点。

⚠️ 静态 EQ，不用自适应（记忆：atempo+自适应EQ 会反向补高频 → 嘶声爆炸）。

用法：python3 eqscan53.py <in.wav> [el.wav]
"""
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
VTH = -40.0
TARGET_MID = 0.0686      # EL 中频占比
EL_CLICK = 0.000124      # EL click 占比（小数）
EL_NOISE = 0.0505        # EL 静音段 4-8k 占比


def apply_eq(p, gain):
    if gain == 0:
        return p
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-af", "equalizer=f=1800:t=q:w=1600:g=%g" % gain,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    return t


def load16(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    return x


def metrics(p):
    x = load16(p).astype(np.float64)
    pk = np.max(np.abs(x))
    xn = x * (0.70 / pk) if pk > 0 else x
    click = float((np.abs(np.diff(xn)) >= 0.30).mean())

    n = int(FRAME * SR)
    nfr = len(x) // n
    fr = x[:nfr * n].reshape(nfr, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    v = db > VTH
    sp = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / SR)

    cents, mids, noise = [], [], []
    for i in range(len(v)):
        s = sp[i]
        tot = s.sum() + 1e-12
        if v[i]:
            cents.append(float((freqs * s).sum() / tot))
            mids.append(float(s[(freqs >= 1000) & (freqs <= 3000)].sum()
                              / tot))
        else:
            noise.append(float(s[(freqs >= 4000) & (freqs <= 8000)].sum()
                              / tot))
    return dict(click=click,
                mid=float(np.median(mids)) if mids else 0,
                centroid=float(np.median(cents)) if cents else 0,
                noise=float(np.median(noise)) if noise else 0)


def main():
    if len(sys.argv) < 2:
        print("用法：eqscan53.py <in.wav>")
        return 1
    src = sys.argv[1]
    print("%-8s %10s %10s %10s %10s" %
          ("增益dB", "中频%", "质心Hz", "click%", "杂音%"))
    best = None
    for g in [0, 1, 1.5, 2, 2.5, 3, 4, 5]:
        t = apply_eq(src, g)
        m = metrics(t)
        mid = m["mid"] * 100
        ck = m["click"] * 100
        nz = m["noise"] * 100
        print("%-8.1f %10.2f %10.0f %10.4f %10.2f"
              % (g, mid, m["centroid"], ck, nz))
        # 评分：中频接近 EL，且 click/杂音不明显恶化
        err = abs(m["mid"] - TARGET_MID) / TARGET_MID
        pen = 0.0
        if m["click"] > EL_CLICK * 1.5:
            pen += 1.0
        if m["noise"] > EL_NOISE * 1.5:
            pen += 1.0
        score = err + pen
        if best is None or score < best[0]:
            best = (score, g, m)
    print()
    print("EL 目标：中频 %.2f%%  click %.4f%%  杂音 %.2f%%"
          % (TARGET_MID * 100, EL_CLICK * 100, EL_NOISE * 100))
    print("最佳增益 = %g dB（中频 %.2f%%，click %.4f%%，杂音 %.2f%%）"
          % (best[1], best[2]["mid"] * 100, best[2]["click"] * 100,
             best[2]["noise"] * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
