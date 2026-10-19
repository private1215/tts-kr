# -*- coding: utf-8 -*-
"""clickfix53.py —— **精准样本级** click 修复（此前没试过的路子）。

## 为什么还敢再试 click
已证伪的是两类「整体处理」：
  · ffmpeg declick 滤镜（它自己检测 + 淡入淡出，检测逻辑不对）
  · 瞬态整形 / 限幅器（作用于整条波形，不是定位到具体样本）
**没试过的是「先精确定位超阈值跳变样本，再只改那几个样本」**。

clickattr53.py 已经给出定位：我们 1484 个 click 样本 / 1306 个簇，
簇大小中位=1（**孤立单样本跳变**占多数），|Δ| p50=0.325。
→ 只改「孤立的那 1~3 个样本」，对音质的影响应该是可控的。

## 做法
迭代若干轮：
    找 |Δ| >= TH 的跳变位置 i（归一化到峰值 0.70 后判定）
    把 x[i+1] 换成 x[i]、x[i+2] 的**线性插值**
    （孤立单样本跳变 → 插值后两侧斜率都被压下来）
扫描 TH，报告 click 率 / 质心 / 中频 / 杂音，挑不改坏音质的点。

⚠️ 铁律：判定 click 必须**归一化到峰值 0.70 + 无条件重采样 16k**。
⚠️ 改完必须 ASR 核对（asr_ab.py）确认没削字。

用法：python3 clickfix53.py <in.wav> <out.wav> [--th 0.30] [--rounds 3]
      python3 clickfix53.py <in.wav> <out_dir> --scan     # 扫描多个阈值
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
PEAK = 0.70


def load16(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    return x.astype(np.float64)


def click_rate(x):
    pk = np.max(np.abs(x))
    xn = x * (PEAK / pk) if pk > 0 else x
    d = np.abs(np.diff(xn))
    return float((d >= 0.30).mean()), int((d >= 0.30).sum())


def fix(x, th=0.30, rounds=3):
    y = x.copy()
    pk = np.max(np.abs(y))
    if pk <= 0:
        return y
    scale = PEAK / pk
    total = 0
    for _ in range(rounds):
        xn = y * scale
        d = np.diff(xn)
        idx = np.where(np.abs(d) >= th)[0]
        if not len(idx):
            break
        total += len(idx)
        for i in idx:
            j = i + 1
            if 0 < j < len(y) - 1:
                y[j] = 0.5 * (y[j - 1] + y[j + 1])
    print("   修复样本 %d 处（%d 轮）" % (total, rounds), flush=True)
    return y


def quality(p):
    """质心 / 中频 / 杂音（与 measure53 同口径）"""
    x = load16(p)
    n = int(0.005 * SR)
    nfr = len(x) // n
    fr = x[:nfr * n].reshape(nfr, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    v = db > -40
    sp = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)) ** 2
    fq = np.fft.rfftfreq(n, 1.0 / SR)
    cents, mids, noise = [], [], []
    for i in range(len(v)):
        s = sp[i]
        tot = s.sum() + 1e-12
        if v[i]:
            cents.append(float((fq * s).sum() / tot))
            mids.append(float(s[(fq >= 1000) & (fq <= 3000)].sum() / tot))
        else:
            noise.append(float(s[(fq >= 4000) & (fq <= 8000)].sum() / tot))
    return (float(np.median(cents)) if cents else 0.0,
            float(np.median(mids)) if mids else 0.0,
            float(np.median(noise)) if noise else 0.0)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__)
        return 1
    src, out = args[0], args[1]
    scan = "--scan" in sys.argv

    def get(k, d):
        return (float(sys.argv[sys.argv.index(k) + 1])
                if k in sys.argv else d)

    th = get("--th", 0.30)
    rounds = int(get("--rounds", 3))

    x = load16(src)
    r0, n0 = click_rate(x)
    c0, m0, z0 = None, None, None
    print("原产  click %.4f%% (%d 个)" % (r0 * 100, n0), flush=True)

    outs = []
    if scan:
        os.makedirs(out, exist_ok=True)
        ths = [0.30, 0.26, 0.22, 0.18, 0.14]
    else:
        ths = [th]

    for t in ths:
        y = fix(x.copy(), th=t, rounds=rounds)
        r1, n1 = click_rate(y)
        tp = tempfile.mktemp(suffix=".wav")
        sf.write(tp, y.astype(np.float32), SR)
        c, m, z = quality(tp)
        if c0 is None:
            c0, m0, z0 = c, m, z
        dst = os.path.join(out, "fix_th%02d.wav" % int(t * 100)) if scan \
            else out
        sf.write(dst, y.astype(np.float32), SR)
        outs.append(dst)
        print("th=%.2f  click %.4f%% (%d)  质心 %.0f  中频 %.1f%%  "
              "杂音 %.1f%%  -> %s"
              % (t, r1 * 100, n1, c, m * 100, z * 100, dst), flush=True)

    if c0:
        print("\n原产：质心 %.0f  中频 %.1f%%  杂音 %.1f%%"
              % (c0, m0 * 100, z0 * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
