# -*- coding: utf-8 -*-
"""eqprobe48.py —— 女声 GS 中低频偏薄：离线筛选固定 EQ 候选。

## 背景
`gs48f_sola`（GS 女声，3416 字全篇）实测频段能量占比 vs EL 标杆
`cases/00b6a45ad6aa/merged.wav`：
    1-4k    0.07217  vs EL 0.11397   （缺 ~2.0dB）
    <500Hz  0.38157  vs EL 0.41759   （缺 ~0.4dB，忽略）
    4-8k    0.00969  vs EL 0.00997   （已持平，🔴 不能补，否则嘶声）
→ 需要一个**只补 1-4k、不碰 4-8k** 的固定 EQ。

## 为什么离线筛而不重新合成
GS 合成一条全篇要 ~3 分钟；EQ 是线性时不变滤波，
直接对已有成品施加 = 对裸输出施加后再走 atempo+volume（顺序不影响频谱形状）。
一次合成、多次试参数，省时间且对比更干净。

## 口径（写死，避免又踩坑）
- 一律降到 **16k 单声道** 再算（跨采样率比频段占比 = 无效比较）。
- 频段占比：**发声帧**（20ms 帧 RMS ≥ p30）的总能量按频段归一。
- 杂音：measure48.hiss（静音 20% 帧的 4-8k 占比）。
- click：measure48.click（先归一峰值 0.70 再判 |Δ|>=0.30）。
- 音色：measure48.timbre（campplus × EL 同文案标杆）。

用法：
  python eqprobe48.py --src /mnt/tts/output/gs48f_sola/xxx.mp3 \
      --el /mnt/tts/data/cases/00b6a45ad6aa/merged.wav
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/diag")
import measure48 as m48  # noqa: E402

BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]


def band_ratio(path):
    """发声帧（RMS >= p30）各频段能量占比 %。"""
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    n = int(sr * 0.02)
    m = (len(x) // n) * n
    fr = x[:m].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    seg = fr[rms >= np.percentile(rms, 30)].ravel()
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    tot = X.sum() + 1e-12
    return [float(X[(f >= lo) & (f < hi)].sum() / tot * 100)
            for lo, hi in BANDS]


def apply_eq(src16, af, dst):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src16,
                    "-af", af, "-ar", "16000", "-ac", "1", dst],
                   check=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--el", default="/mnt/tts/data/cases/00b6a45ad6aa/merged.wav")
    ap.add_argument("--no-timbre", action="store_true")
    ap.add_argument("--cands", nargs="*", default=[
        "anull",
        "equalizer=f=2500:t=q:w=3.0:g=2.0",
        "equalizer=f=1800:t=q:w=2.0:g=2.0",
        "equalizer=f=1500:t=q:w=2.5:g=2.5",
        "equalizer=f=2000:t=q:w=1.5:g=2.5",
    ])
    a = ap.parse_args()

    el = m48.to16k(a.el)
    src16 = m48.to16k(a.src)
    print("源：%s（%.1fs）" % (a.src, sf.info(src16).duration))
    hdr = "%-42s" % "EQ 候选"
    for lo, hi in BANDS:
        hdr += "%9s" % ("%d-%d" % (lo, hi))
    print(hdr + "%10s %9s %9s" % ("杂音4-8k", "click%", "音色"))

    elb = band_ratio(el)
    print("%-42s" % "EL 标杆" + "".join("%9.4f" % v for v in elb)
          + "%10.3f %9.3f %9s" % (m48.hiss(el), m48.click(el), "1.0000"))

    tmp = "/tmp/eqprobe48"
    os.makedirs(tmp, exist_ok=True)
    for i, af in enumerate(a.cands):
        dst = os.path.join(tmp, "c%02d.wav" % i)
        apply_eq(src16, af, dst)
        b = band_ratio(dst)
        # 🔴 候选必须 peak 归一后才能比 click；hiss/click 内部已做归一（click），
        #    hiss 是能量占比，本身与电平无关。
        tb = float("nan")
        if not a.no_timbre:
            try:
                tb = m48.timbre(dst, el)
            except Exception as e:
                print("   ⚠️ 音色失败 %s" % e)
        print("%-42s" % af + "".join("%9.4f" % v for v in b)
              + "%10.3f %9.3f %9s" % (m48.hiss(dst), m48.click(dst),
                                      ("%.4f" % tb) if tb == tb else "n/a"))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
