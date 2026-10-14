# -*- coding: utf-8 -*-
"""补噪声底，把 EQ 啃掉的 flat 找回来。

## 逻辑链（2026-10-14）
1. flat_ablate.py 实测：不加 EQ 时 flat=0.0435(-0.86σ)，v11 全套 0.0350(-3.69σ)。
   **flat 是 EQ 一级一级啃掉的**，不是素材本身的问题。
2. 但 EL 自己 b4000_8000 只有 0.0129（高频很少）却 flat=0.0461。
   少能量 ≠ 低 flat：EL 的高频是**噪声状底噪**，我们的是"被滤波器切干净的空白"。
   空白 -> 频谱更"尖" -> flat 掉。所以正解不是少切，而是**切完补回噪声底**。
3. 记忆里也写过「EL 的 flat 高来自底噪」，只是当时当成"EQ 造不出来"放弃了。
   现在改成分两步：EQ 管频段形状，噪声底管 flat，互不打架。

## 用法
    python noise_floor.py m_v11.wav --out m_v11n.wav
    python noise_floor.py m_v11.wav --scan          # 扫电平找 flat 命中点
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eq_search6 as E  # noqa: E402

TARGET_FLAT = 0.0461


def pink(n, rng):
    """Voss-McCartney 近似：几个不同更新率的随机源叠加，得 ~1/f 谱。"""
    out = np.zeros(n)
    k = 16
    vals = rng.standard_normal(k)
    for i in range(k):
        step = 1 << i
        idx = np.arange(0, n, step)
        out += np.repeat(rng.standard_normal(len(idx)), step)[:n]
    return out / (out.std() + 1e-12)


def flat_of(y16):
    return float(np.mean(E.librosa.feature.spectral_flatness(
        y=y16[:60 * 16000], hop_length=256)[0]))


def measure(path):
    return E.measure(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="")
    ap.add_argument("--scan", action="store_true",
                    help="扫描噪声电平，打印 flat / 频段 z，不写文件")
    ap.add_argument("--levels", default="-46,-42,-38,-34,-30,-26")
    a = ap.parse_args()

    src = a.src if os.path.isabs(a.src) else os.path.join(HERE, a.src)
    y, sr = sf.read(src, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    rng = np.random.default_rng(0)
    noise = pink(len(y), rng).astype(np.float32)
    # 让噪声跟信号同量纲：按 RMS 归一
    noise /= (np.sqrt((noise ** 2).mean()) + 1e-12)
    sig_rms = np.sqrt((y ** 2).mean())

    print("%-8s %8s %7s | %s" % ("电平dB", "flat", "z(flat)",
                                 " ".join("%7s" % k.replace("b", "")
                                          for k in E.BASE if k != "flat")))
    for lv in [float(x) for x in a.levels.split(",")]:
        amp = sig_rms * (10 ** (lv / 20.0))
        mix = (y + amp * noise).astype(np.float32)
        t = tempfile.mktemp(suffix=".wav")
        sf.write(t, mix, sr)
        m = measure(t)
        os.unlink(t)
        mm, ss = E.BASE["flat"]
        print("%-8g %8.4f %+7.2f | %s"
              % (lv, m["flat"], (m["flat"] - mm) / ss,
                 " ".join("%+7.2f" % ((m[k] - E.BASE[k][0]) / E.BASE[k][1])
                          for k in E.BASE if k != "flat")))
    if a.scan:
        return

    # 二分找命中 TARGET_FLAT 的电平
    lo, hi = -60.0, -10.0
    for _ in range(12):
        mid = (lo + hi) / 2
        amp = sig_rms * (10 ** (mid / 20.0))
        mix = (y + amp * noise).astype(np.float32)
        t = tempfile.mktemp(suffix=".wav")
        sf.write(t, mix, sr)
        f = flat_of(E.load16(t)[0])
        os.unlink(t)
        if f < TARGET_FLAT:
            lo = mid
        else:
            hi = mid
    amp = sig_rms * (10 ** (hi / 20.0))
    out = (y + amp * noise).astype(np.float32)
    dst = a.out or (os.path.splitext(src)[0] + "_nf.wav")
    sf.write(dst, out, sr)
    print("\n命中 flat=%.4f 的电平 %.1f dB -> %s" % (TARGET_FLAT, hi, dst))
    m = measure(dst)
    print("复核 flat=%.4f (z=%+.2f)  J=%.1f"
          % (m["flat"], (m["flat"] - E.BASE["flat"][0]) / E.BASE["flat"][1],
             E.cost(m)))


if __name__ == "__main__":
    main()
