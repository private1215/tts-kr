# -*- coding: utf-8 -*-
"""clickfix53b.py —— clickfix53 的**保采样率**版本。

## 为什么要有第二个版本
`clickfix53.py` 里写死了 `SR = 16000`，输出一律 16 kHz。这在当初只是个
「内部诊断脚本」，没人拿它当生产环节。今天（2026-10-19）实测发现它是条
**链式的坑**：

    mcur1019_pipe.wav (16k)  vs EL   音色 0.9360 ✅同人
    mcur1019_pipe.mp3 (16k)  vs EL   音色 0.6903 ❌不同人
    同一对 wav/mp3 互比               0.7900   ← mp3 编码本身吃掉了音色
    对照：44.1k 的 wav → mp3 互比     0.9999   ← 44.1k 下 mp3 完全透明

即 **libmp3lame 在 16 kHz 输入上会明显伤音色，在 44.1 kHz 上是透明的**。
生产交付是 mp3，所以「先降采样再压 mp3」= 白白毁掉音色。

## 做法
检测仍然在 16k 做（click 口径是 16k，不能改），但**改样本发生在原采样率上**：
  ① 读原文件（原生 sr）
  ② ffmpeg 降 16k 只用于**定位**跳变位置
  ③ 位置映射回原生 sr：p = round(i * sr/16000)
  ④ 把 [p, p+k) 这段（k = round(sr/16000)，16k 下即 1 个样本）换成
     两端点的线性插值 —— 与 16k 版「x[i+1] = (x[i]+x[i+2])/2」等价
  ⑤ 原采样率写出

用法：python3 clickfix53b.py <in.wav> <out.wav> [--th 0.30] [--rounds 3]
"""
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

DET_SR = 16000
PEAK = 0.70


def read_mono(p):
    x, sr = sf.read(p, dtype="float32")
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), int(sr)


def load16(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(DET_SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    return x.astype(np.float64)


def click_rate(x):
    pk = np.max(np.abs(x))
    xn = x * (PEAK / pk) if pk > 0 else x
    d = np.abs(np.diff(xn))
    return float((d >= 0.30).mean()), int((d >= 0.30).sum())


def find_idx(x16, th):
    """在 16k 域找超阈值跳变位置（与 clickfix53 同口径）。"""
    pk = np.max(np.abs(x16))
    xn = x16 * (PEAK / pk) if pk > 0 else x16
    d = np.diff(xn)
    return np.where(np.abs(d) >= th)[0]


def apply_at_native(y, idx16, sr):
    """把 16k 域的位置映射回原生 sr，用线性插值替换那一段。

    🔴 不能直接按 p = round(i * sr/16000) 替换：16k 的第 i 个样本是原生
    [2.756*i, 2.756*(i+1)) 这段的**加权平均**，跳变的真实位置可能偏 1~2 个
    样本。实测：直接替换 click 只降 23%（16k 版降 99%）。
    → 在 p 附近 ±k 个样本里找**真实的最大跳变**，以它为中心做插值。
    """
    k = max(1, int(round(sr / float(DET_SR))))
    n = len(y)
    changed = 0
    for i in idx16:
        p = int(round(i * sr / float(DET_SR)))
        lo = max(1, p - k)
        hi = min(n - 1, p + k)
        if hi <= lo:
            continue
        d = np.abs(np.diff(y[lo - 1:hi + 1]))
        if not d.size:
            continue
        j = lo + int(np.argmax(d))          # 最大跳变后那个样本
        a = max(1, j - k)
        b = min(n - 1, j + k)
        if b <= a:
            continue
        y[a:b] = np.linspace(y[a - 1], y[b], b - a + 2)[1:-1]
        changed += (b - a)
    return changed


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__)
        return 1
    src, out = args[0], args[1]

    def get(k, d):
        return (float(sys.argv[sys.argv.index(k) + 1])
                if k in sys.argv else d)

    th = get("--th", 0.30)
    rounds = int(get("--rounds", 3))

    y, sr = read_mono(src)
    print("输入 %s  %.1fs @ %d Hz" % (src, len(y) / float(sr), sr),
          flush=True)

    x16 = load16(src)
    r0, n0 = click_rate(x16)
    print("修前 click %.4f%%（%d 个）" % (r0 * 100, n0), flush=True)

    total = 0
    for rd in range(rounds):
        # 每轮重新在**当前**波形上定位（与 clickfix53 一致）
        cur16 = load16_from_array(y, sr) if rd else x16
        idx = find_idx(cur16, th)
        if not len(idx):
            break
        total += apply_at_native(y, idx, sr)
        print("  第%d轮 定位 %d 处" % (rd + 1, len(idx)), flush=True)

    r1, n1 = click_rate(load16_from_array(y, sr))
    print("修后 click %.4f%%（%d 个）｜改动样本 %d 个（%.4f%%）"
          % (r1 * 100, n1, total, 100.0 * total / max(1, len(y))),
          flush=True)
    sf.write(out, y.astype(np.float32), sr)
    print("输出 %s  @ %d Hz（保持原采样率）" % (out, sr), flush=True)
    return 0


def load16_from_array(y, sr):
    """把内存里的原生波形降采样到 16k（仅用于定位，与 ffmpeg 口径一致）。"""
    t = tempfile.mktemp(suffix=".wav")
    sf.write(t, y.astype(np.float32), sr)
    return load16(t)


if __name__ == "__main__":
    sys.exit(main() or 0)
