# -*- coding: utf-8 -*-
"""verify_0902d.py —— v4：修掉 v3 的两个口径 bug，做干净的频段对照。

v3 的 bug：
1. LTAS 基线用了**全部帧**（含静音帧）的中位 → 每一点都显示 6kHz「+30dB」，
   那只是「有语音 vs 全是静音」的差，不是缺陷。改为只用**有话帧**做基线。
2. 静音段统计的下标乘错，输出不是秒。改为显式按秒累计。
"""
import os
import subprocess

import numpy as np
import soundfile as sf

MP3 = "/mnt/tts/output/202610190902/202610190902-1.mp3"
WORK = "/mnt/tts/diag/v0902"
SR = 16000

POINTS = [
    (0, "开头 怪音"), (5, "磁音"), (14, "音一个字 没了"), (29, "吃字 一个"),
    (44, "额"), (58, "额"), (64, "兹 音"), (73, "吃字"), (83, "(无标注)"),
    (88, "(无标注)"), (100, "啊"), (115, "没音 一次"), (127, "额"),
    (135, "兹 音"), (142, "额"),
]
BANDS = [(100, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 6400),
         (6400, 7900)]


def to16k(path, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", out], check=True)
    return out


def frames(x, n=2048, hop=1024):
    w = np.hanning(n)
    out = []
    for i in range(0, len(x) - n, hop):
        out.append(np.abs(np.fft.rfft(x[i:i + n] * w)) ** 2)
    return np.array(out), n, hop


def band_db(S, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return 10 * np.log10(np.sum(S[:, m], axis=1) + 1e-12)


def main():
    wav = os.path.join(WORK, "x.wav")
    if not os.path.exists(wav):
        to16k(MP3, wav)
    x, sr = sf.read(wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)

    S, n, hop = frames(x)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    t_of = np.arange(len(S)) * hop / sr          # 每帧中心时间
    rms = np.sqrt(np.mean(S, axis=1))
    act = rms > np.percentile(rms, 60)           # 有话帧
    print("总帧 %d，有话帧 %d（阈值 %.5f）"
          % (len(S), int(act.sum()), float(np.percentile(rms, 60))))

    # 全篇有话帧的频段能量分布（归一化到总能量）
    tot = S[act].sum()
    base = {}
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        base[(lo, hi)] = float(S[act][:, m].sum() / tot)

    print("\n=== 每个点的频段能量占比 vs 全篇有话帧（单位 dB，正=该点更突出）===")
    print("  %-8s %-12s " % ("时间", "员工标注") +
          " ".join("%d-%dk" % (lo // 1000, hi // 1000) for lo, hi in BANDS))
    for t, label in POINTS:
        sel = act & (np.abs(t_of - t) <= 1.5)
        if sel.sum() < 5:
            print("  %-8.1f %-12s （附近没有足够有话帧）" % (t, label))
            continue
        tot_p = S[sel].sum()
        row = []
        for lo, hi in BANDS:
            m = (freqs >= lo) & (freqs < hi)
            v = float(S[sel][:, m].sum() / tot_p)
            row.append(10 * np.log10(v / base[(lo, hi)] + 1e-12))
        print("  %-8.1f %-12s " % (t, label) +
              " ".join("%+6.1f" % v for v in row))

    # 静音段（正确按秒）
    print("\n=== 全篇 >0.4s 的静音段（20ms 帧 RMS < -45dBFS）===")
    hop2 = int(0.02 * sr)
    nb = len(x) // hop2
    r2 = np.array([np.sqrt(np.mean(x[i * hop2:(i + 1) * hop2] ** 2) + 1e-12)
                   for i in range(nb)])
    q = 20 * np.log10(r2) < -45
    runs = []
    i = 0
    while i < nb:
        if q[i]:
            j = i
            while j < nb and q[j]:
                j += 1
            if (j - i) * 0.02 >= 0.4:
                runs.append((i * 0.02, j * 0.02))
            i = j
        else:
            i += 1
    print("  共 %d 段，总长 %.1fs（占 %.1f%%）"
          % (len(runs), sum(b - a for a, b in runs),
             100 * sum(b - a for a, b in runs) / (len(x) / sr)))
    near = [r for r in runs if any(abs(r[0] - t) <= 2.5 for t, _ in POINTS)]
    print("  员工反馈点 2.5s 内的长静音段: %s"
          % (["%.2f~%.2f(%.2fs)" % (r[0], r[1], r[1] - r[0]) for r in near]
             or "无"))

    # 每点前后 1s 的最小能量（掉字候选）
    print("\n=== 每点 ±1s 内的 20ms 帧最低能量（dBFS）===")
    for t, label in POINTS:
        a, b = int(max(0, t - 1) / 0.02), int(min(nb - 1, t + 1) / 0.02)
        seg = 20 * np.log10(r2[a:b])
        print("  %-8.1f %-12s min=%.1f  p10=%.1f  p50=%.1f"
              % (t, label, float(seg.min()), float(np.percentile(seg, 10)),
                 float(np.median(seg))))


if __name__ == "__main__":
    main()
