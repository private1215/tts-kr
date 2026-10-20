# -*- coding: utf-8 -*-
"""verify_0902c.py —— 逐点验证 v3：声谱细查 + 句末嗯全量清单对照。

v3 补三件事：
1. 打印**全量**句末粘连嗯清单（v2 只看到前 10 条），和员工 15 个点做对照；
2. 每个点 ±1.5s 的 LTAS（平均谱），找窄带峰（磁音/兹音的候选物理来源）；
3. 文件头 3s 的 10ms 能量包络（「开头怪音」候选：起音太轻/淡入）。
"""
import json
import os
import re
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


def to16k(path, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", out], check=True)
    return out


def ltas(x, sr, t0, t1):
    a, b = int(t0 * sr), int(t1 * sr)
    seg = x[a:b]
    if len(seg) < 512:
        return None, None
    w = np.hanning(1024)
    acc = None
    cnt = 0
    for i in range(0, len(seg) - 1024, 512):
        S = np.abs(np.fft.rfft(seg[i:i + 1024] * w)) ** 2
        acc = S if acc is None else acc + S
        cnt += 1
    if acc is None or cnt < 3:
        return None, None
    acc /= cnt
    freqs = np.fft.rfftfreq(1024, 1.0 / sr)
    db = 10 * np.log10(acc + 1e-12)
    return freqs, db


def main():
    wav = os.path.join(WORK, "x.wav")
    if not os.path.exists(wav):
        to16k(MP3, wav)
    x, sr = sf.read(wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)

    # ---- 1. 全篇 LTAS 基线 ----
    f_all, db_all = None, None
    accs, cnts = [], 0
    w = np.hanning(2048)
    for i in range(0, len(x) - 2048, 2048):
        S = np.abs(np.fft.rfft(x[i:i + 2048] * w)) ** 2
        accs.append(S)
        cnts += 1
    accs = np.array(accs)
    med = 10 * np.log10(np.median(accs, axis=0) + 1e-12)
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)

    print("=== 每个反馈点的 LTAS 窄带峰（±1.5s，只列比全篇同频段中位高 6dB 以上的峰）===")
    for t, label in POINTS:
        fr, db = ltas(x, sr, max(0, t - 1.5), t + 1.5)
        if fr is None:
            continue
        # 与全篇中位比
        interp = np.interp(fr, freqs, med)
        excess = db - interp
        idx = np.argsort(excess)[::-1][:6]
        peaks = [(round(float(fr[k]), 1), round(float(excess[k]), 1))
                 for k in idx if excess[k] > 6.0]
        # 频段能量比
        def band(lo, hi):
            m = (fr >= lo) & (fr < hi)
            return 10 * np.log10(np.sum(10 ** (db[m] / 10.0)) + 1e-12)
        low = band(100, 500)
        mid = band(500, 2000)
        hi = band(2000, 6000)
        print("\n  [%6.1fs] %s" % (t, label))
        print("     低频(0.1-0.5k)=%.1f  中频(0.5-2k)=%.1f  高频(2-6k)=%.1f dB"
              % (low, mid, hi))
        if peaks:
            print("     ⚠ 窄带超出全篇中位: " +
                  ", ".join("%.0fHz(+%.1fdB)" % p for p in peaks))
        else:
            print("     无窄带异常（谱形与全篇一致）")

    # ---- 2. 文件头 3s 的 10ms 包络 ----
    print("\n=== 文件头 3.0s 的 10ms 能量包络（dBFS）===")
    hop = int(0.01 * sr)
    n = int(3.0 / 0.01)
    vals = []
    for i in range(n):
        seg = x[i * hop:(i + 1) * hop]
        vals.append(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)
                    if len(seg) else -120)
    line = " ".join("%.0f" % v for v in vals[:60])
    print("  0.0-0.6s: " + line)
    line = " ".join("%.0f" % v for v in vals[60:180])
    print("  0.6-1.8s: " + line)
    line = " ".join("%.0f" % v for v in vals[180:300])
    print("  1.8-3.0s: " + line)

    # ---- 3. 全篇静音段（找「没音」候选）----
    print("\n=== 全篇 >0.35s 的静音段（RMS < -45dBFS）===")
    hop2 = int(0.02 * sr)
    rms = np.sqrt(np.convolve(x ** 2, np.ones(hop2) / hop2, mode="same"))
    db = 20 * np.log10(rms + 1e-12)
    quiet = db < -45
    runs = []
    i = 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * 0.02 >= 0.35:
                runs.append((i * 0.02, j * 0.02))
            i = j
        else:
            i += 1
    print("  共 %d 段" % len(runs))
    for a, b in runs[:40]:
        print("    %7.2f ~ %7.2f  (%.2fs)" % (a, b, b - a))
    near = [r for r in runs if any(abs(r[0] - t) <= 2.5 for t, _ in POINTS)]
    print("  员工反馈点 2.5s 内的静音段: %s" %
          (["%.2f~%.2f" % r for r in near] or "无"))


if __name__ == "__main__":
    main()
