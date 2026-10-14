# -*- coding: utf-8 -*-
"""量「停顿里的绝对电平」，给 roomtone 定目标值。

## 为什么要这个脚本
2026-10-13 员工说卡顿没改善。已量到的**相对**落差是：

    女 EL 标杆 11.4 dB   vs   女 我们 23.2 dB
    男 EL 标杆 20.9 dB   vs   男 我们 28.3 dB

但相对落差不能告诉我们要把底噪铺到多少 dBFS —— **必须知道语音段本身
在什么电平、停顿段在什么电平**。本脚本就量这两个绝对数。

## 量法（与可听口径同源，别改参数）
16 kHz / 20 ms 帧 / 帧 dB：
  · 有声帧   = 帧 dB > -45
  · gap      = 内部连续静音 >= 0.25 s（去掉首尾 30 ms 边缘，避免过渡段污染）
输出：
  · speech_db 有声帧电平的 p50（语音"在哪"）
  · pause_db  所有 gap 内帧电平的 p50 / p25 / p75（停顿"在哪"）
  · drop      = speech_db - pause_db（就是员工听到的落差）

## 用法
    python tools/pause_level_abs.py 文件1.wav 文件2.wav ...
或在服务器：
    python tools/pause_level_abs.py \
      /mnt/tts/data/cases/1a6c20a60481/merged.wav /tmp/selftest/open_EL_male.wav
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.020
HOP = 0.020
DB_ON = -45.0
GAP_MIN = 0.25
EDGE = 0.030          # gap 首尾各丢 30ms，避开过渡


def load(path, tmp):
    out = os.path.join(tmp, os.path.basename(path) + ".16k.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", str(SR), "-ac", "1", out], check=True)
    x, _ = sf.read(out)
    return np.asarray(x, dtype=np.float64)


def frame_db(x):
    n = int(round(FRAME * SR))
    h = int(round(HOP * SR))
    idx = np.arange(0, max(len(x) - n + 1, 0), h)
    rms = np.array([np.sqrt(np.mean(x[i:i + n] ** 2)) + 1e-12 for i in idx])
    return 20.0 * np.log10(rms)


def measure(path, tmp):
    x = load(path, tmp)
    db = frame_db(x)
    on = db > DB_ON

    # 找 gap：连续静音段
    gaps, i = [], 0
    while i < len(on):
        if not on[i]:
            j = i
            while j < len(on) and not on[j]:
                j += 1
            dur = (j - i) * HOP          # HOP 本身就是秒/帧（0.020），不要再除 SR
            # 仅内部停顿：片头、片尾静音不用于定义句间底噪。
            if dur >= GAP_MIN and i > 0 and j < len(on):
                gaps.append((i, j))
            i = j
        else:
            i += 1

    sp = db[on]
    pf = []
    for a, b in gaps:
        e = int(round(EDGE / HOP * 1.0))          # 30ms 对应帧数
        a2, b2 = a + e, b - e
        if b2 > a2:
            pf.extend(db[a2:b2].tolist())
    pf = np.array(pf) if pf else np.array([np.nan])

    sp_med = float(np.percentile(sp, 50)) if len(sp) else float("nan")
    p_med = float(np.percentile(pf, 50))
    p25 = float(np.percentile(pf, 25))
    p75 = float(np.percentile(pf, 75))
    return {
        "file": os.path.basename(path),
        "sec": round(len(x) / SR, 1),
        "n_gap": len(gaps),
        "gap_rate": round(len(gaps) / (len(x) / SR) * 60.0, 1),
        "speech_db": round(sp_med, 1),
        "pause_db": round(p_med, 1),
        "pause_p25": round(p25, 1),
        "pause_p75": round(p75, 1),
        "drop": round(sp_med - p_med, 1),
    }


def main():
    if len(sys.argv) < 2:
        sys.exit("用法：python tools/pause_level_abs.py 文件1 [文件2 ...]")
    tmp = tempfile.mkdtemp(prefix="plvl_")
    rows = [measure(p, tmp) for p in sys.argv[1:]]
    hdr = ["文件", "时长s", "gap/分", "语音dB", "停顿dB", "p25", "p75", "落差dB"]
    print("%-34s %7s %8s %8s %8s %8s %8s %8s" % tuple(hdr))
    for r in rows:
        print("%-34s %7.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f" % (
            r["file"][:34], r["sec"], r["gap_rate"], r["speech_db"],
            r["pause_db"], r["pause_p25"], r["pause_p75"], r["drop"]))
    print("\n读法：落差 = 有声帧电平中位数 − 内部停顿帧电平中位数。")
    print("注意：本值不是局部边界跳变，不等价于卡顿听感；不应据此单独调整底噪。")


if __name__ == "__main__":
    main()
