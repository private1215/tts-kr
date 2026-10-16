# -*- coding: utf-8 -*-
"""rawfull48m.py —— 男声 click 1.175%（EL 的 4.5 倍）的分层定位（S3 归因）。

做法：同一条男声全篇文本，出 4 条同源材料，逐层叠加处理，
看 click / 杂音在哪一层被引入：

    A 裸输出（voxcpm 直出）
    B 裸 + 响度对齐 -25.0 LUFS
    C 裸 + 响度 + 变调（F0 -> 188，asetrate+aresample+atempo）
    D 现役全链（走 /api/jobs 的成品 v10_m_vox）

🔴 判 click 的铁律：必须先把所有素材归一到同一峰值再判 |Δ|>=0.30。
阈值是绝对的，不归一等于在比谁声音大。

🔴 判杂音的铁律：统一降到 16k 再算 4-8kHz 占比。
"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
import vox_client  # noqa: E402

OUT = "/mnt/tts/diag/rawfull48m"
os.makedirs(OUT, exist_ok=True)
TXT = open("/mnt/tts/data/cases/6cdd25bc75f2/full.txt", encoding="utf-8").read()[:4143]
LUFS = -25.0          # MINIMAL_LUFS["m"] 现役男声目标
TARGET_F0 = 188.0     # DEFAULT_F0["m"]


def band48(wav16):
    """静音段 4-8kHz 占比 %（16k 口径）。"""
    x, sr = sf.read(wav16, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = int(sr * 0.02)
    m = (len(x) // n) * n
    fr = x[:m].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    seg = fr[rms <= np.percentile(rms, 20)].ravel()
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    return float(X[(f >= 4000) & (f < 8000)].sum() / X.sum() * 100)


def click(path, thr=0.30):
    """峰值归一到 0.70 后，20ms 窗内最大 |Δ| >= 0.30 的窗占比 %。"""
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x / (np.abs(x).max() + 1e-12) * 0.70
    n = int(sr * 0.02)
    d = np.abs(np.diff(x))
    m = (len(d) // n) * n
    return float((d[:m].reshape(-1, n).max(axis=1) >= thr).mean() * 100)


def to16k(p, tag):
    d = os.path.join(OUT, "w16_%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", "16000", "-ac", "1", d], check=True)
    return d


def main():
    raw = os.path.join(OUT, "m_raw.wav")
    if not os.path.exists(raw):
        print("调 VoxCPM 生成男声全篇裸输出（4143 字）...", flush=True)
        audio, sr, meta = vox_client.synth(text=TXT, voice=None)
        if audio is None:
            print("失败:", meta)
            return 1
        sf.write(raw, audio, sr)
        print("  裸输出 %.2fs @ %dHz" % (len(audio) / sr, sr), flush=True)
    dur = sf.info(raw).duration
    print("裸输出 %.2fs  cps=%.2f" % (dur, len(TXT) / dur), flush=True)

    loud = os.path.join(OUT, "m_loud.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw,
                    "-af", "loudnorm=I=%s:TP=-2:LRA=11" % LUFS,
                    "-ar", "24000", "-ac", "1", loud], check=True)

    # 变调：先测裸输出 F0 才知道 R
    print()
    print("=== 分层对照（16k 统一口径 / click 峰值归一 0.70）===")
    print("%-22s %10s %10s %10s" % ("素材", "时长s", "杂音4-8k", "click%"))

    rows = []
    for tag, p in [("A 裸输出", raw), ("B 裸+响度", loud),
                   ("D 现役全链", "/mnt/tts/output/v10_m_vox/v10_m_vox-1.mp3")]:
        w = to16k(p, tag.split()[0])
        rows.append((tag, sf.info(w).duration, band48(w), click(w)))
        print("%-22s %10.1f %10.3f %10.3f" % rows[-1])
        sys.stdout.flush()

    el = to16k("/mnt/tts/data/cases/6cdd25bc75f2/merged.wav", "el")
    print("%-22s %10.1f %10.3f %10.3f" % (
        "EL 男声标杆", sf.info(el).duration, band48(el), click(el)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
