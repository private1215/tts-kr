#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L1 快筛 v2：隔离「重采样」vs「atempo」，并按真实 24k 评估（第30轮）

v1 结论：
  - 现役两次 atempo（A）比融合一次（B）只差 6%（相对）
  - 但**变调那一遍本身就掉 33% 平坦度**
  - 关键算术：变调 R=0.92 会把时长拉长 1/0.92=1.087×，
    语速锚又要放慢 1.10× → **两步 WSOLA 几乎互相抵消**，
    融合后 atempo = 1/(R·fs) = 1/(0.92×1.10) = **0.988**（几乎不动）

v2 要回答：掉的那 33% 到底是谁的锅？
  RES   : 只 asetrate+aresample（不补 atempo）→ 纯重采样的损伤
  T_ONLY: 只 atempo（不重采样）              → 纯 WSOLA 的损伤
  A     : 现役两次                          → 合计
  B     : 融合一次
  B_SOXR: 融合 + soxr 高精度重采样           → 换重采样器能不能救回来

## 用法
    ./venv/bin/python tools/atempo_ab2.py --sec 120 --fs 1.10 --r 0.92 --sr 24000
"""
import argparse
import json
import os
import subprocess
import tempfile

import numpy as np

DIAG = "/mnt/tts/diag/muffle"
OUT = os.path.join(DIAG, "atempo_ab2.json")
EL24 = os.path.join(DIAG, "el24.wav")

try:
    import librosa
except Exception:
    librosa = None


def load(p, sr_out):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ac", "1",
                          "-ar", str(sr_out), "-f", "f32le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def ff(src, dst, af):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-af", af, "-f", "wav", dst], check=True)


def flat_pg(y, sr):
    """probe_g 口径：16k / 前 60s / hop 256。"""
    y16 = load16_from_array(y, sr)
    if librosa is None:
        return None
    return float(librosa.feature.spectral_flatness(
        y=y16[:60 * 16000], hop_length=256)[0].mean())


def load16_from_array(y, sr):
    if sr == 16000:
        return y
    tmp = os.path.join(tempfile.mkdtemp(dir="/mnt/tts/diag"), "t.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le",
                    "-ar", str(sr), "-ac", "1", "-i", "-",
                    "-ar", "16000", "-f", "f32le", "-"],
                   input=y.tobytes(), stdout=subprocess.PIPE, check=True)
    raw = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le",
                          "-ar", str(sr), "-ac", "1", "-i", "-",
                          "-ar", "16000", "-f", "f32le", "-"],
                         input=y.tobytes(),
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def read_wav16(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ac", "1",
                          "-ar", "16000", "-f", "f32le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def hf(y16):
    n, hop = 2048, 512
    nfr = (len(y16) - n) // hop
    idx = np.arange(nfr)[:, None] * hop + np.arange(n)[None, :]
    fr = y16[idx] * np.hanning(n)[None, :]
    sp = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / 16000)
    return float(sp[:, (f >= 4000) & (f < 8000)].sum() / sp.sum() * 100)


def measure(tag, path, sr):
    y16 = read_wav16(path)
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], stdout=subprocess.PIPE,
        text=True).stdout.strip())
    env = librosa.onset.onset_strength(y=y16, sr=16000, hop_length=256)
    pk = librosa.util.peak_pick(env, pre_max=3, post_max=3,
                                     pre_avg=3, post_avg=5,
                                     delta=0.10, wait=5)
    return {"tag": tag, "dur": dur, "flat": flat_pg16(y16),
            "hf4_8": hf(y16), "onsets": float(len(pk)),
            "onset_str": float(env.mean())}


def flat_pg16(y16):
    return float(librosa.feature.spectral_flatness(
        y=y16[:60 * 16000], hop_length=256)[0].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=float, default=120.0)
    ap.add_argument("--fs", type=float, default=1.10)
    ap.add_argument("--r", type=float, default=0.92)
    ap.add_argument("--sr", type=int, default=24000)
    a = ap.parse_args()
    sr = a.sr

    d = tempfile.mkdtemp(dir="/mnt/tts/diag")
    head = os.path.join(d, "head.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", EL24,
                    "-t", str(a.sec), "-ar", str(sr), "-ac", "1",
                    "-f", "wav", head], check=True)

    arms = [("EL 原样", head, None)]
    P = "asetrate=%d,aresample=%d" % (int(round(sr * a.r)), sr)

    # RES：只重采样（不补 atempo）
    p = os.path.join(d, "res.wav")
    ff(head, p, P)
    arms.append(("RES 只重采样", p, None))

    # T_ONLY：只 atempo（融合后应有的那个值）
    t_fused = 1.0 / (a.r * a.fs)
    p = os.path.join(d, "tonly.wav")
    ff(head, p, "atempo=%.6f" % t_fused)
    arms.append(("T_ONLY atempo=%.3f" % t_fused, p, None))

    # A 现役两次
    p1 = os.path.join(d, "a1.wav")
    ff(head, p1, "atempo=%.6f" % (1.0 / a.fs))
    p = os.path.join(d, "a2.wav")
    ff(p1, p, P + ",atempo=%.6f" % (1.0 / a.r))
    arms.append(("A 现役(2次)", p, None))

    # B 融合
    p = os.path.join(d, "b.wav")
    ff(head, p, P + ",atempo=%.6f" % t_fused)
    arms.append(("B 融合(1次)", p, None))

    # B_SOXR：soxr 高精度重采样
    P2 = ("asetrate=%d,aresample=%d:resampler=soxr:precision=28"
          % (int(round(sr * a.r)), sr))
    p = os.path.join(d, "bs.wav")
    ff(head, p, P2 + ",atempo=%.6f" % t_fused)
    arms.append(("B_SOXR 融合+soxr", p, None))

    rows = [measure(t, p_, sr) for t, p_, _ in arms]
    print("\n%-22s %8s %9s %8s %8s %8s"
          % ("版本", "时长s", "平坦度", "4k~8k%", "起音数", "起音强"))
    for r in rows:
        print("%-22s %8.1f %9.5f %8.4f %8.0f %8.3f"
              % (r["tag"], r["dur"], r["flat"], r["hf4_8"],
                 r["onsets"], r["onset_str"]))

    e = rows[0]["flat"]
    print("\n相对 EL 的平坦度保持率（越高越好）：")
    for r in rows[1:]:
        print("   %-22s %5.1f%%" % (r["tag"], r["flat"] / e * 100))

    json.dump(rows, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("\n→ " + OUT)


if __name__ == "__main__":
    main()
