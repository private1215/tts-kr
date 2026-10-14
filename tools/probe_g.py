# -*- coding: utf-8 -*-
"""通用 σ 探针：男女共用，基线从 JSON 读，不再把数字抄进代码。

为什么要有它：
    eq_search6.py 的 BASE 是**手抄**的男声清洗值，女声根本没有对应物。
    抄进去的数字会腐烂（男声已经发生过一次：原始基线把脏数据算进去，
    代码里却是清洗值，两边对不上没人发现）。现在统一读 el_base_clean_*.json，
    el_baseline_clean.py 重算一次就能刷新全部脚本。

用法：
    python probe_g.py female /path/a.wav /path/b.wav
    python probe_g.py male m_v10.wav
"""
import json
import os
import subprocess
import sys
import tempfile

import librosa
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = [(60, 150), (150, 300), (300, 500), (500, 1000),
         (1000, 2000), (2000, 4000), (4000, 8000)]
ORDER = ["b60_150", "b150_300", "b300_500", "b500_1000",
         "b1000_2000", "b2000_4000", "b4000_8000", "flat", "voiced"]
NAME = {"b60_150": "60~150", "b150_300": "150~300", "b300_500": "300~500",
        "b500_1000": "500~1k", "b1000_2000": "1k~2k",
        "b2000_4000": "2k~4k", "b4000_8000": "4k~8k",
        "flat": "平坦度", "voiced": "发声占比"}


def base_path(gender, d="/mnt/tts/diag/x8197"):
    return os.path.join(d, "el_base_clean_%s.json" % gender)


def load_base(gender):
    p = base_path(gender)
    if not os.path.exists(p):
        sys.exit("缺基线 %s，先跑 el_baseline_clean.py" % p)
    return json.load(open(p, encoding="utf-8"))


def to24(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "24000", "-ac", "1", t], check=True)
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y


def to16(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", t], check=True)
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y


def measure(path):
    y = to24(path)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=24000, n_fft=2048)
    tot = S.sum() + 1e-12
    out = {}
    for lo, hi in BANDS:
        out["b%d_%d" % (lo, hi)] = float(S[(f >= lo) & (f < hi)].sum() / tot)
    y16 = to16(path)
    out["flat"] = float(librosa.feature.spectral_flatness(
        y=y16[:60 * 16000], hop_length=256)[0].mean())
    hop = int(0.01 * 16000)
    n = len(y16) // hop
    env = np.sqrt(np.array([(y16[i * hop:(i + 1) * hop] ** 2).mean()
                            for i in range(n)]))
    thr = np.percentile(env, 30) * 2.2
    out["voiced"] = float((env > thr).mean())
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit("用法: probe_g.py <male|female> <wav> [wav...]")
    gender, paths = sys.argv[1], sys.argv[2:]
    base = load_base(gender)
    keys = [k for k in ORDER if k in base]

    print("基线：%s  n=%d" % (base_path(gender),
                             max(base[k]["n"] for k in keys)))
    for p in paths:
        p = p if os.path.isabs(p) else os.path.join(HERE, p)
        if not os.path.exists(p):
            print("缺文件 %s" % p)
            continue
        m = measure(p)
        zs = {}
        for k in keys:
            b = base[k]
            zs[k] = (m[k] - b["mean"]) / b["sd"] if b["sd"] > 1e-12 else 0.0
        j = sum(v ** 2 for v in zs.values())
        worst = max(zs, key=lambda k: abs(zs[k]))
        print("=" * 74)
        print("%-28s  Σz²=%7.1f   最差 %s %+.2fσ"
              % (os.path.basename(p)[:28], j, NAME.get(worst, worst), zs[worst]))
        for k in keys:
            z = zs[k]
            flag = "[X]" if abs(z) > 2 else ("[!]" if abs(z) > 1.5 else "[ok]")
            print("  %-4s %-9s 我们 %9.4f  EL %9.4f  z=%+6.2f"
                  % (flag, NAME.get(k, k), m[k], base[k]["mean"], z))


if __name__ == "__main__":
    main()
