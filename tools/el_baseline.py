#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""🔴 建立 EL 案例的**统计基线**，把"像不像"变成"偏离几个 σ"。

## 为什么必须有这个
过去所有常量都是拿 **1~2 条** EL 案例定的：
    DEFAULT_F0   = 197.7   ← 13 条男声的均值，但没看过 σ
    DEFAULT_CPS  = 8.37    ← 8197 单条实测
    频段目标      ← 8197 + 1a6c 两条的均值
样本小到没法判断"我们差 0.01 到底是达标还是没达标"。

而验收标准是「员工听不出差别」—— 这在统计上等价于：
    **我们成品的特征，落在 EL 案例的特征分布之内。**
36 条案例（男 13 / 女 22）全是 EL 生成的，就是一份现成的 EL 分布样本。
以前只拿它当参照物比一比，没当成分布用，这是最大的浪费。

## 输出
每个维度给出 EL 的 均值 / 标准差 / 最小~最大 / 样本数，
再把我们的成品喂进来，算 **偏离几个 σ**（|z|<=1 算达标，>2 是要优先修的）。

口径（改一处要改全部，否则数字不可比）：
    F0      pyin / 16k / fmin60 fmax400 / **前 60s**
    频段    librosa stft / 24k / **全篇**（分窗处理后前 60s 不是代表样本）
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import librosa

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")

from fix_chain import f0_median, load16  # noqa: E402

CASES_JSON = "/mnt/tts/data/cases.json"
CASE_DIR = "/mnt/tts/data/cases"
BANDS = [(60, 150), (150, 300), (300, 500), (500, 1000),
         (1000, 2000), (2000, 4000), (4000, 8000)]

# 我们自己的成品（跑完 fix_chain 的），用来算偏离 σ
OURS = [
    ("v7 分窗变调", "/mnt/tts/diag/x8197/m_v7.wav"),
    ("v5 全局变调", "/mnt/tts/diag/x8197/m_v5.wav"),
]


def to24(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "24000", "-ac", "1", t], check=True)
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y


def f0_profile(path, win=60.0):
    """F0 曲线：中位 / 极差 / 线性斜率（Hz per 分钟）。"""
    y, sr = load16(path)
    n = int(win * sr)
    vals = []
    i = 0
    while i + int(20 * sr) < len(y):
        seg = y[i:i + n]
        if len(seg) < int(20 * sr):
            break
        f = f0_median(seg, sr, win=1e9)
        if f:
            vals.append(f)
        i += n
    if len(vals) < 2:
        return None
    v = np.array(vals)
    t = np.arange(len(v)) * (win / 60.0)      # 单位：分钟
    slope = float(np.polyfit(t, v, 1)[0])
    return {"f0_med": float(np.median(v)),
            "f0_range": float(v.max() - v.min()),
            "f0_slope": slope}


def band_feats(path):
    y = to24(path)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=24000, n_fft=2048)
    tot = S.sum() + 1e-12
    out = {}
    for lo, hi in BANDS:
        out["b%d_%d" % (lo, hi)] = float(S[(f >= lo) & (f < hi)].sum() / tot)
    y16, _ = load16(path)
    out["flat"] = float(librosa.feature.spectral_flatness(
        y=y16[:60 * 16000], hop_length=256)[0].mean())
    return out


def voiced_ratio(path):
    y, sr = load16(path)
    hop = int(0.01 * sr)
    n = len(y) // hop
    env = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean()
                            for i in range(n)]))
    thr = np.percentile(env, 30) * 2.2
    return float((env > thr).mean())


def feats_of(path):
    d = {}
    p = f0_profile(path)
    if p:
        d.update(p)
    d.update(band_feats(path))
    d["voiced"] = voiced_ratio(path)
    return d


def stats(rows, key):
    v = np.array([r[key] for r in rows if r.get(key) is not None],
                 dtype=float)
    if len(v) < 2:
        return None
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max()), "n": len(v)}


if __name__ == "__main__":
    cases = json.load(open(CASES_JSON, encoding="utf-8"))
    cases = cases.values() if isinstance(cases, dict) else cases
    males = [c for c in cases if (c.get("gender") or "") == "male"]
    print("男声 EL 案例：%d 条" % len(males))

    rows = []
    for c in males:
        p = os.path.join(CASE_DIR, c["id"], "merged.wav")
        if not os.path.exists(p):
            print("  跳过（无 merged.wav）:", c["id"])
            continue
        try:
            d = feats_of(p)
            d["_id"] = c["id"]
            rows.append(d)
            print("  %s F0=%.1f 极差=%.1f 斜率=%+.1f"
                  % (c["id"], d.get("f0_med", 0), d.get("f0_range", 0),
                     d.get("f0_slope", 0)))
        except Exception as e:
            print("  FAIL", c["id"], repr(e)[:80])

    keys = [k for k in rows[0] if not k.startswith("_")]
    print("\n" + "=" * 78)
    print("EL 男声基线（%d 条）" % len(rows))
    print("=" * 78)
    base = {}
    for k in keys:
        s = stats(rows, k)
        if s:
            base[k] = s
            print("%-12s 均值 %8.4f  σ %8.4f  范围 %8.4f ~ %8.4f  (n=%d)"
                  % (k, s["mean"], s["sd"], s["min"], s["max"], s["n"]))

    print("\n" + "=" * 78)
    print("我们的成品偏离几个 σ（|z|<=1 达标，>2 优先修）")
    print("=" * 78)
    for name, p in OURS:
        if not os.path.exists(p):
            continue
        d = feats_of(p)
        print("\n--- %s ---" % name)
        zs = []
        for k in keys:
            if k not in base or d.get(k) is None:
                continue
            s = base[k]
            z = (d[k] - s["mean"]) / (s["sd"] if s["sd"] > 1e-9 else 1e-9)
            zs.append((abs(z), k, d[k], z))
        zs.sort(reverse=True)
        for az, k, val, z in zs:
            flag = "🔴" if az > 2 else ("⚠️ " if az > 1 else "✅")
            print("  %s %-12s 我们 %8.4f  EL %8.4f  z=%+6.2f"
                  % (flag, k, val, base[k]["mean"], z))

    out = os.path.join(HERE, "el_baseline_male.json")
    json.dump(base, open(out, "w"), ensure_ascii=False, indent=1)
    print("\n基线已存:", out)
