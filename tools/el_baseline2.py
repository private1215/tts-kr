# -*- coding: utf-8 -*-
"""EL 频段基线（清洗版）—— 逐条缓存，脏数据可剔除，以后不用重算。

为什么要重做一遍 el_baseline.py：
    第一版把每条的统计都丢了，只留了聚合值，发现脏数据（c3117b7b0fb1
    F0=275.7 却被标成 male）后没法离线重算，只能整篇重跑。这次逐条落盘。

跳过了 pyin（那是 20 分钟的大头），F0 逐条值从 elbase.log 解析补进来。
口径必须和 el_baseline.py 完全一致：librosa stft / 24k / 全篇。
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import librosa
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")

from fix_chain import load16  # noqa: E402

CASES_JSON = "/mnt/tts/data/cases.json"
CASE_DIR = "/mnt/tts/data/cases"
CACHE = os.path.join(HERE, "el_feats.json")
BANDS = [(60, 150), (150, 300), (300, 500), (500, 1000),
         (1000, 2000), (2000, 4000), (4000, 8000)]

# F0 与性别明显不符的案例（男声区间约 185~205Hz）
BAD_MALE = {"c3117b7b0fb1"}


def to24(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "24000", "-ac", "1", t], check=True)
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y


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


def parse_f0_log(path):
    """从 el_baseline.py 的日志里把逐条 F0 捞回来，省一次 20 分钟的 pyin。"""
    out = {}
    if not os.path.exists(path):
        return out
    pat = re.compile(r"^\s*([0-9a-f]{12})\s+F0=([\d.]+)\s+极差=([\d.]+)"
                     r"\s+斜率=([+-][\d.]+)")
    for line in open(path, encoding="utf-8"):
        m = pat.match(line)
        if m:
            out[m.group(1)] = {"f0_med": float(m.group(2)),
                               "f0_range": float(m.group(3)),
                               "f0_slope": float(m.group(4))}
    return out


def main():
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) \
        else {}
    f0s = parse_f0_log(os.path.join(HERE, "elbase.log"))
    cases = json.load(open(CASES_JSON, encoding="utf-8"))
    cases = cases.values() if isinstance(cases, dict) else cases

    for c in cases:
        cid, g = c["id"], (c.get("gender") or "")
        if cid in cache:
            continue
        p = os.path.join(CASE_DIR, cid, "merged.wav")
        if not os.path.exists(p):
            continue
        try:
            d = {"gender": g}
            d.update(band_feats(p))
            d["voiced"] = voiced_ratio(p)
            d.update(f0s.get(cid, {}))
            cache[cid] = d
            print("  %s %-6s flat=%.4f voiced=%.4f"
                  % (cid, g, d["flat"], d["voiced"]), flush=True)
            json.dump(cache, open(CACHE, "w"), ensure_ascii=False, indent=1)
        except Exception as e:
            print("  FAIL", cid, repr(e)[:90], flush=True)

    print("\n缓存条数 %d -> %s" % (len(cache), CACHE))

    keys = ["f0_med", "f0_range", "f0_slope", "flat", "voiced"] + \
           ["b%d_%d" % b for b in BANDS]
    for g in ("male", "female"):
        for clean in (False, True):
            rows = [v for k, v in cache.items() if v.get("gender") == g]
            if clean:
                rows = [v for k, v in cache.items()
                        if v.get("gender") == g and k not in BAD_MALE
                        and not (g == "male" and v.get("f0_med", 0) > 230)]
            tag = "%s %s(n=%d)" % (g, "清洗后" if clean else "原始", len(rows))
            print("\n" + "=" * 76)
            print("EL %s 频段基线" % tag)
            print("=" * 76)
            for k in keys:
                v = np.array([r[k] for r in rows if r.get(k) is not None],
                             dtype=float)
                if len(v) < 2:
                    continue
                print("  %-12s 均值 %8.4f  sigma %8.4f  范围 %8.4f ~ %8.4f"
                      % (k, v.mean(), v.std(ddof=1), v.min(), v.max()))
    return cache


if __name__ == "__main__":
    main()
