#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把员工说的「电子音 / 吃字 / 卡字」三个症状**量化**（2026-10-15 第30轮）

用户最新反馈：
> 电子音，吃字，卡字——这些是目前主要问题
> 感情和说话带气声都很自然了

所以本脚本只测这三项：

1. **吃字** = CER（ASR 转写 vs 原文的字符错误率）。同时测 EL 标杆做对照。
2. **卡字** = 时域突变：帧间电平暴跌（硬切）、数字零帧、以及
   **音节层面的异常短音段**（过短的发声段 = 被吞/被卡）。
3. **电子音** = 频谱平坦度（flatness）沿时间轴的分布 + 音色相似度（campplus）。
   平坦度低 = 过于周期性/过于干净 = 电子味。

用法（服务器）：
    ./venv/bin/python tools/symptom_probe.py
"""
import json
import os
import subprocess
import sys

import numpy as np

SR = 24000
DIAG = "/mnt/tts/diag"
OUT = os.path.join(DIAG, "symptom_20261015.json")

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")

ITEMS = [
    ("EL 男标杆", "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"),
    ("修复后男声", "/mnt/tts/output/longprobe-r30fix-6cdd-2/longprobe-r30fix-6cdd-1.mp3"),
    ("改坏那版", "/mnt/tts/output/longprobe-r30vox-6cdd-3/longprobe-r30vox-6cdd-1.mp3"),
]
TEXT = "/mnt/tts/data/cases/6cdd25bc75f2/full.txt"


def load(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path,
                          "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def flatness_profile(x, sr=SR, win_ms=30.0):
    """沿时间轴的频谱平坦度（几何均值/算术均值）。"""
    n = int(sr * win_ms / 1000.0)
    nfr = len(x) // n
    if nfr < 2:
        return 0.0, 0.0
    fr = x[:nfr * n].reshape(nfr, n)
    spec = np.abs(np.fft.rfft(fr * np.hanning(n)[None, :], axis=1)) ** 2
    spec = spec[:, 1:] + 1e-20
    gm = np.exp(np.log(spec).mean(axis=1))
    am = spec.mean(axis=1)
    fl = gm / am
    voiced = fr.std(axis=1) > (fr.std(axis=1).max() * 0.02)
    fv = fl[voiced] if voiced.sum() > 5 else fl
    return float(np.median(fv)), float(np.std(fv))


def short_segments(x, sr=SR, thresh_db=-40.0):
    """发声段里「过短」的碎片占比 —— 太短 = 被吞/被卡。"""
    n = int(sr * 0.01)
    nfr = len(x) // n
    fr = x[:nfr * n].reshape(nfr, n)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-30)
    db = 20 * np.log10(rms + 1e-30)
    on = db > thresh_db
    edges = np.diff(on.astype(np.int8))
    st = list(np.where(edges == 1)[0] + 1)
    en = list(np.where(edges == -1)[0] + 1)
    if on[0]:
        st.insert(0, 0)
    if on[-1]:
        en.append(len(on))
    durs = np.array([b - a for a, b in zip(st, en) if b > a]) * 0.01
    if len(durs) == 0:
        return 0, 0.0, 0.0
    return (len(durs), float(np.median(durs)),
            float(100.0 * (durs < 0.08).mean()))


def main():
    out = []
    text = open(TEXT, encoding="utf-8").read().strip() if os.path.exists(TEXT) else ""
    for name, p in ITEMS:
        if not os.path.exists(p):
            print("缺:", p)
            continue
        x = load(p)
        n = int(SR * 0.01)
        nfr = len(x) // n
        fr = x[:nfr * n].reshape(nfr, n)
        rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-30)
        db = 20 * np.log10(rms + 1e-30)
        d = np.diff(db)
        fl_med, fl_sd = flatness_profile(x)
        nseg, seg_med, short_pct = short_segments(x)
        r = {
            "name": name,
            "dur_s": round(len(x) / float(SR), 1),
            "cps": round(len(text) / (len(x) / float(SR)), 2) if text else None,
            "吃字_硬切数": int((d < -60).sum()),
            "吃字_真零帧": int((db < -200).sum()),
            "卡字_发声段数": nseg,
            "卡字_段长中位_s": round(seg_med, 3),
            "卡字_超短段占比%": round(short_pct, 1),
            "电子音_平坦度中位": round(fl_med, 5),
            "电子音_平坦度σ": round(fl_sd, 5),
        }
        out.append(r)
        print(json.dumps(r, ensure_ascii=False))

    # CER（需要 faster-whisper，失败就跳过）
    try:
        import asr_check
        for r, (name, p) in zip(out, ITEMS[:len(out)]):
            try:
                res = asr_check.check(p, text)
                r["吃字_CER"] = res.get("cer")
            except Exception as e:
                r["吃字_CER"] = "err:%s" % repr(e)[:60]
    except Exception as e:
        print("asr_check 不可用:", repr(e)[:100])

    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("\n→ " + OUT)


if __name__ == "__main__":
    main()
