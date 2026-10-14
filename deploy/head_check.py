#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查盲听切片开头是否是「硬切爆音」伪影。

员工口头反馈「开头有点别扭，别的很接近了」。切片是用 `ffmpeg -ss 40` 硬切的，
如果正好切在语音中间，第一个样本会是一个非零突变 → 咔哒声(pop)。
那种"别扭"是**我切片的问题，不是音频的问题**，必须先排除掉，否则会去改本来没问题的链路。
"""
import subprocess
import tempfile
import os
import numpy as np
import soundfile as sf


def load16(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", t], check=True)
    y, sr = sf.read(t, dtype="float32")
    os.unlink(t)
    return y, sr


def probe(path):
    y, sr = load16(path)
    out = {"path": os.path.basename(path)}
    # 首样本幅值 vs 前 50ms / 整体 RMS
    first = abs(float(y[0]))
    rms_head = float(np.sqrt((y[:int(0.05 * sr)] ** 2).mean()))
    rms_all = float(np.sqrt((y ** 2).mean()))
    out["first_amp"] = round(first, 5)
    out["rms_all"] = round(rms_all, 5)
    # 硬切判据：首样本幅值 > 5× 整体 RMS 且 > 10× 前 50ms RMS
    out["first_vs_all"] = round(first / max(rms_all, 1e-9), 1)
    out["pop"] = (first > 5 * rms_all) and (first > 10 * rms_head)
    # 起音斜率：前 5ms 最大跳变
    d = np.abs(np.diff(y[:int(0.005 * sr)]))
    out["max_step_5ms"] = round(float(d.max()) / max(rms_all, 1e-9), 1)
    # 最开始是不是静音（切在静音处就没爆音）
    out["head_sil_ms"] = None
    thr = max(rms_all * 0.1, 1e-5)
    idx = np.where(np.abs(y[:int(0.5 * sr)]) > thr)[0]
    out["first_loud_sample_ms"] = round(float(idx[0]) / sr * 1000, 1) if len(idx) else None
    return out


if __name__ == "__main__":
    import sys
    keys = ["path", "first_amp", "rms_all", "first_vs_all",
            "max_step_5ms", "first_loud_sample_ms", "pop"]
    print("\t".join(keys))
    for p in sys.argv[1:]:
        try:
            r = probe(p)
            print("\t".join(str(r.get(k, "")) for k in keys))
        except Exception as e:
            print("FAIL", p, repr(e)[:120])
