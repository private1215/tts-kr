# -*- coding: utf-8 -*-
"""measure48.py —— 女声 L3 全篇口径统一测量（v1.0 vs v1.1 vs EL 标杆）。

🔴 口径铁律
- 杂音（静音段 4-8kHz 占比）：**统一降到 16k** 再算。
  16k 的 Nyquist=8k，4-8k 是最高频段；24k 源与 16k 源的占比天然不同，
  不统一就在比采样率，不是比噪声。
- click：必须**先归一到峰值 0.70** 再判 |Δ|>=0.30。阈值是绝对的，
  不归一等于在比谁声音大。
- 音色：campplus 16k，与 EL 标杆（同文案）比余弦，全篇多窗平均。
- cps：字数与时长必须取自**同一条 job 记录**，绝不手拼。
"""
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")

SR = 16000
EL_F = "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"   # EL 女声原声（同文案）
OUT = "/mnt/tts/output"


def to16k(path):
    """任何格式 -> 16k 单声道 wav（缓存到 /tmp）。

    🔴 缓存名必须用**全路径**派生，不能只用 basename：男女声的 EL 标杆
    都叫 `merged.wav`（cases/00b6a45ad6aa/ 与 cases/6cdd25bc75f2/），
    按 basename 缓存会让第二次测量静默复用第一次的文件
    —— 实测男声成品被拿去和**女声**标杆比，音色 0.2760（"完全不同人"）
    就是这么来的，差点当真。
    """
    import hashlib
    h = hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:10]
    dst = os.path.join("/tmp", "m48_%s_%s.wav"
                       % (h, os.path.basename(path).rsplit(".", 1)[0][:20]))
    if not os.path.exists(dst) or os.path.getsize(dst) < 1000:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                        "-ar", str(SR), "-ac", "1", dst], check=True)
    return dst


def hiss(path):
    """静音段 4-8kHz 能量占比 %（取最安静 20% 帧）。"""
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = int(sr * 0.02)
    m = (len(x) // n) * n
    fr = x[:m].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    seg = fr[rms <= np.percentile(rms, 20)].ravel()
    if len(seg) < sr:
        return float("nan")
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


_SESS = None
_SPK = None


def timbre(path, ref):
    """campplus 音色余弦（×EL 标杆）。同人阈值 0.75。"""
    global _SESS, _SPK
    try:
        if _SPK is None:
            sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")
            import spkchk
            import onnxruntime
            opt = onnxruntime.SessionOptions()
            opt.graph_optimization_level = \
                onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            opt.intra_op_num_threads = 2
            _SESS = onnxruntime.InferenceSession(
                spkchk.CAMPPLUS, sess_options=opt,
                providers=["CPUExecutionProvider"])
            _SPK = spkchk
        a = _SPK.embed(_SESS, _SPK.read16(path))
        b = _SPK.embed(_SESS, _SPK.read16(ref))
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    except Exception as e:
        print("   ⚠️ 音色测量失败: %s" % e)
        return float("nan")


def find(title):
    """产物目录可能被加 -N 后缀：挑 mp3 最多且最新的。"""
    cands = sorted(glob.glob(os.path.join(OUT, title + "*")))
    best, bk = None, None
    for d in cands:
        fs = glob.glob(os.path.join(d, "*.mp3"))
        if not fs:
            continue
        k = (len(fs), max(os.path.getmtime(f) for f in fs))
        if bk is None or k > bk:
            best, bk = d, k
    return os.path.join(best, sorted(glob.glob(os.path.join(best, "*.mp3")))[0]) if best else None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles", nargs="+", required=True,
                    help="output 目录下的 title（可多个）")
    ap.add_argument("--el", default=EL_F)
    ap.add_argument("--no-timbre", action="store_true")
    a = ap.parse_args()

    el = to16k(a.el)
    rows = []
    print("=== L3 全篇口径（16k 统一）===")
    print("%-18s %10s %10s %10s %10s" % ("素材", "时长s", "杂音4-8k", "click%", "音色"))
    for t in a.titles:
        p = find(t)
        if not p:
            print("%-18s  ❌ 找不到成品" % t)
            continue
        w = to16k(p)
        dur = sf.info(w).duration
        h, c = hiss(w), click(w)
        tb = float("nan")
        if not a.no_timbre:
            r = timbre(w, el)
            tb = r[0] if isinstance(r, (tuple, list)) else r
        rows.append((t, dur, h, c, tb))
        print("%-18s %10.1f %10.3f %10.3f %10s" % (
            t, dur, h, c, ("%.4f" % tb) if tb == tb else "n/a"))
        sys.stdout.flush()

    eh, ec = hiss(el), click(el)
    print("%-18s %10.1f %10.3f %10.3f %10s" % (
        "EL 标杆", sf.info(el).duration, eh, ec, "1.0000"))
    print()
    print("（杂音越低越接近 EL；click 越低越好；音色越高越接近 EL）")


if __name__ == "__main__":
    main()
