# -*- coding: utf-8 -*-
"""measure52.py —— 第52轮统一测量入口（一个口径走到底）。

## 为什么要有它

本项目最大的坑就是**口径不统一**：同一个改动，用不同脚本/不同采样率/
不同片段量出来的结论可以相反（第47轮栽了两次，第51轮我自己又栽了一次
片段 vs 全篇）。本脚本把「每次验证都要看的那几项」固定成**唯一实现**，
后面 M1/M2/M3/M4 每一步都只准用它，杜绝口径漂移。

## 固定口径（与《项目规范.md》第 3 节一致）

| 指标 | 口径 |
|---|---|
| click | ffmpeg **无条件降到 16k** → 峰值归一到 **0.70** → 判 \|Δ\|>=0.30 |
| 停顿 | 16k，20ms 帧，相对阈值 p35 切静音，MIN_PAUSE 0.12s，丢首尾 |
| 频谱 | 16k 全篇（不是片段！），质心 + 1k-3k / 3k-6k 占比 |
| LUFS | ffmpeg ebur128 整篇 |
| cps | **含空格**字符数 ÷ 时长；字数与时长必须同一条产物 |

## 用法

    python3 measure52.py a.wav b.wav ... [--out report.json] [--chars N]
    # --chars 只在这些 wav 字数相同时有意义；否则用 CHARS:name=123 逐个指定
    python3 measure52.py a.wav --chars-map 'a.wav=830,b.wav=830'
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
CLICK_THR = 0.30
PEAK = 0.70
FRAME = 0.02
MIN_PAUSE = 0.12


def load16(p, sr=SR):
    """无条件 ffmpeg 降到 16k 单声道。🔴 不能按扩展名跳过（见 clickpos48 注释）。"""
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32", always_2d=False)
    try:
        os.unlink(t)
    except OSError:
        pass
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def clicks(x, sr=SR):
    """按 clickpos48 同一算法：峰值归一到 0.70，判 |diff|>=0.30。
    本项目历史口径用 **密度百分比**（hits / 总样本 × 100），
    因此主输出为 pct；per_s 仅作辅助参考。"""
    y = x / (np.abs(x).max() + 1e-12) * PEAK
    hits = np.where(np.abs(np.diff(y)) >= CLICK_THR)[0]
    dur = len(y) / sr
    pct = len(hits) / len(y) * 100
    return {"n": int(len(hits)), "pct": round(pct, 4),
            "per_s": round(pct / 100 * sr, 3)}


def pauses(x, sr=SR):
    """返回停顿统计。相对阈值（db > p35 为发声），与绝对音量无关。"""
    n = int(FRAME * sr)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > np.percentile(db, 35)
    segs, i = [], 0
    while i < len(voiced):
        if not voiced[i]:
            j = i
            while j < len(voiced) and not voiced[j]:
                j += 1
            if i != 0 and j != len(voiced):
                segs.append((i, j))
            i = j
        else:
            i += 1
    lens = np.array([(b - a) * FRAME for a, b in segs], dtype=float)
    lens = lens[lens >= MIN_PAUSE]
    if len(lens) == 0:
        return {"n": 0}
    return {"n": int(len(lens)),
            "med": round(float(np.median(lens)), 3),
            "cv": round(float(np.std(lens) / np.mean(lens)), 3),
            "p90": round(float(np.percentile(lens, 90)), 3),
            "_lens": lens.tolist()}


def spectrum(x, sr=SR):
    """全篇平均频谱 → 质心 + 频段能量占比。"""
    n = 1024
    hop = 512
    nfr = 1 + (len(x) - n) // hop
    if nfr < 1:
        return {}
    idx = np.arange(nfr)[:, None] * hop + np.arange(n)[None, :]
    frames = x[idx] * np.hanning(n)[None, :]
    S = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    S = S.mean(axis=0)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    tot = S.sum() + 1e-12

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return round(float(S[m].sum() / tot * 100), 2)

    cent = float((f * S).sum() / tot)
    return {"centroid": int(cent), "b1k_3k": band(1000, 3000),
            "b3k_6k": band(3000, 6000), "b0_500": band(0, 500)}


def lufs(p):
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", p,
                            "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=600)
        tail = [l for l in r.stderr.splitlines() if "I:" in l]
        if tail:
            return round(float(tail[-1].split("I:")[1].split()[0]), 1)
        # 有些版本写 "Integrated loudness:  I:  -xx.x LUFS"
        for l in r.stderr.splitlines():
            if "Integrated loudness" in l:
                return round(float(l.split("I:")[-1].replace("LUFS", "").strip()), 1)
    except Exception:
        pass
    return None


def measure(path, chars=None):
    x = load16(path)
    sr = SR
    dur = len(x) / sr
    r = {"name": os.path.basename(path), "path": path,
         "dur_s": round(dur, 2)}
    r.update({"click_" + k: v for k, v in clicks(x, sr).items()})
    p = pauses(x, sr)
    r["pause_" + "n"] = p.get("n", 0)
    for k in ("med", "cv", "p90"):
        if k in p:
            r["pause_" + k] = p[k]
    r.update(spectrum(x, sr))
    r["lufs"] = lufs(path)
    if chars:
        r["cps"] = round(chars / dur, 2)     # 🔴 含空格字符数
        r["chars"] = int(chars)
    return r


def main():
    argv = sys.argv[1:]
    args, out, chars, cmap, i = [], None, None, {}, 0
    while i < len(argv):
        a = argv[i]
        if a == "--out" and i + 1 < len(argv):
            out = argv[i + 1]; i += 2; continue
        if a == "--chars" and i + 1 < len(argv):
            chars = int(argv[i + 1]); i += 2; continue
        if a.startswith("--chars-map="):
            for kv in a.split("=", 1)[1].split(","):
                k, v = kv.split(":"); cmap[k.strip()] = int(v)
            i += 1; continue
        if a.startswith("--"):
            i += 1; continue
        args.append(a); i += 1
    rows = []
    for p in args:
        print("测量", p, flush=True)
        c = cmap.get(os.path.basename(p))
        if c is None and chars is not None:
            c = chars
        rows.append(measure(p, c))
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(rows, open(out, "w"), ensure_ascii=False, indent=1)

    cols = [("name", "文件", 22, "s"), ("dur_s", "时长s", 7, ".1f"),
            ("click_pct", "click%", 8, ".4f"),
            ("click_per_s", "次/秒", 6, ".2f"),
            ("pause_n", "停顿", 6, "d"), ("pause_med", "中位s", 6, ".2f"),
            ("pause_cv", "CV", 5, ".2f"), ("centroid", "质心", 6, "d"),
            ("b1k_3k", "1k-3k%", 7, ".2f"), ("b3k_6k", "3k-6k%", 7, ".2f"),
            ("lufs", "LUFS", 6, ".1f"), ("cps", "cps", 6, ".2f")]
    hdr = "".join(["%-*s" % (w, h) for _, h, w, _ in cols])
    print("\n" + hdr)
    for r in rows:
        line = ""
        for k, _, w, f in cols:
            v = r.get(k)
            if v is None:
                line += "%-*s" % (w, "-")
            elif f == "s":
                line += "%-*s" % (w, str(v)[:w])
            elif f == "d":
                line += "%-*d" % (w, int(v))
            else:
                line += "%-*s" % (w, format(float(v), f))
        print(line)
    if out:
        print("\njson -> %s" % out)


if __name__ == "__main__":
    sys.exit(main())
