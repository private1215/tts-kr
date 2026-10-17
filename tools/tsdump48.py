# -*- coding: utf-8 -*-
"""tsdump48.py —— 导出「我们 vs EL」的发声段/静音段时长序列（JSON）。

服务器 venv 没有 matplotlib，图只能在本地画，所以这里只负责算 + 导出。

用法：python3 tsdump48.py --ours <音频...> --el <音频...> [--out json]
      不给 --out 就打到 stdout。
"""
import argparse
import json
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005


def load(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    return x


def struct(p):
    x = load(p)
    n = int(SR * FRAME)
    m = len(x) // n * n
    db = 20 * np.log10(np.maximum(
        np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1)), 1e-12))
    v = db > db.mean() + 2.0
    mg = int(0.05 / FRAME)
    vv = v.copy()
    i = 0
    while i < len(vv):
        if not vv[i]:
            j = i
            while j + 1 < len(vv) and not vv[j + 1]:
                j += 1
            if (j - i + 1) <= mg:
                vv[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    segs, gaps, i = [], [], 0
    while i < len(vv):
        if vv[i]:
            j = i
            while j + 1 < len(vv) and vv[j + 1]:
                j += 1
            segs.append((j - i + 1) * FRAME)
            i = j + 1
        else:
            j = i
            while j + 1 < len(vv) and not vv[j + 1]:
                j += 1
            gaps.append((j - i + 1) * FRAME)
            i = j + 1
    return segs, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", nargs="+", required=True)
    ap.add_argument("--el", nargs="+", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {}
    for tag, fs in (("ours", a.ours), ("el", a.el)):
        s, g = [], []
        for f in fs:
            x, y = struct(f)
            s += x
            g += y
        s = np.array(s)
        g = np.array(g)
        res[tag] = {"seg": [round(float(v), 4) for v in s],
                    "gap": [round(float(v), 4) for v in g]}
        print("%s: seg n=%d med=%.3f | gap n=%d long>=0.4s %.1f%%"
              % (tag, len(s), float(np.median(s)), len(g),
                 float((g >= 0.4).mean() * 100)), file=sys.stderr)
    txt = json.dumps(res)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
        print("saved", a.out, file=sys.stderr)
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
