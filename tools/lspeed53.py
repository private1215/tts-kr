# -*- coding: utf-8 -*-
"""lspeed53.py —— 局部语速代理：5s 窗的发声帧占比 + 段中位时长。

语速加快 → 停顿变少 → 发声帧占比升高、段中位时长变短。
用于验证员工「30秒后语速突然加快」是否真实（无对齐文本，用占比代理）。

用法：python3 lspeed53.py <wav> [起 止]
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
VTH = -40.0


def load16(p, sr=SR):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def main():
    if len(sys.argv) < 2:
        print("用法：lspeed53.py <wav> [起 止]")
        return 1
    p = sys.argv[1]
    t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    t1 = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
    x = load16(p)
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    dur = len(x) / SR
    t1 = min(t1, dur)
    print("=== %s  局部语速代理  %.0f~%.0fs ==="
          % (os.path.basename(p), t0, t1))
    print("  %7s %9s %9s %9s" % ("窗口s", "发声占比", "段数", "段中位ms"))
    s = t0
    while s < t1:
        a = int(s / FRAME)
        b = min(int((s + 5) / FRAME), len(db))
        if b <= a:
            break
        seg = db[a:b]
        v = seg > VTH
        ratio = float(v.mean())
        # 段中位时长
        segs, i = [], 0
        while i < len(v):
            if v[i]:
                j = i
                while j < len(v) and v[j]:
                    j += 1
                segs.append((j - i) * FRAME * 1000)
                i = j
            else:
                i += 1
        med = float(np.median(segs)) if segs else 0.0
        print("  %7.0f %8.1f%% %9d %9.0f"
              % (s, ratio * 100, len(segs), med))
        s += 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
