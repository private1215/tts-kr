#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l3check53 - L3 长篇验收复核（规范 S5：L3 是唯一质量验收）

对任意长篇跑同口径对照：我们的产物 vs EL 标杆。
判据（2026-10-21 v3.6 数字零填充）：
  zero>=20ms   必须 0（v3.5 是 52/44）
  steep30ms    向 EL 看齐（男 EL 28 / 女 EL 0）
  p5           向 EL 看齐（男 EL -91 / 女 EL -76）
  med          **不许变**（v3.6 只填深坑，不抬底噪：男 EL -59 / 女 EL -54）
  cps          男 ≈8.37 / 女 ≈7.57
用法：venv/bin/python l3check53.py <ours.mp3> <el.wav> <chars>
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

SR = 16000


def decode(path, sr=SR):
    out = tempfile.mktemp(suffix=".raw")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    "-ac", "1", "-ar", str(sr), "-c:a", "pcm_f32le",
                    "-f", "f32le", out], check=True)
    y = np.fromfile(out, dtype="<f4").astype(np.float32)
    os.remove(out)
    return y


def zero_runs(y, sr, min_ms=20.0):
    z = np.abs(y) < 1e-9
    d = np.diff(np.concatenate(([0], z.astype(np.int8), [0])))
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return [(a / sr, (b - a) / sr * 1000.0)
            for a, b in zip(s, e) if (b - a) / sr * 1000.0 >= min_ms]


def steep_drops(y, sr, ms=30.0, drop=25.0):
    w = int(ms / 1000.0 * sr)
    env = 10.0 * np.log10(np.convolve(y * y, np.ones(w) / w, mode="same") + 1e-20)
    return sum(1 for i in range(0, len(env) - 2 * w, w)
               if env[i:i + w].max() - env[i + w:i + 2 * w].max() >= drop)


def lufs(p):
    r = subprocess.run(["ffmpeg", "-nostats", "-i", p, "-af",
                        "ebur128=framelog=quiet", "-f", "null", "-"],
                       capture_output=True, text=True)
    for line in reversed(r.stderr.splitlines()):
        if "I:" in line and "LUFS" in line:
            try:
                return float(line.split("I:")[1].split("LUFS")[0].strip())
            except ValueError:
                continue
    return None


def stats(path, tag, chars=None):
    y = decode(path)
    dur = len(y) / SR
    db = 20 * np.log10(np.abs(y) + 1e-12)
    pause = db < -45
    q = np.percentile(db[pause], [5, 25, 50]) if pause.sum() > 10 else [np.nan] * 3
    zr = zero_runs(y, SR)
    sd = steep_drops(y, SR)
    cps = (chars / dur) if chars else float("nan")
    print("%-20s dur=%7.1fs  zero=%3d  steep=%3d  p5=%5.0f  p25=%5.0f  "
          "med=%5.0f  peak=%.1fdB  LUFS=%.1f%s"
          % (tag, dur, len(zr), sd, q[0], q[1], q[2], db.max(),
             lufs(path) or 0,
             ("  cps=%.2f" % cps) if chars else ""))
    return dict(dur=dur, zero=len(zr), steep=sd, p5=q[0], med=q[2], cps=cps)


def main():
    ours, el, chars = sys.argv[1], sys.argv[2], (
        int(sys.argv[3]) if len(sys.argv) > 3 else None)
    g = sys.argv[4] if len(sys.argv) > 4 else ""
    a = stats(ours, "OURS " + g, chars)
    b = stats(el, "EL bench " + g)
    print("  → 零段 %d vs EL %d ；陡降 %d vs EL %d ；p5 %.0f vs EL %.0f ；"
          "med %.0f vs EL %.0f"
          % (a["zero"], b["zero"], a["steep"], b["steep"],
             a["p5"], b["p5"], a["med"], b["med"]))
    ok = (a["zero"] == 0 and abs(a["med"] - b["med"]) <= 3.0)
    print("  L3 判定: %s" % ("PASS（零段归零且静音中位未偏离）" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
