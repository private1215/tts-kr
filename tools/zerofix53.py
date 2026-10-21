#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zerofix53 - 验证 declick（数字零填充）能否消除员工反馈的「没音/嗯/额/咔」

背景（2026-10-21 员工反馈）：
  男 202610210738 / 女 202610210939 逐秒报「0.1秒没音」「嗯/额/啊」「电磁音」
  「碰碰声」。fback53d 实测：员工报的时间点 **逐个落在 |y|<1e-9 的数字零段**上
  （男 13 个点中 11 个命中，女 13 个点全命中）。EL 标杆同位置 0 处零段。

本脚本回答：把零段填成环境声之后，这些客观痕迹还在不在？填多响合适？
  - 零段 / 陡降 是否归零
  - 5% 分位是否回到 EL 标杆水平（男 -91dB / 女 -76dB）
  - 时长 / LUFS / cps 是否不动（declick 的三条不可违反约束）

用法（服务器）：
  /mnt/tts/projects/qwen3-tts/venv/bin/python zerofix53.py
"""
import os
import subprocess
import tempfile

import numpy as np

SR = 16000
OUT_SR = 44100
BITRATE = "192k"

APP = "/mnt/tts/projects/qwen3-tts/app"
import sys
sys.path.insert(0, APP)
import declick  # noqa: E402


def decode(path, sr=SR):
    out = tempfile.mktemp(suffix=".raw")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    "-ac", "1", "-ar", str(sr), "-c:a", "pcm_f32le",
                    "-f", "f32le", out], check=True)
    y = np.fromfile(out, dtype="<f4").astype(np.float32)
    os.remove(out)
    return y


def encode_mp3(y, sr, path):
    tmp = tempfile.mktemp(suffix=".wav")
    import soundfile as sf
    sf.write(tmp, y, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                    "-codec:a", "libmp3lame", "-b:a", BITRATE,
                    "-ar", str(OUT_SR), path], check=True)
    os.remove(tmp)


def zero_runs(y, sr, min_ms=20.0):
    z = np.abs(y) < 1e-9
    d = np.diff(np.concatenate(([0], z.astype(np.int8), [0])))
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return [(a / sr, (b - a) / sr * 1000.0)
            for a, b in zip(s, e) if (b - a) / sr * 1000.0 >= min_ms]


def steep_drops(y, sr, ms=30.0, drop=25.0):
    win = int(ms / 1000.0 * sr)
    env = 10.0 * np.log10(np.convolve(y * y, np.ones(win) / win, mode="same") + 1e-20)
    n = 0
    for i in range(0, len(env) - 2 * win, win):
        if env[i:i + win].max() - env[i + win:i + 2 * win].max() >= drop:
            n += 1
    return n


def lufs(p):
    r = subprocess.run(["ffmpeg", "-nostats", "-i", p, "-af", "ebur128=framelog=quiet",
                        "-f", "null", "-"], capture_output=True, text=True)
    v = None
    for line in reversed(r.stderr.splitlines()):
        if "I:" in line and "LUFS" in line:
            try:
                v = float(line.split("I:")[1].split("LUFS")[0].strip())
                break
            except ValueError:
                continue
    return v


def row(tag, y, sr, lufs_val=None):
    db = 20 * np.log10(np.abs(y) + 1e-12)
    pause = db < -45
    q = np.percentile(db[pause], [5, 50]) if pause.sum() > 10 else [float("nan")] * 2
    print("%-26s dur=%7.1fs  zero>=20ms=%3d  steep30ms=%3d  "
          "p5=%s  med=%s  peak=%.1fdB%s"
          % (tag, len(y) / sr, len(zero_runs(y, sr)), steep_drops(y, sr),
             ("%.0f" % q[0]) if q[0] == q[0] else "na",
             ("%.0f" % q[1]) if q[1] == q[1] else "na",
             db.max(),
             ("  LUFS=%.1f" % lufs_val) if lufs_val is not None else ""))
    return len(zero_runs(y, sr)), steep_drops(y, sr), q[0]


def main():
    cases = [
        ("MALE", "/mnt/tts/output/202610210738/202610210738-1.mp3",
         "/mnt/tts/data/cases/1a6c20a60481/merged.wav", "m", [-89.0, -85.0]),
        ("FEMALE", "/mnt/tts/output/202610210939/202610210939-1.mp3",
         "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav", "f", [-89.0, -83.0, -76.0]),
    ]
    for gname, mp3, elwav, g, fills in cases:
        print("=" * 84)
        print("%s   ours=%s" % (gname, os.path.basename(mp3)))
        base = decode(mp3)
        dur0 = len(base) / SR
        # EL 标杆（同口径解码）
        el = decode(elwav)
        row("EL bench (target)", el, SR)
        row("ours (as delivered)", base, SR)
        for fdb in fills:
            y2 = declick.apply(base, SR, fill_db=fdb, enabled=True,
                               mode="floor", quiet_db=declick.QUIET_DB)
            d = len(y2) / SR
            tag = "ours + declick %.0fdB" % fdb
            z0, s0, p50 = row(tag, y2, SR)
            if abs(d - dur0) > 1e-3:
                print("   !!! length changed %.4fs" % (d - dur0))
            # mp3 往返：确认编码后零段不会回来
            mp = tempfile.mktemp(suffix=".mp3")
            encode_mp3(y2, SR, mp)
            y3 = decode(mp)
            row("   ^ after mp3 roundtrip", y3, SR, lufs_val=lufs(mp))
            os.remove(mp)


if __name__ == "__main__":
    main()
