#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zerofix53b - declick 参数网格：找「对齐 EL 标杆」的填充电平

zerofix53 已证明：零段 52/44 -> 0（男女都清零）。但男声 p5 只从 -138 回到 -112，
EL 标杆是 -91 —— 差 21dB 意味着仍有大量样本在 mp3 里被压成准零（电磁音源）。

本脚本按「EL 标杆自己的 p5」做目标，网格搜索 (quiet_db, fill_db)：
  quiet_db = 只处理低于此电平的样本
  fill_db  = 顶到多响
验收三看：
  1. zero>=20ms 必须为 0
  2. p5 逼近 EL（男 -91 / 女 -76），**不得超过**（超过 = 比 EL 还吵）
  3. 语音帧（>-40dB）样本**零改动**，时长零变化
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

SR = 16000
APP = "/mnt/tts/projects/qwen3-tts/app"
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


def zero_runs(y, sr, min_ms=20.0):
    z = np.abs(y) < 1e-9
    d = np.diff(np.concatenate(([0], z.astype(np.int8), [0])))
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return [(b - a) / sr * 1000.0 for a, b in zip(s, e)
            if (b - a) / sr * 1000.0 >= min_ms]


def steep_drops(y, sr, ms=30.0, drop=25.0):
    win = int(ms / 1000.0 * sr)
    env = 10.0 * np.log10(np.convolve(y * y, np.ones(win) / win, mode="same") + 1e-20)
    return sum(1 for i in range(0, len(env) - 2 * win, win)
               if env[i:i + win].max() - env[i + win:i + 2 * win].max() >= drop)


def metrics(y, sr):
    db = 20 * np.log10(np.abs(y) + 1e-12)
    pause = db < -45
    q = np.percentile(db[pause], [5, 25, 50]) if pause.sum() > 10 else [np.nan] * 3
    return dict(zero=len(zero_runs(y, sr)), steep=steep_drops(y, sr),
                p5=q[0], p25=q[1], med=q[2], peak=db.max(), dur=len(y) / sr)


def main():
    cases = [
        ("MALE", "/mnt/tts/output/202610210738/202610210738-1.mp3",
         "/mnt/tts/data/cases/1a6c20a60481/merged.wav", -91.0),
        ("FEMALE", "/mnt/tts/output/202610210939/202610210939-1.mp3",
         "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav", -76.0),
    ]
    grid = [(-100.0, -89.0), (-100.0, -95.0), (-95.0, -95.0),
            (-91.0, -91.0), (-85.0, -85.0), (-80.0, -80.0)]
    for gname, mp3, elwav, target_p5 in cases:
        print("=" * 92)
        print("%s   EL target p5 = %.0f dB" % (gname, target_p5))
        base = decode(mp3)
        d0 = metrics(base, SR)
        elm = metrics(decode(elwav), SR)
        print("%-22s zero=%3d steep=%3d  p5=%6.0f p25=%6.0f med=%6.0f"
              % ("EL bench", elm["zero"], elm["steep"], elm["p5"], elm["p25"], elm["med"]))
        print("%-22s zero=%3d steep=%3d  p5=%6.0f p25=%6.0f med=%6.0f"
              % ("ours (as delivered)", d0["zero"], d0["steep"], d0["p5"], d0["p25"], d0["med"]))
        voiced = np.abs(base) > 10 ** (-40 / 20.0)
        for qdb, fdb in grid:
            y2 = declick.apply(base, SR, fill_db=fdb, enabled=True,
                               mode="floor", quiet_db=qdb)
            m = metrics(y2, SR)
            moved = int((np.abs(y2 - base) > 1e-12).sum())
            mv_v = int(((np.abs(y2 - base) > 1e-12) & voiced).sum())
            flag = ""
            if m["p5"] > target_p5 + 0.5:
                flag = "  <-- over EL (too loud)"
            elif abs(m["p5"] - target_p5) <= 3.0:
                flag = "  <== closest to EL"
            print("q=%6.0f f=%6.0f  zero=%3d steep=%3d  p5=%6.0f p25=%6.0f "
                  "med=%6.0f  dur=%.1f(%.3f)  moved=%.2f%%(voiced %.4f%%)%s"
                  % (qdb, fdb, m["zero"], m["steep"], m["p5"], m["p25"], m["med"],
                     m["dur"], m["dur"] - d0["dur"],
                     100.0 * moved / len(base), 100.0 * mv_v / max(int(voiced.sum()), 1),
                     flag))


if __name__ == "__main__":
    main()
