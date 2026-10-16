#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""平坦度口径核查：mp3 vs wav 到底差多少（2026-10-15 第30轮）

## 为什么要做
上一轮 `flat_tune.py` 得出「我们平坦度 0.0222 / EL 0.0493，差 2.2 倍」，
但口径不一致：
  - 我们测的是**成品 mp3**（192k libmp3lame）
  - EL 测的是**原始 wav**
mp3 是心理声学有损编码，会砍掉高频细节 → **系统性压低平坦度**。
这个 2.2 倍很可能是**测量口径造出来的假差距**，必须排除。

## 做法
两组同口径对照：
  A. wav vs wav：`t_cur.wav`（我们 24k 成品 wav） vs `el24.wav`（EL 24k）
  B. mp3 vs mp3：两者都过一遍 libmp3lame 192k 再解码
再补一组：EL 的 wav 单独编成 mp3，看它自己掉多少。

## 用法
    ./venv/bin/python tools/flat_fair.py
"""
import json
import os
import subprocess

import numpy as np

SR16 = 16000
DIAG = "/mnt/tts/diag/muffle"
OUT = os.path.join(DIAG, "flat_fair.json")

CUR_WAV = os.path.join(DIAG, "t_cur.wav")
EL_WAV = os.path.join(DIAG, "el24.wav")
CUR_MP3 = "/mnt/tts/output/longprobe-r30fix-6cdd-2/longprobe-r30fix-6cdd-1.mp3"
EL_MP3 = os.path.join(DIAG, "el_mp3.mp3")   # 已存在（192k）

try:
    import librosa
except Exception:
    librosa = None


def load16(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ac", "1",
                          "-ar", str(SR16), "-f", "f32le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def flat_pg(y16):
    """完全复刻 probe_g：16k / 前 60 秒 / hop 256 / n_fft 2048。"""
    if librosa is None:
        return None
    return float(librosa.feature.spectral_flatness(
        y=y16[:60 * SR16], hop_length=256)[0].mean())


def to_mp3(src, dst, br="192k"):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src,
                    "-codec:a", "libmp3lame", "-b:a", br, dst], check=True)
    return dst


def main():
    res = {}
    rows = []

    def rec(tag, path):
        f = flat_pg(load16(path))
        res[tag] = f
        rows.append({"tag": tag, "path": path, "flat": f})
        print("%-22s %-10.5f  %s" % (tag, f, path))
        return f

    print("=== A. 同为 wav（公平） ===")
    f_cur_w = rec("我们 wav", CUR_WAV)
    f_el_w = rec("EL wav", EL_WAV)

    print("\n=== B. 同为 mp3（公平） ===")
    f_cur_m = rec("我们 mp3", CUR_MP3)
    f_el_m = rec("EL mp3(192k)", EL_MP3)

    print("\n=== 结论 ===")
    print("EL 自己 wav→mp3 掉幅: %.5f → %.5f  (%.2f 倍)"
          % (f_el_w, f_el_m, f_el_m / f_el_w))
    print("wav 口径差距:  我们 %.5f vs EL %.5f  → %.2f 倍"
          % (f_cur_w, f_el_w, f_el_w / f_cur_w))
    print("mp3 口径差距:  我们 %.5f vs EL %.5f  → %.2f 倍"
          % (f_cur_m, f_el_m, f_el_m / f_cur_m))

    json.dump({"rows": rows, "res": res,
               "wav_gap": f_el_w / f_cur_w,
               "mp3_gap": f_el_m / f_cur_m},
              open(OUT, "w"), indent=1)
    print("\n→ " + OUT)


if __name__ == "__main__":
    main()
