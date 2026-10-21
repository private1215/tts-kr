#!/usr/bin/env python3
"""fback53d - 三栏对照：参考模版 / 我们的产出 / EL标杆 的「数字零」取证

背景（2026-10-21 员工反馈 202610210738 男 / 202610210939 女）：
  员工逐秒报「0.1秒没音 / 嗯 / 额 / 啊 / 电磁音全程很多 / 碰碰声全程个别地方有」。
  本脚本回答三件事：
    1. 模版（参考音频）本身的停顿底噪是多少 —— 目标值从这量出来，不是拍脑袋；
    2. 我们的产出是不是数字真零（|y|<1e-9，即 -180dB 以下）；
    3. EL 标杆同样位置有没有这个现象。
判据：
  - >=20ms 连续全零段  -> 停顿被写成数字真零，mp3 解码后是「瞬间静断」
  - 30ms 内陡降 >=25dB -> 硬切，听成「碰/咔」
  - 停顿段底噪中位     -> 参考音频是真实房间底噪，模型学不到 -> 输出成零
用法（在服务器上跑）：
  /mnt/tts/projects/qwen3-tts/venv/bin/python fback53d.py
"""
import os
import subprocess
import tempfile

import numpy as np

SR = 16000


def load(path, sr=SR):
    """any audio -> mono float32 ndarray (resampled via ffmpeg)"""
    out = tempfile.mktemp(suffix=".raw")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
         "-ac", "1", "-ar", str(sr), "-c:a", "pcm_f32le", "-f", "f32le", out],
        check=True)
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
    win = int(ms / 1000.0 * sr)
    env = 10.0 * np.log10(np.convolve(y * y, np.ones(win) / win, mode="same") + 1e-20)
    n = 0
    hits = []
    for i in range(0, len(env) - 2 * win, win):
        a = env[i:i + win].max()
        b = env[i + win:i + 2 * win].max()
        if a - b >= drop:
            n += 1
            hits.append(round((i + win) / sr, 2))
    return n, hits


def stats(path, tag, sr=SR, show=8):
    y = load(path, sr)
    db = 20 * np.log10(np.abs(y) + 1e-12)
    pause = db < -45
    if pause.sum() > 10:
        q = np.percentile(db[pause], [5, 50, 95])
    else:
        q = [float("nan")] * 3
    zr = zero_runs(y, sr)
    nd, hits = steep_drops(y, sr)
    print("%-22s dur=%7.1fs  pause=%6.1fs  floor(5%%/med)=%s/%s dB  "
          "zero>=20ms=%3d  steep30ms=%2d"
          % (tag, len(y) / sr, pause.sum() / sr,
             ("%.0f" % q[0]) if q[0] == q[0] else "na",
             ("%.0f" % q[1]) if q[1] == q[1] else "na",
             len(zr), nd))
    if zr:
        print("     zero@  " + ", ".join("%.1fs/%.0fms" % (t, m)
                                         for t, m in zr[:show])
              + (" ..." if len(zr) > show else ""))
    if hits:
        print("     drop@  " + ", ".join("%.1fs" % t for t in hits[:show])
              + (" ..." if len(hits) > show else ""))
    return dict(dur=len(y) / sr, pause=pause.sum() / sr, floor5=q[0],
                floormed=q[1], zero=len(zr), steep=nd)


def main():
    groups = [
        ("MALE", [
            ("/mnt/tts/data/voices/EL_male.wav", "ref EL_male"),
            ("/mnt/tts/output/202610210738/202610210738-1.mp3", "ours 202610210738"),
            ("/mnt/tts/data/cases/1a6c20a60481/merged.wav", "EL bench male"),
        ]),
        ("FEMALE", [
            ("/mnt/tts/data/voices/EL_female.wav", "ref EL_female"),
            ("/mnt/tts/output/202610210939/202610210939-1.mp3", "ours 202610210939"),
            ("/mnt/tts/data/cases/00b6a45ad6aa/merged.wav", "EL bench female"),
        ]),
    ]
    # 员工报的时间点，用来核对是否落在零段/陡降上
    report = {
        "202610210738": [0.0, 5.6, 8.0, 14.0, 27, 42, 57, 71, 84, 100, 113, 126, 139],
        "202610210939": [0.0, 12.0, 27, 43, 55, 68, 81, 95, 110, 122, 137, 152, 164],
    }
    for gname, items in groups:
        print("=" * 78)
        print(gname)
        res = {}
        for p, tag in items:
            if not os.path.exists(p):
                print("%-22s MISSING %s" % (tag, p))
                continue
            res[tag] = stats(p, tag)
        # 员工报点 vs 实测零段
        for p, tag in items:
            key = tag.split()[-1]
            if key not in report or not os.path.exists(p):
                continue
            y = load(p)
            zr = zero_runs(y, SR, min_ms=5.0)
            zt = [t for t, m in zr]
            print("  -- reported timestamps vs zero-runs (>=5ms) --")
            for t in report[key]:
                near = [z for z in zt if abs(z - t) <= 1.5]
                print("     %6.1fs -> %s" % (
                    t, ("zero at " + ", ".join("%.2f" % z for z in near))
                    if near else "no zero within 1.5s"))


if __name__ == "__main__":
    main()
