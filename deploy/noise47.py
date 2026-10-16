# -*- coding: utf-8 -*-
"""第47轮：定位员工说的「波段杂音」。

## 背景
z45 上线后员工在 free 里连着 4 题写「波段杂音」，第 1 题还点名
「2 号（= GS 微调段）波段杂音」。这是**新问题**，z44 没人提过。

## 首要嫌疑：停顿处被 silfloor 填进去的底噪
生产日志（job 01e6cd91d140，GS 那条）：
    第 1 块 silfloor: 深谷 91→8 段, <−100dB 51→0 帧, 停顿 p5 **-91.6→-58.6 dB**
    第 2 块 silfloor: 深谷 92→8 段, <−100dB 41→0 帧, 停顿 p5 **-91.5→-59.2 dB**
→ GS 的停顿被**抬了 33dB**；现役那条只抬 18dB（−76.8→−58.2）。
填得越多越可能被听到。若底噪是「拼来的帧 / 合成粉噪」，
在停顿处就是一片持续的沙沙声 —— 正好对应「波段杂音」。

## 做法
1. 对 z45 的 8 段素材做长时平均谱（LTAS），找**窄带峰**（比局部中位高 >8dB）。
2. 单独拎出**停顿帧**（5ms 帧 RMS < −50dB）的平均谱 —— 杂音在停顿处最明显。
3. 与 **EL 原声未经任何处理的同段 wav** 对照（黄金基线）。
4. 顺带报高频截止（看是不是双重 mp3 编码把高频砍了）。
"""
import os
import subprocess
import sys

import numpy as np

SR = 24000
TMP = "/mnt/tts/diag/_noise47"
os.makedirs(TMP, exist_ok=True)

FB = "/mnt/tts/fb"
EL = "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"
CUR = "/mnt/tts/diag/sola47_f_omnivoice.wav"
GS = "/mnt/tts/diag/sola47_f_gptsovits.wav"

# (标签, 文件, 说明)
MP3S = [
    ("A_现役 s1", os.path.join(FB, "z45_a_s1.mp3")),
    ("B_GS   s1", os.path.join(FB, "z45_b_s1.mp3")),
    ("A_现役 s2", os.path.join(FB, "z45_a_s2.mp3")),
    ("B_GS   s2", os.path.join(FB, "z45_b_s2.mp3")),
    ("A_现役 s3", os.path.join(FB, "z45_a_s3.mp3")),
    ("B_GS   s3", os.path.join(FB, "z45_b_s3.mp3")),
    ("A_现役 el", os.path.join(FB, "z45_a_el.mp3")),
    ("B_EL   el", os.path.join(FB, "z45_b_el.mp3")),
]
# 黄金基线：直接从源 wav 切，**不经 mp3 编码**
RAW = [
    ("RAW_EL  s1", EL, 12.0),
    ("RAW_CUR s1", CUR, 12.0 * (458.83 / 459.13)),
    ("RAW_GS  s1", GS, 12.0 * (450.38 / 459.13)),
]
DUR = 20.0


def wav(src, t0=None, tag="x"):
    o = os.path.join(TMP, "%s.wav" % tag.replace(" ", "_"))
    c = ["ffmpeg", "-y", "-v", "error"]
    if t0 is not None:
        c += ["-ss", "%.2f" % t0, "-t", "%.2f" % DUR]
    c += ["-i", src, "-ac", "1", "-ar", str(SR)]
    c += ["-c:a", "pcm_s16le", o]
    subprocess.run(c, check=True)
    import soundfile as sf
    return sf.read(o, dtype="float32")[0]


def frames(y, ms=5.0):
    n = int(SR * ms / 1000.0)
    k = len(y) // n
    return y[:k * n].reshape(k, n)


def db(x):
    return 20 * np.log10(max(float(x), 1e-12))


def spec(y, nfft=2048):
    w = np.hanning(nfft)
    acc = np.zeros(nfft // 2 + 1)
    n = 0
    hop = nfft // 2
    for i in range(0, len(y) - nfft, hop):
        acc += np.abs(np.fft.rfft(y[i:i + nfft] * w)) ** 2
        n += 1
    return acc / max(n, 1)


def desc(name, y):
    F = frames(y)
    rms = np.sqrt((F ** 2).mean(axis=1))
    lvl = np.array([db(r) for r in rms])
    quiet = lvl < -50.0
    out = {}
    out["name"] = name
    out["lvl_med"] = float(np.median(lvl))
    out["quiet_pct"] = float(quiet.mean() * 100)
    # 停顿区电平（有杂音就会抬高）
    out["quiet_lvl_med"] = float(np.median(lvl[quiet])) if quiet.any() else float("nan")
    out["quiet_lvl_p10"] = float(np.percentile(lvl[quiet], 10)) if quiet.any() else float("nan")
    # 停顿区的频谱：找窄带峰
    if quiet.any():
        idx = np.where(quiet)[0]
        seg = y[idx[0] * len(F[0]): (idx[-1] + 1) * len(F[0])]
        if len(seg) > 4096:
            S = spec(seg)
        else:
            S = None
    else:
        S = None
    if S is not None:
        Sd = 10 * np.log10(S + 1e-20)
        # 局部中位（宽度 41 bin ≈ 470Hz）比，找突出峰
        k = 21
        med = np.array([np.median(Sd[max(0, i - k):i + k])
                        for i in range(len(Sd))])
        pk = Sd - med
        i = int(np.argmax(pk))
        out["peak_hz"] = float(i * SR / 2048.0)
        out["peak_db"] = float(pk[i])
        # 高频能量占比（>10kHz）
        hi = slice(int(10000 * 2048 / SR), len(Sd))
        out["hi_ratio"] = float(
            (S[hi].sum() / max(S.sum(), 1e-20)))
        # 极低频（<120Hz，哼声）
        lo = slice(0, int(120 * 2048 / SR))
        out["lo_ratio"] = float(S[lo].sum() / max(S.sum(), 1e-20))
    return out


def main():
    rows = []
    for nm, p in MP3S:
        rows.append(desc(nm, wav(p, None, nm)))
    for nm, p, t0 in RAW:
        rows.append(desc(nm, wav(p, t0, nm)))
    print("%-12s %8s %8s %10s %10s %9s %8s %8s %8s"
          % ("样本", "电平中位", "停顿%", "停顿中位", "停顿p10", "峰Hz", "峰dB", "低频占比", "高频占比"))
    for r in rows:
        print("%-12s %8.1f %8.1f %10.1f %10.1f %9.0f %8.1f %8.4f %8.4f"
              % (r["name"], r["lvl_med"], r["quiet_pct"],
                 r["quiet_lvl_med"], r["quiet_lvl_p10"],
                 r.get("peak_hz", float("nan")), r.get("peak_db", float("nan")),
                 r.get("lo_ratio", float("nan")), r.get("hi_ratio", float("nan"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
