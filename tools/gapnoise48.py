# -*- coding: utf-8 -*-
"""gapnoise48.py —— 找「停顿里冒出来的孤立短音」。

## 背景（员工原话）
- 用户听 `xf5272-prod`（女声 omnivoice，300 字）：
  「说完 你 马上出来有个'嘎哈'的 0.2 秒音」
- z18 留言（디아）：「停顿时有 啊的声音」
- z15 留言（디아）：「第五个问题音频背景有异声」

三次反馈指向同一类现象：**本该安静的停顿里冒出一段短音**。
这跟记忆里已封存的「句间停顿往下沉」不是一回事 —— 那是**能量下沉**，
这次是**多出来一段音**。所以不能套用旧结论，要重新量。

## 判据
孤立短音 = 一段连续发声，满足：
  · 时长 0.08 ~ 0.60 s（"0.2 秒音"这种量级）
  · 前面静音 ≥ 0.15 s，后面静音 ≥ 0.15 s（**夹在停顿中间**）
发声阈值用「峰值帧 RMS 往下 -35dB」，不跟绝对音量挂钩。

🔴 必须与 EL 标杆同法对照：EL 也有 = 呼吸/自然现象，不是缺陷；
  只有我们有 = 真缺陷。
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FR = 0.02                 # 帧长 20ms
SIL_AROUND = 0.15         # 前后静音至少这么久才算"夹在停顿里"
MIN_LEN, MAX_LEN = 0.08, 0.60
DROP_DB = -35.0           # 发声阈值：相对峰值帧


def load(p):
    if not p.endswith(".wav"):
        t = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                        "-ar", str(SR), "-ac", "1", t], check=True)
        p = t
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def db(v):
    return 20 * np.log10(v + 1e-12)


def find(x, sr, name):
    n = int(sr * FR)
    m = (len(x) // n) * n
    fr = x[:m].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    thr = rms.max() * (10 ** (DROP_DB / 20.0))
    voiced = rms > thr

    # 连续 voiced 段
    segs = []
    i = 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j + 1 < len(voiced) and voiced[j + 1]:
                j += 1
            segs.append((i, j))
            i = j + 1
        else:
            i += 1

    out = []
    for k, (a, b) in enumerate(segs):
        dur = (b - a + 1) * FR
        if not (MIN_LEN <= dur <= MAX_LEN):
            continue
        pre = a * FR if k == 0 else (a - segs[k - 1][1] - 1) * FR
        nxt = ((len(voiced) - 1 - b) * FR if k == len(segs) - 1
               else (segs[k + 1][0] - b - 1) * FR)
        if pre >= SIL_AROUND and nxt >= SIL_AROUND:
            out.append({
                "t0": a * FR, "t1": (b + 1) * FR, "dur": dur,
                "pre_sil": pre, "post_sil": nxt,
                "db_rel": float(db(rms[a:b + 1].max()) - db(rms.max())),
            })
    return out, len(segs), float(m / sr)


def main():
    if len(sys.argv) < 2:
        print("用法: gapnoise48.py <音频1> [音频2 ...]")
        return 1
    for p in sys.argv[1:]:
        x, sr = load(p)
        hits, nseg, dur = find(x, sr, p)
        print("\n=== %s（%.1fs）===" % (os.path.basename(p), dur))
        print("  发声段总数 %d，其中「夹在停顿里的孤立短音」%d 处"
              % (nseg, len(hits)))
        for h in hits:
            print("    %.2fs ~ %.2fs  时长 %.2fs  "
                  "（前静音 %.2fs / 后静音 %.2fs）  电平 %+.1f dB"
                  % (h["t0"], h["t1"], h["dur"], h["pre_sil"],
                     h["post_sil"], h["db_rel"]))
        if not hits:
            print("    ✅ 无（停顿里没有冒出孤立短音）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
