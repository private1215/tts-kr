# -*- coding: utf-8 -*-
"""pausecompress52.py —— 把过长的句间停顿压缩到目标长度（发声段不动）。

## 为什么需要它

第51轮结论：「停顿节奏是最大的机器感来源」。实测（全篇口径）：

| 链路 | 停顿中位 | 相对 EL |
|---|---|---|
| EL 男 | 0.16s | 1.00 |
| voxcpm 男 | 0.18~0.20s | 1.12~1.25 |
| omnivoice 女（现役，1.5 分） | 0.22s | 1.22 |
| **omnivoice 男** | **0.40s** | **2.50** |
| GS 男（0 分） | 0.34s | 2.13 |

omnivoice 男声音色 0.9477（voxcpm 最好 0.90）、cps 8.24（EL 8.27）、
质心 827（EL 818）全面占优，唯独停顿是 EL 的 2.5 倍。
→ 若只换引擎不修停顿，员工大概率仍判「机器」；故做「换引擎 + 修停顿」组合版。

## 做法

1. 相对阈值（db > p35 判发声，与绝对音量无关）切出停顿段
2. **按比例缩放**每个长停顿：新长度 = 原长 × scale（下限 --floor）
   —— 🔴 不能统一压到同一目标值：实测压到固定 0.25s 会让停顿 CV 从 0.55
   掉到 0.36，反而变成 GS 那种「等长节拍」，机器感更强。
   等比例缩放保持分布形态（CV 不变），只是整体缩短。
3. 用 ffmpeg `atempo` 做压缩（只揉静音，不动发声）
4. 5ms 淡入淡出，避免压缩边界出 click
5. 发声段原样保留，总时长随之缩短

🔴 这是**时长重分配**，不是「往停顿里加码」——发声段一个样本都没碰。

用法：
    python3 pausecompress52.py in.wav out.wav [--scale 0.6] [--floor 0.20]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.02


def load16(p, sr=SR):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return x


def pauses(x, sr=SR, min_p=0.25):
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
            if i != 0 and j != len(voiced) and (j - i) * FRAME >= min_p:
                segs.append((i * FRAME, j * FRAME))
            i = j
        else:
            i += 1
    return segs


def atempo(src_path, ratio, dst_path):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src_path,
                    "-af", "atempo=%.6f" % ratio, dst_path], check=True)


def fade(a, sr=SR, ms=5):
    n = int(sr * ms / 1000)
    if len(a) < 2 * n:
        return a
    w = np.linspace(0, 1, n)
    a = a.copy()
    a[:n] *= w
    a[-n:] *= w[::-1]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--scale", type=float, default=0.6,
                    help="停顿长度缩放系数（等比例，保持 CV）")
    ap.add_argument("--floor", type=float, default=0.20,
                    help="缩放后的下限(s)，避免压成瞬切")
    ap.add_argument("--min", type=float, default=0.25,
                    help="只处理超过这个长度的停顿(s)")
    a = ap.parse_args()

    x = load16(a.src)
    segs = pauses(x, min_p=a.min)
    # 新长度 = 原长 × scale，但不低于 floor、不超过原长
    plan = []
    for s, e in segs:
        d = e - s
        nd = min(d, max(d * a.scale, a.floor))
        if nd < d - 1e-6:
            plan.append((s, e, nd))
    if not plan:
        sf.write(a.dst, x, SR)
        print("无需压缩（无超过 %.2fs 的停顿）" % a.min)
        return 0

    tmpd = tempfile.mkdtemp()
    out, prev, n_done, saved = [], 0.0, 0, 0.0
    for s, e, nd in plan:
        out.append(x[int(prev * SR):int(s * SR)])
        seg = x[int(s * SR):int(e * SR)]
        ratio = (e - s) / nd
        t1 = os.path.join(tmpd, "p.wav")
        t2 = os.path.join(tmpd, "o.wav")
        sf.write(t1, fade(seg), SR)
        atempo(t1, ratio, t2)
        c, _ = sf.read(t2, dtype="float32")
        out.append(fade(c))
        saved += (e - s) - len(c) / SR
        prev = e
        n_done += 1
    out.append(x[int(prev * SR):])
    y = np.concatenate(out)
    sf.write(a.dst, y, SR)
    print("等比例压缩 %d 处停顿（scale=%.2f floor=%.2f）；节省 %.1fs"
          "（%.1fs -> %.1fs）" % (n_done, a.scale, a.floor, saved,
                                len(x) / SR, len(y) / SR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
