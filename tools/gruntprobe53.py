# -*- coding: utf-8 -*-
"""gruntprobe53.py —— 按员工标注的位置探测「嗯」的真实声学特征。

## 背景（2026-10-17 z28 员工反馈）

员工给了**位置范围**（不是精确秒数），这是三轮以来第一次拿到准确定位：

    第一题 A（男 v3，片段起点 80s）  ：50s、1分03s      -> 绝对 130、143
    第二题 A（男 v4，片段起点 80s）  ：5s、21s、49s、1分03s -> 85、101、129、143
    第三题 A（女 v3，片段起点 340s） ：10s、23s、34s、47/50s、1分、1分08s
                                     -> 350、363、374、387/390、400、408

员工还给了**长度**：0.05s / 0.001s / 0.01s —— 比我设的 min-ms=20 下限还短。

对比我检测到的（v3/v4 判据）：
    男：85.09 / 133.55 / 134.12 ...  →  **101、143 完全没检测到**
    女：351.07 / 387.42 / 389.79     →  **363、374、400、408 没检测到**

→ 员工听到的「嗯」**更多、更短**，我的判据漏了大半。

## 本工具做什么

对每个标注位置，取 ±WIN 秒窗口，逐 5ms 帧输出 dB 包络，并列出窗口内
所有发声段（含**低于 p35 门槛的**低能量段）的
    起止 / 时长 / 峰值dB / zcr / 低频占比 / 与前后段的间隔
用来重新校准判据。

用法：
    python3 gruntprobe53.py <wav> <pos1> <pos2> ... [--win 1.5]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005


def load16(p, sr=SR):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def voiced_quality(a, sr=SR):
    if len(a) < 16:
        return 1.0, 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(a)))) / 2)
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    fr = np.fft.rfftfreq(len(a), 1.0 / sr)
    low = float(sp[fr < 300].sum() / (sp.sum() + 1e-12))
    return zcr, low


def frames(x):
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    return 20 * np.log10(rms + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("pos", nargs="+", type=float)
    ap.add_argument("--win", type=float, default=1.5)
    ap.add_argument("--floor", type=float, default=-45.0,
                    help="低于这个 dB 视为静音（比 p35 更敏感，"
                         "用来抓员工说的 10~50ms 极短嗯）")
    a = ap.parse_args()

    x = load16(a.wav)
    db = frames(x)
    n = int(FRAME * SR)
    print("=== %s  静音门槛 %.0fdB ===" % (os.path.basename(a.wav), a.floor))
    for pos in a.pos:
        i0 = max(0, int((pos - a.win) / FRAME))
        i1 = min(len(db), int((pos + a.win) / FRAME))
        seg_db = db[i0:i1]
        v = seg_db > a.floor
        segs, i = [], 0
        while i < len(v):
            if v[i]:
                j = i
                while j < len(v) and v[j]:
                    j += 1
                segs.append((i0 + i, i0 + j))
                i = j
            else:
                i += 1
        print("\n--- 标注 %.0fs（窗口 %.0f~%.0fs）共 %d 个发声段 ---"
              % (pos, pos - a.win, pos + a.win, len(segs)))
        print("   %8s %7s %8s %7s %7s %9s %9s"
              % ("起始s", "时长ms", "峰值dB", "zcr", "低频", "前间隔ms", "后间隔ms"))
        for k, (s, e) in enumerate(segs):
            st = s * FRAME
            du = (e - s) * FRAME
            pk = float(db[s:e].max())
            seg = x[int(st * SR):int((st + du) * SR)]
            zcr, low = voiced_quality(seg)
            pre = (s - segs[k - 1][1]) * FRAME * 1000 if k > 0 else -1
            nxt = (segs[k + 1][0] - e) * FRAME * 1000 if k + 1 < len(segs) else -1
            mark = ""
            # 突出：极短（<250ms）且是浊音的
            if du * 1000 <= 250 and zcr < 0.15 and low > 0.35:
                mark = "  <<< 疑似嗯"
            print("   %8.2f %7.0f %8.1f %7.3f %7.2f %9.0f %9.0f%s"
                  % (st, du * 1000, pk, zcr, low, pre, nxt, mark))
    return 0


if __name__ == "__main__":
    sys.exit(main())
