#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fback53.py —— 员工反馈取证：202610210738(男) / 202610210939(女)

员工报的点：开头异声 / 5.6s 电磁音 / 8s 0.1s 没音 / 14s「额」/
之后每隔 13~15s 一个「嗯」/ 全程电磁音 + 偶发「碰碰」声。

本脚本只做**客观取证**（不猜）：
1. 孤立浊音事件：在停顿里出现的短浊音块（候选「嗯/啊/额」），列时间与时长
2. 断音：语音中间的近零能量洞（候选「0.1s 没音」）
3. 窄带：长时平均谱里的窄峰（候选「电磁音」= 单频啸叫/嗡声）
4. 瞬态：短时能量突跳（候选「碰碰」声 = click/pop）
"""
import subprocess
import sys

import numpy as np
import soundfile as sf

SR = 16000
T_END = 200.0          # 员工只听到 ~2:44，取 200s 足够覆盖并留余量

FILES = [
    ("男 202610210738", "/mnt/tts/output/202610210738/202610210738-1.mp3"),
    ("女 202610210939", "/mnt/tts/output/202610210939/202610210939-1.mp3"),
]


def load16k(mp3, t_end=T_END):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp3,
                    "-ar", str(SR), "-ac", "1", "-t", "%.1f" % t_end,
                    "/tmp/fb53.wav"], check=True)
    y, sr = sf.read("/tmp/fb53.wav", dtype="float32")
    return y, sr


def frames(y, hop_ms=10.0, win_ms=20.0):
    hop = int(hop_ms / 1000 * SR)
    win = int(win_ms / 1000 * SR)
    n = max(1, (len(y) - win) // hop)
    rms = np.array([np.sqrt(np.mean(y[i * hop:i * hop + win] ** 2) + 1e-12)
                    for i in range(n)])
    return 20 * np.log10(rms + 1e-12), hop / SR


def voiced_events(db, dt, thr=-38.0):
    """返回 [(起点s, 终点s, 时长s)]，只保留 0.05~0.6s 的短浊音块。"""
    v = db > thr
    ev, i = [], 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            dur = (j - i) * dt
            if 0.04 <= dur <= 0.60:
                ev.append((i * dt, j * dt, dur))
            i = j
        else:
            i += 1
    return ev


def isolated(ev, db, dt, pad=0.35):
    """孤立事件：前后 pad 秒内几乎没有声音（在停顿里冒出来的一个音）。"""
    out = []
    for a, b, d in ev:
        i0 = max(0, int((a - pad) / dt))
        i1 = max(0, int(a / dt))
        j0 = min(len(db), int(b / dt))
        j1 = min(len(db), int((b + pad) / dt))
        pre = db[i0:i1].max() if i1 > i0 else -99
        post = db[j0:j1].max() if j1 > j0 else -99
        if pre < -40 and post < -40:
            out.append((a, b, d, db[int(a / dt):int(b / dt)].max()))
    return out


def dropouts(db, dt, thr=-55.0, min_d=0.05, max_d=0.5):
    """语音中间的近零洞：前后都有明显语音，中间掉到 thr 以下。"""
    v = db < thr
    out, i = [], 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            d = (j - i) * dt
            i0 = max(0, i - int(0.3 / dt))
            j1 = min(len(db), j + int(0.3 / dt))
            pre = db[i0:i].max() if i > i0 else -99.0
            post = db[j:j1].max() if j1 > j else -99.0
            if (min_d <= d <= max_d and pre > -40 and post > -40):
                out.append((i * dt, j * dt, d))
            i = j
        else:
            i += 1
    return out


def narrowband(y):
    """长时平均谱 + 窄峰检测（单频啸叫/嗡声）。"""
    n = 4096
    hopn = n // 2
    P = []
    for i in range(0, len(y) - n, hopn):
        seg = y[i:i + n] * np.hanning(n)
        P.append(np.abs(np.fft.rfft(seg)) ** 2)
    P = np.array(P)
    ltas = 10 * np.log10(P.mean(0) + 1e-12)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    # 窄峰 = 比左右 5 个 bin 的中位数高出 X dB
    peaks = []
    for k in range(6, len(ltas) - 6):
        loc = ltas[k]
        nb = np.median(np.concatenate([ltas[k - 6:k - 1], ltas[k + 2:k + 7]]))
        if loc - nb > 6.0 and loc > max(ltas) - 45:
            peaks.append((freqs[k], round(loc - nb, 1), round(loc, 1)))
    # 合并相邻 bin
    merged = []
    for f, d, lv in peaks:
        if merged and f - merged[-1][0] < 40:
            if d > merged[-1][1]:
                merged[-1] = (f, d, lv)
        else:
            merged.append([f, d, lv])
    return ltas, freqs, merged


def pops(y, db, dt, jump=12.0):
    """瞬态突跳：相邻帧能量跳变 > jump dB 且绝对值不太小（pop/click）。"""
    out = []
    for i in range(1, len(db)):
        if db[i] - db[i - 1] > jump and db[i] > -35:
            out.append((i * dt, round(db[i] - db[i - 1], 1), round(db[i], 1)))
    return out


def main():
    for tag, path in FILES:
        print("=" * 72)
        print("%s" % tag)
        y, sr = load16k(path)
        print("  前 %.0fs  RMS=%.1f dB  峰值=%.1f dB"
              % (T_END, 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12),
                 20 * np.log10(np.max(np.abs(y)) + 1e-12)))
        db, dt = frames(y)
        pc = np.percentile(db, [1, 5, 10, 25, 50, 75, 90, 99])
        print("  帧 dB 分位 1/5/10/25/50/75/90/99：%s"
              % " ".join("%.0f" % p for p in pc))
        ev = voiced_events(db, dt)
        iso = isolated(ev, db, dt)
        print("\n  ① 停顿里的孤立浊音块（候选「嗯/啊/额」）共 %d 个：" % len(iso))
        prev = None
        for a, b, d, lv in iso:
            gap = "" if prev is None else "  (间隔 %.1fs)" % (a - prev)
            print("     %6.2fs  +%.2fs  %5.1f dB%s" % (a, d, lv, gap))
            prev = a
        if iso:
            ivs = np.diff([a for a, _, _, _ in iso])
            print("     → 间隔：中位 %.1fs  均值 %.1fs  最小 %.1f  最大 %.1f"
                  % (np.median(ivs), ivs.mean(), ivs.min(), ivs.max()))
        dr = dropouts(db, dt)
        print("\n  ② 语音中间的断音洞（候选「0.1s 没音」）共 %d 个：" % len(dr))
        for a, b, d in dr[:12]:
            print("     %6.2fs  %.3fs" % (a, d))
        ltas, freqs, peaks = narrowband(y)
        print("\n  ③ 窄带峰（候选「电磁音」）：")
        if peaks:
            for f, d, lv in peaks:
                print("     %7.1f Hz  高出邻域 %.1f dB  绝对 %.1f dB" % (f, d, lv))
        else:
            print("     无（无单频啸叫/嗡声）")
        hi = ltas[freqs > 7000].max()
        mid = ltas[(freqs > 1000) & (freqs < 4000)].max()
        print("     高频(>7k) 最高 %.1f dB / 中频(1-4k) 最高 %.1f dB" % (hi, mid))
        pp = pops(y, db, dt)
        print("\n  ④ 瞬态突跳（候选「碰碰」声）共 %d 个：" % len(pp))
        for t, j, lv in pp[:12]:
            print("     %6.2fs  跳 +%.1f dB  到 %.1f dB" % (t, j, lv))
        print()


if __name__ == "__main__":
    sys.exit(main())
