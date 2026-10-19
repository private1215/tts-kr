# -*- coding: utf-8 -*-
"""tailclean53b.py —— 清理「句末粘连型鼻音颤簇」（tailclean53 的簇扩展）。

## 背景（z29 员工反馈）

z29（v4+tailclean）后，员工仍报男声有「嗯 / 砰砰 / 卡顿」：
    男 100s -> 99.99~100.70 一连串 5ms 鼻音爆点（zcr≈0.01~0.08，低频>0.6）

`tailclean53.py` 要求「后段间隔 >0.05s」，而这些爆点彼此间距只有 5~10ms，
所以被漏掉。

## 改动

不再以「单个发声段」为单位，而是把满足以下条件的相邻短段**合并成簇**：
    · 每个短段都满足原 tailclean 的时长/峰值/浊音/前间隔条件
    · 短段之间间距 < 0.03s（说明是同一个颤簇被 -40dB 门槛切碎了）
    · 整簇被 >0.05s 的静音包围（后面是真正停顿/下一句）
    · 整簇时长 5~250ms，峰值 -45~-20dB，整体浊音（zcr<0.15, low>0.45）

这样同时覆盖：
    · 原来的孤立句末尾巴（单一段，后面跟大静音）
    · 新发现的密集鼻音颤簇（多段紧邻，合并后才跟大静音）

## 用法
    python3 tailclean53b.py in.wav out.wav [--dry]
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
VTH = -40.0


def load16(p, sr=SR):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def fade(a, ms=5):
    n = int(SR * ms / 1000)
    if len(a) < 2 * n:
        return a
    w = np.linspace(0, 1, n)
    a = a.copy()
    a[:n] *= w
    a[-n:] *= w[::-1]
    return a


def voiced_quality(a, sr=SR):
    if len(a) < 16:
        return 1.0, 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(a)))) / 2)
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    fr = np.fft.rfftfreq(len(a), 1.0 / sr)
    low = float(sp[fr < 300].sum() / (sp.sum() + 1e-12))
    return zcr, low


def find_tail_clusters(x, cluster_gap=0.030, cluster_max=0.250):
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    v = db > VTH
    raw, i = [], 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            raw.append((i, j))
            i = j
        else:
            i += 1

    # 第一遍：找所有满足原 tailclean 条件但**不要求后间隔**的候选段，
    # 并记录单段自己的后间隔（用于保留原 tailclean 单段捕捉）。
    cands = []
    for k in range(1, len(raw) - 1):
        a, b = raw[k]
        d = (b - a) * FRAME
        if not (0.005 <= d <= 0.120):
            continue
        pk = float(db[a:b].max())
        if not (-45 <= pk <= -20):
            continue
        prev_dur = (raw[k - 1][1] - raw[k - 1][0]) * FRAME
        if prev_dur < 0.10:
            continue
        gap_pre = (a - raw[k - 1][1]) * FRAME
        if gap_pre > 0.030:
            continue
        gap_nxt = (raw[k + 1][0] - b) * FRAME
        seg = x[int(a * FRAME * SR):int(b * FRAME * SR)]
        zcr, low = voiced_quality(seg)
        if not (zcr < 0.15 and low > 0.45):
            continue
        cands.append({
            "k": k, "s": a * FRAME, "e": b * FRAME, "d": d, "pk": pk,
            "zcr": zcr, "low": low, "gap_pre": gap_pre, "gap_nxt": gap_nxt,
        })

    # 按间距 < cluster_gap 合并成簇
    groups = []
    cur = []
    for c in cands:
        if not cur:
            cur.append(c)
            continue
        gap = c["s"] - cur[-1]["e"]
        if gap < cluster_gap:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    if cur:
        groups.append(cur)

    # 对每簇：优先尝试「整簇捕捉」（2 段以上、整簇后间隔>50ms）;
    # 若整簇失败，再退回到「单段捕捉」（保留原 tailclean53 行为）。
    out = []
    for grp in groups:
        # 尝试整簇
        if len(grp) >= 2:
            first = grp[0]
            last = grp[-1]
            s = first["s"]
            e = last["e"]
            d = e - s
            if 0.005 <= d <= cluster_max:
                pk = float(db[int(s / FRAME):int(e / FRAME)].max())
                if -45 <= pk <= -20:
                    gap_pre = first["gap_pre"]
                    gap_nxt = (raw[last["k"] + 1][0] * FRAME) - last["e"]
                    if gap_nxt >= 0.050:
                        seg = x[int(s * SR):int(e * SR)]
                        zcr, low = voiced_quality(seg)
                        if zcr < 0.15 and low > 0.45:
                            out.append((s, d, pk, zcr, low,
                                        gap_pre, gap_nxt, len(grp)))
                            continue  # 整簇已处理，不再拆单段
        # 整簇失败或只有 1 段：逐段按原 tailclean53 判断
        for c in grp:
            if c["gap_nxt"] >= 0.050:
                out.append((c["s"], c["d"], c["pk"], c["zcr"], c["low"],
                            c["gap_pre"], c["gap_nxt"], 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--dry", action="store_true", help="只报告不写文件")
    a = ap.parse_args()

    x = load16(a.src)
    tails = find_tail_clusters(x)
    dur = len(x) / SR
    print("%s  %.1fs  句末粘连型嗯/颤簇 %d 个 (%.2f/分钟)"
          % (os.path.basename(a.src), dur, len(tails), len(tails) / dur * 60),
          flush=True)
    for s, d, pk, zcr, low, gp, gn, nseg in tails:
        print("   t=%7.2fs %3.0fms 峰值%.0fdB zcr=%.3f 低频=%.2f "
              "前间隔%.0fms 后间隔%.0fms 段数=%d"
              % (s, d * 1000, pk, zcr, low, gp * 1000, gn * 1000, nseg),
              flush=True)

    if a.dry or not tails:
        if not tails and not a.dry:
            sf.write(a.dst, x, SR)
            print("无需处理，已原样写出", flush=True)
        return 0

    y = x.copy()
    for s, d, *_ in tails:
        i0 = int(s * SR)
        i1 = int((s + d) * SR)
        n = i1 - i0
        seg = np.zeros(n, dtype=np.float32)
        y[i0:i1] = fade(seg)
    sf.write(a.dst, y, SR)
    print("-> %s  已替换 %d 处（等长静音，总时长不变）" % (a.dst, len(tails)),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
