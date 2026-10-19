# -*- coding: utf-8 -*-
"""clickattr53.py —— click 归因定位（不猜，先看清它们长在哪）。

## 为什么写它
click 是我们**唯一**剩下比 EL 差的指标，但两条文案结论相反：
    文案1（d172b8c4efb3）1.30×EL
    文案2（8197d9ac8a35）0.78×EL（我们更好）
→ click 不是全局属性，有具体来源。全频段瞎试已经证伪两次（hfscan / lpfscan），
  必须先看清 click 的**形态和位置**再决定动不动、动哪里。

## 本工具输出（我们 vs EL 同口径对比）
1. click 落在发声段的哪个相位（段首 0~20% / 段中 / 段尾 80~100%）
2. click 处局部能量（5ms 帧 dB）—— 是响处的跳变还是弱处的跳变
3. click 是**孤立单样本**还是**连续簇**（连续 N 个样本都超限）
4. 跳变幅值分布（|Δ| 的 p50 / p90 / max）
5. click 密度随时间的分布（全篇均匀？还是集中在某几段）
6. 所在发声段的时长分布 —— 是长段里的还是碎段里的

用法：python3 clickattr53.py <ours.wav> <el.wav>
"""
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
VTH = -40.0
CLICK_TH = 0.30
PEAK = 0.70


def load16(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    return x.astype(np.float64)


def analyze(p):
    x = load16(p)
    dur = len(x) / SR
    pk = np.max(np.abs(x))
    xn = x * (PEAK / pk) if pk > 0 else x
    d = np.abs(np.diff(xn))
    idx = np.where(d >= CLICK_TH)[0]          # 跳变位置（样本）
    n = len(idx)

    # 分帧
    nf = int(FRAME * SR)
    nfr = len(x) // nf
    fr = x[:nfr * nf].reshape(nfr, nf)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > VTH

    # 发声段（帧索引）
    segs, i = [], 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1

    # 每个 click 的相位 / 局部 dB / 段时长
    phase, locdb, seglen = [], [], []
    for k in idx:
        f = k // nf
        # 找所属段（含前后 1 帧容差）
        hit = None
        for (a, b) in segs:
            if a - 1 <= f <= b:
                hit = (a, b)
                break
        if hit is None:
            phase.append(-1.0)      # 落在静音里
            seglen.append(0.0)
        else:
            a, b = hit
            ph = (f - a) / max(1, (b - a))
            phase.append(min(1.0, max(0.0, ph)))
            seglen.append((b - a) * FRAME)
        locdb.append(db[min(f, len(db) - 1)])

    phase = np.array(phase)
    locdb = np.array(locdb)
    seglen = np.array(seglen)

    # 连续簇：相邻 click 样本间隔 <=2
    if n:
        brk = np.where(np.diff(idx) > 2)[0]
        nclust = len(brk) + 1
        sizes = np.diff(np.concatenate(([-1], brk, [n - 1])))
    else:
        nclust, sizes = 0, np.array([])

    # 时间分布：每 30s 一格
    nbin = max(1, int(dur / 30))
    hist = np.zeros(nbin, dtype=int)
    for k in idx:
        b = min(nbin - 1, int(k / SR / 30))
        hist[b] += 1

    print("=== %s  %.1fs ===" % (p.split("/")[-1], dur))
    print("  click 样本 %d (%.4f%%)  簇 %d 个（簇=连续跳变）"
          % (n, n / len(x) * 100, nclust))
    if n == 0:
        print()
        return None
    print("  簇大小 中位=%.0f p90=%.0f 最大=%d"
          % (np.median(sizes), np.percentile(sizes, 90), sizes.max()))
    print("  |Δ|  p50=%.3f p90=%.3f max=%.3f"
          % (np.percentile(d[idx], 50), np.percentile(d[idx], 90),
             d[idx].max()))
    print("  相位  段首<0.2: %.1f%%  段中: %.1f%%  段尾>0.8: %.1f%%  "
          "静音中: %.1f%%"
          % ((phase < 0.2).mean() * 100,
             ((phase >= 0.2) & (phase <= 0.8)).mean() * 100,
             (phase > 0.8).mean() * 100,
             (phase < 0).mean() * 100))
    print("  局部dB p10=%.0f 中位=%.0f p90=%.0f"
          % (np.percentile(locdb, 10), np.median(locdb),
             np.percentile(locdb, 90)))
    print("  所在段时长 中位=%.0fms p90=%.0fms（0=静音）"
          % (np.median(seglen) * 1000, np.percentile(seglen, 90) * 1000))
    print("  每30s密度（个）: %s" % " ".join(str(v) for v in hist))
    print("    密度 中位=%.0f max=%d  max/中位=%.1f"
          % (np.median(hist), hist.max(),
             hist.max() / max(1, np.median(hist))))
    print()
    return dict(n=n, rate=n / len(x), nclust=nclust,
                clus_med=float(np.median(sizes)),
                d50=float(np.percentile(d[idx], 50)),
                head=float((phase < 0.2).mean()),
                tail=float((phase > 0.8).mean()),
                sil=float((phase < 0).mean()),
                db_med=float(np.median(locdb)),
                seglen_med=float(np.median(seglen)),
                dens_cv=float(hist.max() / max(1, np.median(hist))))


def main():
    if len(sys.argv) < 3:
        print("用法：clickattr53.py <ours.wav> <el.wav>")
        return 1
    a = analyze(sys.argv[1])
    b = analyze(sys.argv[2])
    if a and b:
        print("=" * 52)
        print("%-14s %10s %10s" % ("特征", "我们", "EL"))
        for k, lab in [("rate", "click率%"), ("nclust", "簇数"),
                       ("clus_med", "簇中位"), ("d50", "|Δ|p50"),
                       ("head", "段首占比"), ("tail", "段尾占比"),
                       ("sil", "静音占比"), ("db_med", "局部dB中位"),
                       ("seglen_med", "段时长中位s"),
                       ("dens_cv", "密度max/中位")]:
            print("%-14s %10.4f %10.4f" % (lab, a[k], b[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
