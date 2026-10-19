# -*- coding: utf-8 -*-
"""fixchk53.py —— 验证 clickfix53 的修复**没有削掉真实瞬态**。

担心什么：把超阈值样本拉平，可能把真正的辅音爆破（ㅂ/ㅍ/ㅋ/ㅌ 的起音）
一起削掉 → 声音发糊。频谱中位看不出来，要看**动态范围的尾部**。

查什么：
1. 峰值 / p99 / p95 dB 有没有被削（真实瞬态被削 → 峰值下降）
2. 每 5ms 帧的 dB 相关性（整体包络是否被改动）
3. 被改样本的空间分布：是**孤立散点**还是**连成片**（连成片=削平了一段）
4. 改动量 |Δx| 的分布（改动是否温和）
5. 发声段的**起音斜率**（段首前 10ms 的上升速率）——爆破音在这里

用法：python3 fixchk53.py <原产.wav> <修复后.wav>
"""
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000


def load16(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    return x.astype(np.float64)


def main():
    if len(sys.argv) < 3:
        print("用法：fixchk53.py <原产.wav> <修复后.wav>")
        return 1
    a = load16(sys.argv[1])
    b = load16(sys.argv[2])
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    print("1) 动态范围（被削掉真实瞬态 → 峰值下降）")
    for lab, x in [("原产", a), ("修复", b)]:
        pk = np.max(np.abs(x))
        s = np.sort(np.abs(x))[::-1]
        print("   %s  峰值 %.4f  p99.9 %.4f  p99 %.4f  p95 %.4f"
              % (lab, pk, s[int(n * 0.001)], s[int(n * 0.01)],
                 s[int(n * 0.05)]))

    print("\n2) 5ms 帧 dB 包络相关性（整体响度结构是否改动）")
    nf = int(0.005 * SR)
    nfr = n // nf
    ra = np.sqrt((a[:nfr * nf].reshape(nfr, nf) ** 2).mean(axis=1))
    rb = np.sqrt((b[:nfr * nf].reshape(nfr, nf) ** 2).mean(axis=1))
    da = 20 * np.log10(ra + 1e-12)
    dbb = 20 * np.log10(rb + 1e-12)
    print("   Pearson r = %.6f   帧 dB 差 中位 %.4f  p99 %.3f  max %.3f"
          % (np.corrcoef(da, dbb)[0, 1], np.median(dbb - da),
             np.percentile(np.abs(dbb - da), 99),
             np.max(np.abs(dbb - da))))

    print("\n3) 被改样本的空间分布（连成片=削平）")
    dif = np.abs(b - a)
    idx = np.where(dif > 1e-9)[0]
    print("   改动样本 %d 个（占 %.4f%%）" % (len(idx), len(idx) / n * 100))
    if len(idx):
        gaps = np.diff(idx)
        print("   相邻间隔 中位=%.0f  p5=%d  （1=连续成片）"
              % (np.median(gaps), np.percentile(gaps, 5)))
        # 连续块
        brk = np.where(gaps > 1)[0]
        sizes = np.diff(np.concatenate(([-1], brk, [len(idx) - 1])))
        print("   连续块数 %d   块大小 中位=%.0f p90=%.0f 最大=%d"
              % (len(sizes), np.median(sizes), np.percentile(sizes, 90),
                 sizes.max()))
        print("   改动量 |Δx| 中位=%.5f  p90=%.5f  max=%.5f"
              "  （相对峰值 %.2f%%）"
              % (np.median(dif[idx]), np.percentile(dif[idx], 90),
                 dif[idx].max(), dif[idx].max() / np.max(np.abs(a)) * 100))

    print("\n4) 起音斜率（爆破音是否被削）")
    v = da > -40
    segs, i = [], 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            if (j - i) >= 20:
                segs.append((i, j))
            i = j
        else:
            i += 1

    def rise(x, segs):
        out = []
        for (s, e) in segs:
            p0 = s * nf
            p1 = p0 + int(0.010 * SR)      # 起音后 10ms
            seg = x[p0:p1]
            if len(seg) < 2:
                continue
            out.append(float(np.max(np.abs(np.diff(seg)))))
        return np.array(out)

    ra_ = rise(a, segs)
    rb_ = rise(b, segs)
    print("   段数 %d" % len(segs))
    print("   起音 10ms 内最大 |Δ|:  原产 中位 %.4f  p90 %.4f"
          % (np.median(ra_), np.percentile(ra_, 90)))
    print("                        修复 中位 %.4f  p90 %.4f"
          % (np.median(rb_), np.percentile(rb_, 90)))
    print("   保留率 中位 %.1f%%  p90 %.1f%%"
          % (np.median(rb_ / ra_) * 100, np.percentile(rb_ / ra_, 90) * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
