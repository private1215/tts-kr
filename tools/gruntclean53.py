# -*- coding: utf-8 -*-
"""gruntclean53.py —— 移除「极短填充音」（员工说的「特别特别短的嗯，超级短」）。

## 问题（2026-10-17 z25 员工反馈 + 离线定位）

员工原话：「第一道题 A 特别特别短的 嗯 超级短 存在」（两段男声都有）。
离线定位（16k 统一口径，`tools/grunt53b.py`）：

    裸输出 yohan 参考   12 个 (1.30/分钟)
    裸输出 EL 参考       3 个 (0.33/分钟)
    EL参考+fixchain      5 个 (0.54/分钟)
    现役成品 mp3         9 个 (0.96/分钟)
    男 EL 标杆           **0 个**

→ 「嗯」是 **OmniVoice 模型自发生成**（不是后处理引入：裸输出 12 → fixchain 后 9）。
→ 换 EL 参考能减少 50-75% 但消不掉；EL 本身是 0。
→ 员工明确听得见，且与她说 N 次的「断句有异声」吻合。

## 做法

检测「夹在两个长停顿之间的极短孤立发声段」：
    · 时长 20~120ms（5ms 帧 RMS，db > p35 判发声）
    · 前后静音都 > 0.15s（保证是孤立填充音，不是正常语流中的短音节）
满足则用**等长静音**替换（总时长不变），5ms 淡入淡出避免 click。

🔴 为什么用静音替换而不是删段：删段会改变总时长、破坏与其他指标的配比；
   静音替换只去掉那一声「嗯」，其余一个样本都不动。

🔴 为什么阈值卡在 120ms：韩语正常短音节（네/예/어）通常 >150ms，
   20~120ms 且前后都是 >0.15s 静音的组合，正常语流中几乎不会出现。

用法：
    python3 gruntclean53.py in.wav out.wav [--min-ms 20] [--max-ms 120]
                                      [--sil 0.15] [--dry]
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


def fade(a, ms=5):
    n = int(SR * ms / 1000)
    if len(a) < 2 * n:
        return a
    w = np.linspace(0, 1, n)
    a = a.copy()
    a[:n] *= w
    a[-n:] *= w[::-1]
    return a


def find_grunts(x, min_ms, max_ms, sil):
    """返回 [(start_s, dur_s)]，极短孤立发声段。"""
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > np.percentile(db, 35)
    segs, i = [], 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            segs.append((i * FRAME, (j - i) * FRAME))
            i = j
        else:
            i += 1
    out = []
    lo, hi = min_ms / 1000.0, max_ms / 1000.0
    for k in range(1, len(segs) - 1):
        s, d = segs[k]
        if lo <= d <= hi:
            pe = segs[k - 1][0] + segs[k - 1][1]
            nx = segs[k + 1][0]
            if (s - pe) > sil and (nx - (s + d)) > sil:
                out.append((s, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--min-ms", type=float, default=20.0)
    ap.add_argument("--max-ms", type=float, default=120.0)
    ap.add_argument("--sil", type=float, default=0.15)
    ap.add_argument("--dry", action="store_true", help="只报告不写文件")
    a = ap.parse_args()

    x = load16(a.src)
    g = find_grunts(x, a.min_ms, a.max_ms, a.sil)
    dur = len(x) / SR
    print("%s  %.1fs  极短填充音 %d 个 (%.2f/分钟)"
          % (os.path.basename(a.src), dur, len(g), len(g) / dur * 60), flush=True)
    for s, d in g:
        print("   t=%7.2fs  %3.0fms" % (s, d * 1000), flush=True)
    if a.dry or not g:
        if not g and not a.dry:
            sf.write(a.dst, x, SR)
            print("无需处理，已原样写出", flush=True)
        return 0

    y = x.copy()
    for s, d in g:
        i0 = int(s * SR)
        i1 = int((s + d) * SR)
        # 等长静音替换 + 5ms 淡入淡出（用邻近静音的电平，不是数字零）
        n = i1 - i0
        # 取该段前的静音电平做底（避免突然掉到 -inf）
        pre = x[max(0, i0 - int(0.2 * SR)):i0]
        floor = float(np.sqrt((pre ** 2).mean())) if len(pre) else 1e-4
        seg = np.full(n, floor, dtype=np.float32)
        y[i0:i1] = fade(seg)
    sf.write(a.dst, y, SR)
    print("-> %s  已替换 %d 处（等长静音，总时长不变）" % (a.dst, len(g)),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
