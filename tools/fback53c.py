#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fback53c.py —— 「数字死寂」取证：我们 vs EL 的停顿底噪对照

假设：链路里有硬门把静音削成**数字零**（-120dB），而 EL 真录音的停顿有
房间底噪（-50~-60dB）。这会造成：句尾突兀切断（听成「嗯/啊/额」）、
句中掉音（0.1s 没音）、开关瞬态（「碰碰」）。

取证（不猜）：
1. 统计「严格数字零」帧（连续 N 个采样点全 0）的数量与位置分布
2. 停顿段的实际底噪：我们 vs EL 同口径
3. 句尾切断：语音 -> 数字零 的跳变有多陡（ms 级陡降 = 硬切）
"""
import subprocess
import sys

import numpy as np
import soundfile as sf

SR = 16000
T_END = 200.0

PAIRS = [
    ("男 我们", "/mnt/tts/output/202610210738/202610210738-1.mp3",
     "男 EL", "/mnt/tts/data/cases/1a6c20a60481/merged.wav"),
    ("女 我们", "/mnt/tts/output/202610210939/202610210939-1.mp3",
     "女 EL", "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"),
]


def load16k(path, t_end=T_END):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", str(SR), "-ac", "1", "-t", "%.1f" % t_end,
                    "/tmp/fb53c.wav"], check=True)
    y, _ = sf.read("/tmp/fb53c.wav", dtype="float32")
    return y


def zero_runs(y, min_ms=20.0):
    """连续全零段（>=min_ms）。返回 [(起点s, 时长s)] 与总占比。"""
    z = (np.abs(y) < 1e-9).astype(np.int8)
    n = int(min_ms / 1000 * SR)
    out, i = [], 0
    while i < len(z):
        if z[i]:
            j = i
            while j < len(z) and z[j]:
                j += 1
            if (j - i) >= n:
                out.append((i / SR, (j - i) / SR))
            i = j
        else:
            i += 1
    return out, float(z.mean())


def env(db_fn, y, hop_ms=10.0):
    hop = int(hop_ms / 1000 * SR)
    n = max(1, len(y) // hop)
    return 20 * np.log10(np.array(
        [np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2) + 1e-12)
         for i in range(n)]) + 1e-12), hop / SR


def pause_floor(y, db, dt):
    """停顿段（< -45dB）的底噪分布：中位 / 5% / 最小。"""
    p = db[db < -45]
    if len(p) == 0:
        return None, 0
    return (np.median(p), np.percentile(p, 5), p.min()), len(p)


def steep_drops(db, dt, ms=30.0, drop=25.0):
    """语音 -> 静音 的陡降：ms 毫秒内掉 drop dB 以上（硬切特征）。"""
    k = max(1, int(ms / 1000 / dt))
    out = []
    for i in range(k, len(db)):
        if db[i - k] > -40 and db[i] < -70 and db[i - k] - db[i] > drop:
            out.append((i * dt, round(db[i - k] - db[i], 1)))
    # 去重复（同一次切断只算一次）
    ded = []
    for t, d in out:
        if not ded or t - ded[-1][0] > 0.15:
            ded.append((t, d))
    return ded


def main():
    for tag_o, p_o, tag_e, p_e in PAIRS:
        print("=" * 74)
        for tag, path in ((tag_o, p_o), (tag_e, p_e)):
            y = load16k(path)
            zr, frac = zero_runs(y)
            db, dt = env(None, y)
            pf, npf = pause_floor(y, db, dt)
            sd = steep_drops(db, dt)
            print("\n  %s" % tag)
            print("    数字零帧占比        %.2f%%   连续≥20ms 的全零段 %d 处"
                  % (frac * 100, len(zr)))
            if zr:
                print("      前 5 处：%s"
                      % " ".join("%.1fs/%.2fs" % (a, d) for a, d in zr[:5]))
                ds = np.array([d for _, d in zr])
                print("      全零段时长：中位 %.2fs  最长 %.2fs" % (np.median(ds), ds.max()))
            if pf:
                print("    停顿段底噪 dB      中位 %.0f / 5%% %.0f / 最小 %.0f  (%d 帧)"
                      % (pf[0], pf[1], pf[2], npf))
            else:
                print("    停顿段底噪 dB      无 <-45dB 的帧（不停？）")
            print("    30ms 内陡降≥25dB   %d 处（硬切特征）" % len(sd))
            if sd:
                print("      前 6 处：%s"
                      % " ".join("%.1fs(-%.0fdB)" % (t, d) for t, d in sd[:6]))
        print()


if __name__ == "__main__":
    sys.exit(main())
