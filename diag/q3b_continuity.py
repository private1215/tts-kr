# -*- coding: utf-8 -*-
"""q3b_continuity.py —— Q3 追加：尾音是「自然衰减」还是「截断/拼接伪影」。

判据：
1. 前段（浊音段）时长分布 —— 我们如果系统性截短，尾音会显得突兀；
2. 尾音的基频 vs 前段末尾基频 —— 平滑延续 = 声带振动的自然释放；
   基频跳变/无基频 = 独立事件（伪影）；
3. 尾音的能量斜率 —— 自然衰减应单调下降；突然出现小峰 = 另起一个事件。
"""
import re
import subprocess

import numpy as np
import soundfile as sf

FILES = [
    ("我们成品", "/mnt/tts/diag/v0902/202610190902-1.mp3.16k.wav", False),
    ("EL 标杆", "/mnt/tts/diag/v0902/hm_merged.wav.wav", False),
]


def load(path, native=False):
    w = "/tmp/q3b.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", w], check=True)
    x, sr = sf.read(w, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def f0_of(s, sr):
    """自相关法测基频，返回 Hz 或 None。"""
    if len(s) < int(0.02 * sr):
        return None
    s = s - s.mean()
    ac = np.correlate(s, s, "full")[len(s) - 1:]
    if ac[0] <= 0:
        return None
    acn = ac / ac[0]
    lo, hi = int(0.0035 * sr), int(0.016 * sr)   # 62~286 Hz
    seg = acn[lo:hi]
    if len(seg) == 0:
        return None
    k = int(np.argmax(seg))
    if seg[k] < 0.3:
        return None
    return float(sr / (lo + k))


def rms_db(s):
    return 20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-12)


def main():
    for lab, p, _ in FILES:
        x, sr = load(p)
        hop = int(0.005 * sr)
        n = len(x) // hop
        rms = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12)
                        for i in range(n)])
        db = 20 * np.log10(rms + 1e-12)
        voiced = db > -40
        segs, i = [], 0
        while i < n:
            if voiced[i]:
                j = i
                while j < n and voiced[j]:
                    j += 1
                segs.append((i, j))
                i = j
            else:
                i += 1

        cands = []
        for k in range(1, len(segs)):
            pi, pj = segs[k - 1]
            ci, cj = segs[k]
            if (ci - pj) * 0.005 >= 0.03:
                continue
            ln = (cj - ci) * 0.005
            if not (0.005 <= ln <= 0.12):
                continue
            if (pj - pi) * 0.005 <= 0.10:
                continue
            s = x[ci * hop:cj * hop]
            if len(s) < 16:
                continue
            pk = float(np.max(np.abs(s)))
            pkdb = 20 * np.log10(pk + 1e-12)
            if not (-45 <= pkdb <= -20):
                continue
            zcr = float(np.mean(np.abs(np.diff(np.sign(s))) > 0))
            S = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
            fr = np.fft.rfftfreq(len(s), 1.0 / sr)
            lowf = float(S[fr < 300].sum() / (S.sum() + 1e-12))
            if not (zcr < 0.15 and lowf > 0.45):
                continue
            # 前段与尾音的基频
            pre = x[pi * hop:pj * hop]
            pre_tail = pre[-int(0.03 * sr):] if len(pre) > int(0.03 * sr) else pre
            f0_pre = f0_of(pre_tail, sr)
            f0_c = f0_of(s, sr) if ln >= 0.02 else None
            # 尾音内部 10ms 子帧能量斜率（末帧/首帧，dB）
            sub = int(0.01 * sr)
            nf = len(s) // sub
            slope = None
            if nf >= 3:
                e = [rms_db(s[i * sub:(i + 1) * sub]) for i in range(nf)]
                slope = e[-1] - e[0]
            cands.append({
                "t": ci * 0.005, "dur": ln, "pkdb": pkdb,
                "pre_dur": (pj - pi) * 0.005,
                "f0_pre": f0_pre, "f0_c": f0_c, "slope": slope,
            })

        print("=" * 80)
        print("%s：候选段 %d 个" % (lab, len(cands)))
        pre = np.array([c["pre_dur"] for c in cands])
        print("  前段时长 中位%.3fs  p10 %.3f  p90 %.3f  min %.3f"
              % (np.median(pre), np.percentile(pre, 10), np.percentile(pre, 90), pre.min()))
        both = [c for c in cands if c["f0_pre"] and c["f0_c"]]
        print("  前后都测到基频的段：%d 个" % len(both))
        for c in both[:20]:
            ratio = c["f0_c"] / c["f0_pre"]
            verdict = "延续" if 0.75 <= ratio <= 1.33 else ("跳变" if ratio < 0.5 or ratio > 2 else "偏离")
            print("    t=%7.2f 尾音%.0fms/%5.1fdB 前段%.0fms  f0 前%6.1fHz 尾%6.1fHz  比%.2f %s  斜率%s"
                  % (c["t"], c["dur"] * 1000, c["pkdb"], c["pre_dur"] * 1000,
                     c["f0_pre"], c["f0_c"], ratio, verdict,
                     ("%+.0fdB" % c["slope"]) if c["slope"] is not None else "n/a"))
        # 斜率统计
        sl = [c["slope"] for c in cands if c["slope"] is not None]
        if sl:
            sl = np.array(sl)
            print("  尾音内部能量斜率：中位%+.1fdB（负=衰减）  衰减(<0)占%.0f%%  上升(>3dB)占%.0f%%"
                  % (np.median(sl), 100 * float((sl < 0).mean()), 100 * float((sl > 3).mean())))


if __name__ == "__main__":
    main()
