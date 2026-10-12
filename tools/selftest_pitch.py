# -*- coding: utf-8 -*-
"""音高偏差单变量实验：把成品整体音高抬到案例原声水平，看音色分回来多少。

🔴 结论（2026-10-13 实测，**阴性**）：**不要做。**
    男 0.8481 → 0.8440（−0.0040）；女 0.8743 → 0.8550（−0.0193）。
    不升反降，而且 Praat 实际只抬动了 0.7~2.0 Hz 就已经开始有损。
    说明 campplus 认的是声道/发音质感，不是绝对音高；
    线上成品比案例低 9~13Hz 是**结果**不是**原因**，硬抬只会引入处理痕迹。

留这个脚本是为了：下次再有人提出"我们把音高调上去是不是就更像了"，
不用重新写一遍实验，直接跑一次就有数。

用法
----
    python tools/selftest_pitch.py --ref <案例原声> --prod <成品> \
           --target 188.9 [--male]
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import engine_score_all as ES  # noqa: E402


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def pitch_median(path, is_male):
    import parselmouth
    snd = parselmouth.Sound(path)
    lo, hi = (75, 300) if is_male else (100, 500)
    p = snd.to_pitch(time_step=0.01, pitch_floor=lo, pitch_ceiling=hi)
    f = p.selected_array["frequency"]
    f = f[f > 0]
    return float(np.median(f)) if len(f) else 0.0


def shift(snd, hz, pfloor, out):
    import parselmouth
    # Praat 签名：Change gender(pitch_floor, pitch_median, formant_ratio,
    #                            range_ratio, dur_ratio, dur_factor)
    s = parselmouth.praat.call(snd.copy(), "Change gender",
                               pfloor, hz, 1.0, 1.0, 1.0, 1.0)
    s.save(out, "WAV")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--prod", required=True)
    ap.add_argument("--target", type=float, required=True, help="目标音高中位(Hz)")
    ap.add_argument("--male", action="store_true")
    ap.add_argument("--out-dir", default="/tmp/selftest")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    pfloor = 75.0 if a.male else 100.0
    e_ref = ES.timbre(a.ref)
    base = cos(e_ref, ES.timbre(a.prod))
    pm0 = pitch_median(a.prod, a.male)
    pm_ref = pitch_median(a.ref, a.male)
    print("参照音高中位 %.1fHz  成品 %.1fHz  基线音色 %.4f" % (pm_ref, pm0, base))

    import parselmouth
    snd = parselmouth.Sound(a.prod)
    for tag, hz in (("full", a.target), ("half", pm0 + (a.target - pm0) * 0.5)):
        out = os.path.join(a.out_dir, "pitch_%s.wav" % tag)
        shift(snd, hz, pfloor, out)
        c = cos(e_ref, ES.timbre(out))
        print("  抬到 %6.1fHz(实测 %.1f) → 音色 %.4f (%+.4f)"
              % (hz, pitch_median(out, a.male), c, c - base))


if __name__ == "__main__":
    main()
