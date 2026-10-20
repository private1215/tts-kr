# -*- coding: utf-8 -*-
"""count_syl.py —— 用能量峰个数当音节数，判定三处争议到底有没有「多出来的音」。

韩语音节 ≈ 一个能量峰。全篇语速 7.5 字/秒 → 峰间距约 133ms。
对每个候选窗口数峰，和「原文应有的音节数」对：
  若峰数 ≈ 原文音节数 → 没有多字，ASR 幻觉；
  若峰数明显多于原文 → 真的多出了音。
"""
import os
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

MP3 = "/mnt/tts/output/202610190902/202610190902-1.mp3"
WORK = "/mnt/tts/diag/v0902"
SR = 16000
CPS = 7.5

CASES = [
    # (名字, 窗口起, 窗口止, 原文这段的音节数, 说明)
    ("A 58.1~59.4s", 58.10, 59.40, 5, "거래대금은（若多一个그래서则应为 8）"),
    ("B 98.9~100.1s", 98.90, 100.10, 6, "기계적인 익절（若漏익절则为 5）"),
    ("C 141.0~142.5s", 141.00, 142.50, 8, "하나만 보셔도 달달하게 中的 보셔도 달달하게"),
    ("对照 56.3~58.1s", 56.30, 58.10, 11, "다시 뚫어내야 됩니다"),
    ("对照 133.5~136.5s", 133.50, 136.50, 12, "눈으로 직접 보세요 매일"),
]


def to16k(path, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", out], check=True)
    return out


def main():
    wav = os.path.join(WORK, "x.wav")
    if not os.path.exists(wav):
        to16k(MP3, wav)
    x, sr = sf.read(wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    hop = int(0.01 * sr)
    print("%-20s %-8s %-8s %-10s %-10s %s"
          % ("窗口", "峰数", "原文音节", "实测cps", "应有cps", "判定"))
    for name, t0, t1, nsyl, note in CASES:
        a, b = int(t0 / 0.01), int(t1 / 0.01)
        env = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12)
                        for i in range(a, b)])
        db = 20 * np.log10(env + 1e-12)
        # 只在有话的部分数峰
        voiced = db > (np.median(db) + 6)
        pk, _ = find_peaks(db, height=np.median(db) + 6,
                           distance=int(0.09 / 0.01))
        dur = t1 - t0
        cps_obs = len(pk) / dur
        exp_cps = nsyl / dur
        verdict = "一致" if abs(cps_obs - exp_cps) < 0.9 * exp_cps else "⚠ 不一致"
        print("%-20s %-8d %-8d %-10.2f %-10.2f %s   %s"
              % (name, len(pk), nsyl, cps_obs, exp_cps, verdict, note))
        print("      峰位置(s): " +
              " ".join("%.2f" % (t0 + i * 0.01) for i in pk))


if __name__ == "__main__":
    main()
