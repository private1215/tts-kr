# -*- coding: utf-8 -*-
"""verify_0902f.py —— 验证三处「疑似多字/漏字」在波形上是否真的站得住。

候选：
  A. 58.20s  ASR 念出原文没有的「그래서」（员工 58s 报「额」）
  B. 99.3s   原文「익절」ASR 没听到（员工 100s 报「啊」）
  C. 141.66s ASR 念出原文没有的「더」（员工 142s 报「额」）
对每处打印 10ms 能量包络 + 浊音判据（zcr / 低频占比 / 谱形），
并和紧邻的静音段做对照 —— 如果该处只是静音，ASR 就是幻觉；
如果有独立浊音段，就是真的多出了一个音。
"""
import os
import subprocess

import numpy as np
import soundfile as sf

MP3 = "/mnt/tts/output/202610190902/202610190902-1.mp3"
WORK = "/mnt/tts/diag/v0902"
SR = 16000

CASES = [
    ("A 58.20s 多字「그래서」", 56.60, 59.00),
    ("B 99.28s 漏字「익절」", 98.60, 100.80),
    ("C 141.66s 多字「더」", 140.90, 142.60),
    ("对照：44s 员工报「额」但无 ASR 异常", 43.40, 45.20),
    ("对照：73s 员工报「吃字」但文本完整", 71.60, 74.80),
]


def to16k(path, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", out], check=True)
    return out


def feat(x, sr, t0, t1):
    a, b = int(t0 * sr), int(t1 * sr)
    s = x[a:b]
    if len(s) < 64:
        return None
    S = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    fr = np.fft.rfftfreq(len(s), 1.0 / sr)
    low = S[(fr > 100) & (fr < 400)].sum() / (S.sum() + 1e-12)
    mid = S[(fr > 400) & (fr < 3000)].sum() / (S.sum() + 1e-12)
    hi = S[fr > 3000].sum() / (S.sum() + 1e-12)
    zcr = float(np.mean(np.abs(np.diff(np.sign(s))) > 0))
    return dict(rms=float(np.sqrt(np.mean(s ** 2))), db=float(
        20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-12)),
        low=float(low), mid=float(mid), hi=float(hi), zcr=zcr)


def main():
    wav = os.path.join(WORK, "x.wav")
    if not os.path.exists(wav):
        to16k(MP3, wav)
    x, sr = sf.read(wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)

    hop = int(0.01 * sr)
    for name, t0, t1 in CASES:
        print("\n=== %s（窗口 %.2f~%.2f s）===" % (name, t0, t1))
        vals = []
        for i in range(int(t0 / 0.01), int(t1 / 0.01)):
            seg = x[i * hop:(i + 1) * hop]
            vals.append(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)
                        if len(seg) else -120)
        print("  10ms 能量 dBFS:")
        for k in range(0, len(vals), 50):
            chunk = vals[k:k + 50]
            print("    %6.2f-s  " % (t0 + k * 0.01) +
                  " ".join("%5.0f" % v for v in chunk))
        # 逐 50ms 细查浊音
        print("  逐 50ms 特征（浊音段 = 低频占比>0.5 且 zcr<0.15）：")
        t = t0
        while t < t1 - 0.05:
            f = feat(x, sr, t, t + 0.05)
            tag = ""
            if f and f["low"] > 0.5 and f["zcr"] < 0.15 and f["db"] > -45:
                tag = "  <== 浊音"
            if f:
                print("    %6.2f  %6.1f dB  低%.2f 中%.2f 高%.2f zcr%.3f%s"
                      % (t, f["db"], f["low"], f["mid"], f["hi"], f["zcr"], tag))
            t += 0.05


if __name__ == "__main__":
    main()
