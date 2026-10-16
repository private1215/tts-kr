#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给「断句异声」的候选短段做特征分析：哪些是伪音（该删），哪些是真词（不能删）

两位员工一致反馈「都挺好，就断句有异声」+ 精确秒数（成品 95 / 99 / 112 / 170s）。
`blip_find.py` 已扫出全篇 12 处「停顿中的孤立短发声段」，且**几乎全在句号/逗号后**。

🔴 不能直接全删：20.78s 那处文案是 "…있거든요. **자,** 심호흡…"（자 = "来"），是真词。
   必须靠特征把伪音和真词分开。

## 判据（伪音 = 句尾残留的元音尾巴）
- **起音**：伪音起音平滑（元音直接出来）；真词有辅音瞬态（能量陡升 + 高频爆音）
- **时长**：伪音多 < 250ms
- **能量**：伪音通常低于全篇发声均值（模型收尾时能量已衰减）
- **谐度**：伪音是纯元音 → 谐波明显、频谱平坦度低
用法：/mnt/tts/projects/qwen3-tts/venv/bin/python blip_feat.py
"""
import os
import subprocess

import numpy as np
import librosa
import soundfile as sf

SRC = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
TMP = "/mnt/tts/diag/_blip"
os.makedirs(TMP, exist_ok=True)

# blip_find.py 扫出的 12 处：(起点, 时长)
HITS = [(8.86, 0.225), (20.78, 0.130), (35.11, 0.080), (66.02, 0.095),
        (69.48, 0.095), (78.98, 0.185), (85.38, 0.205), (95.50, 0.200),
        (135.00, 0.115), (146.74, 0.160), (185.00, 0.215), (189.34, 0.105)]
MARK = [95.0, 99.0, 112.0, 170.0]


def load16():
    out = os.path.join(TMP, "b.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", SRC,
                    "-ac", "1", "-ar", "16000", out], check=True)
    return sf.read(out, dtype="float32")


def main():
    y, sr = load16()
    hop = int(0.005 * sr)
    n = len(y) // hop
    e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    db = 20 * np.log10(e + 1e-12)
    thr = np.percentile(db, 25) + 6
    voiced_db = db[db > thr].mean()          # 全篇发声平均电平（参照）

    print("全篇发声平均电平 %.1f dB\n" % voiced_db)
    print("%-7s %-6s %-7s %-7s %-7s %-7s %s"
          % ("起点", "时长", "电平dB", "起音dB", "平坦度", "F0中位", "判定"))
    for t, ln in HITS:
        a, b = int(t * sr), int((t + ln) * sr)
        seg = y[a:b]
        lvl = 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-12)
        # 起音：开头 20ms 的能量 vs 整段能量（辅音瞬态会让开头更"冲"）
        head = seg[:int(0.02 * sr)]
        head_db = 20 * np.log10(np.sqrt((head ** 2).mean()) + 1e-12)
        onset = head_db - lvl
        flat = float(librosa.feature.spectral_flatness(y=seg, hop_length=256).mean())
        try:
            f0, _, _ = librosa.pyin(seg, fmin=100, fmax=500, sr=sr)
            v = f0[np.isfinite(f0)]
            f0m = float(np.median(v)) if len(v) else float("nan")
            f0cv = float(np.std(v) / np.mean(v)) if len(v) > 3 else float("nan")
        except Exception:
            f0m = f0cv = float("nan")
        # 判定：电平低于发声均值 + 起音平滑(onset<3dB) + 短 → 伪音
        low = lvl < voiced_db - 1.0
        smooth = onset < 3.0
        verdict = "伪音(可删)" if (low and smooth) else "疑真词(保留)"
        flag = " <<<员工点名" if any(abs(t - m) < 1.5 for m in MARK) else ""
        print("%-7.2f %-6.3f %-7.1f %-7.1f %-7.4f %-7.1f %s%s"
              % (t, ln, lvl, onset, flat, f0m, verdict, flag))
    print("\nBLIPFEAT_DONE")


if __name__ == "__main__":
    main()
