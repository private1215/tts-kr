#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第十八轮盲听切片（k15）：v10（现役）vs PSOLA v2，让员工裁决客观指标的冲突。

## 为什么出这一轮
上一轮（k14）员工判定 v10 无改善。顺着查下去发现真凶是**共振峰**，
于是做了 PSOLA（只改 F0 不动频谱包络）。但两者各赢一半，指标裁决不了：

                  F1偏离  F2偏离   频段J   flat(16k)   EL=409.8/1555.4/0.0424
    v10 现役      +3.8%  +3.8%     5.9    0.0356     ← 频段与自然度赢
    PSOLA v2      -0.5%  -2.9%    88.3    0.0251     ← 共振峰赢（感冒音对应项）

「共振峰 vs 频段/自然度」哪个对员工感知更重要，只能靠耳朵。
所以这一轮不预设答案，就是让员工在两个方案里挑。

## 听感对应关系（供解读用）
- F1/F2 偏高 = 声道变短 = 鼻塞 / 感冒 / 发假  → PSOLA 应在这项赢
- flat 低    = 频谱过净 = 机器味 / 发假        → v10 应在这项赢

## 控制变量
三者语速已拉到同一水平：v10 8.49 / PSOLA 8.46 / EL 8.46 字每秒，
语速不会成为混淆项。切片统一取第 300s、20 秒、LUFS 拉平到 -26.1。
🔴 响度用 volume 拉平，**不用 loudnorm**；⚠️ 必须写 `volume=%gdB`（多一个 B 会变 +18dB）。
"""
import os
import re
import subprocess
import tempfile

OUT = "/mnt/tts/data/tone-ab"
DIAG = "/mnt/tts/diag/x8197"
EL = "/mnt/tts/data/cases/8197d9ac8a35/merged.wav"
V10 = DIAG + "/m_v10.wav"           # 现役：asetrate 变调 + 自动搜 EQ
PS = DIAG + "/m_psola_v2.wav"       # 新方案：PSOLA 变调 + 专用 EQ
TARGET = -25.7
DUR = 20

# 题 2 与题 3 用同一对音频、顺序相反 —— 检出「位置惯性」
CUTS = [
    ("k15_c1", PS, 300, DUR),       # 控制题第 1 遍
    ("k15_c2", PS, 300, DUR),       # 控制题第 2 遍（正确=一样快）
    ("k15_v_t", V10, 300, DUR),     # v10 现役
    ("k15_p_t", PS, 300, DUR),      # PSOLA v2
    ("k15_el_t", EL, 300, DUR),     # EL 标杆
]


def lufs(p):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", p, "-af", "ebur128",
                        "-f", "null", "-"], capture_output=True, text=True)
    ms = re.findall(r"I:\s+(-?[\d.]+)\s+LUFS", r.stderr)
    return float(ms[-1]) if ms else None


def main():
    for name, src, ss, dur in CUTS:
        assert os.path.exists(src), src
        tmpw = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(ss),
                        "-t", str(dur), "-i", src, "-ar", "44100", "-ac", "1",
                        tmpw], check=True)
        cur = lufs(tmpw)
        gain = round(TARGET - cur, 2)
        tmp2 = tempfile.mktemp(suffix=".mp3")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmpw,
                        "-af", "volume=%gdB" % gain,
                        "-codec:a", "libmp3lame", "-b:a", "128k",
                        "-ar", "44100", tmp2], check=True)
        dst = os.path.join(OUT, name + ".mp3")
        if subprocess.run(["cp", tmp2, dst]).returncode != 0:
            subprocess.run(["sudo", "-S", "-p", "", "cp", tmp2, dst],
                           input=os.environ.get("TTS_SUDO_PW", "a19910119") + "\n",
                           text=True, check=True)
        got = lufs(dst)
        flag = "" if abs(got - (-26.1)) < 0.35 else "  ⚠️响度偏离，检查"
        print("%-9s 源 %-12s LUFS %6.2f -> 增益 %+.2f dB -> %.2f%s"
              % (name, os.path.basename(src), cur, gain, got, flag))
        os.unlink(tmpw)
        os.unlink(tmp2)


if __name__ == "__main__":
    main()
