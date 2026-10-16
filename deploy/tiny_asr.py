#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把员工点名那几处（±2s）逐个转写，看**有没有比文案多念出一个音节**

为什么：blip_find 扫到的 95.50s 那处是「句号后、停顿前的孤立 200ms 元音」，
F0 297Hz、电平 −22.7dB（≈正常语音）。文案那儿是 "…있거든요.**주가가**…"。
如果转写比文案多一个 아/어 → 模型自己在句末加了个元音 = 员工听到的"啊"。

口径：whisper-medium（small 韩语不可信），强制 language=ko。
每个窗口 4 秒，打印逐词时间戳 + 该处文案（按 7.55 字/秒估算字符位置）。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python tiny_asr.py
"""
import os
import subprocess

import numpy as np
import soundfile as sf

SRC = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
TXT = "/mnt/tts/data/cases/52727ec43154/full.txt"
N_CHARS, DUR = 1500, 198.66
TMP = "/mnt/tts/diag/_tinyasr"
os.makedirs(TMP, exist_ok=True)

# (中心秒, 标注)  员工点名：95 / 99 / 112 / 170
MARKS = [(95.5, "第2题5s·blip命中"), (99.0, "第1题9s"),
         (112.0, "第1题22s"), (170.0, "第3题5s")]
WIN = 2.0


def main():
    txt = open(TXT, encoding="utf-8").read().replace("\n", " ").replace(" ", "")[:N_CHARS]
    cps = N_CHARS / DUR
    from faster_whisper import WhisperModel
    m = WhisperModel("Systran/faster-whisper-medium", device="cpu",
                     compute_type="int8", cpu_threads=4, local_files_only=True)

    for c, note in MARKS:
        s0 = max(0.0, c - WIN)
        f = os.path.join(TMP, "m%.0f.wav" % c)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", SRC, "-ss", "%.2f" % s0,
                        "-t", "%.1f" % (WIN * 2), "-ac", "1", "-ar", "16000", f], check=True)
        y, _ = sf.read(f, dtype="float32")
        segs, info = m.transcribe(y, language="ko", word_timestamps=True, beam_size=5,
                                  condition_on_previous_text=False)
        print("\n" + "=" * 76)
        print("【%.1fs】%s   窗口 %.1f~%.1fs" % (c, note, s0, s0 + WIN * 2))
        print("  文案(第%d字附近): …%s…" % (int(c * cps),
                                            txt[max(0, int(c * cps) - 20):int(c * cps) + 20]))
        print("  转写:")
        for s in segs:
            print("    %.2f~%.2f  %s" % (s.start + s0, s.end + s0, s.text.strip()))
            for w in getattr(s, "words", None) or []:
                tag = "  <<<点名" if abs(w.start + s0 - c) < 0.6 else ""
                print("      %.2f  %s%s" % (w.start + s0, w.word, tag))
        # 能量包络 10ms
        hop = 160
        n = len(y) // hop
        e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
        db = 20 * np.log10(e + 1e-12)
        print("  包络(10ms):")
        for i in range(0, n, 10):
            row = " ".join("%.0f" % v for v in db[i:i + 10])
            print("    %6.2fs %s" % (s0 + i * 0.01, row))
    print("\nTINYASR_DONE")


if __name__ == "__main__":
    main()
