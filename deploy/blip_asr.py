#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 Whisper 转写员工点名的那几段，把「啊/额」短音变成**看得见的字**

员工（两位一致）在现役女声成品里听到断句处蹦出「啊 / 额」短音，给的秒数：
    第1题(90~120s): 9s → 99s、22s → 112s
    第2题(90~120s): 5s → 95s
    第3题(165~185s): 5s → 170s

光看波形分不出「模型多念了一个字」还是「念了一个本来就该有的字」。
唯一办法是**转写**：把音频里实际出现的音节列出来，再和文案逐字对。
若转写结果比文案多出 아 / 어 / 으 之类 → 伪音，可删。
若转写和文案一致 → 真词，问题在韵律（时长/能量）不在内容。

同时打印 5ms 粒度能量包络的 ASCII 图，看清短音的形状（元音=平滑，爆音=尖峰）。
⚠️ 必须用 whisper-medium：small 在韩语上错得离谱（已踩过坑）。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python blip_asr.py
"""
import os
import subprocess

import numpy as np
import soundfile as sf

SRC = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
TXT = "/mnt/tts/data/cases/52727ec43154/full.txt"
N_CHARS = 1500
DUR = 198.66
TMP = "/mnt/tts/diag/_blipasr"
os.makedirs(TMP, exist_ok=True)

# (片段起点, 时长, 员工点名的绝对秒)
WINDOWS = [
    (90.0, 30.0, [95.0, 99.0, 112.0]),
    (165.0, 20.0, [170.0]),
]


def read16(path):
    out = os.path.join(TMP, "w.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ac", "1", "-ar", "16000", out], check=True)
    return sf.read(out, dtype="float32")[0], 16000


def env_ascii(y, sr, s0, s1, t0):
    """5ms 粒度包络 ASCII，行=0.25s"""
    a, b = int(s0 * sr), int(s1 * sr)
    y = y[a:b]
    hop = int(0.005 * sr)
    n = len(y) // hop
    e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    db = 20 * np.log10(e + 1e-12)
    lo, hi = np.percentile(db, 5), np.percentile(db, 95)
    rows = []
    per = 50                      # 50*5ms = 0.25s 一行
    for i in range(0, n, per):
        seg = db[i:i + per]
        if not len(seg):
            continue
        m = seg.mean()
        lvl = int(np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1) * 20)
        rows.append("%7.2f |%s| %5.1fdB" % (t0 + i * 0.005, "█" * lvl, m))
    return "\n".join(rows)


def main():
    txt = open(TXT, encoding="utf-8").read().replace("\n", " ").replace(" ", "")
    txt = txt[:N_CHARS]
    cps = N_CHARS / DUR                    # 字/秒（成品实测）

    from faster_whisper import WhisperModel
    print("加载 whisper-medium (cpu/int8)…", flush=True)
    m = WhisperModel("Systran/faster-whisper-medium", device="cpu",
                     compute_type="int8", cpu_threads=4, local_files_only=True)

    for s0, dur, marks in WINDOWS:
        f = os.path.join(TMP, "seg_%.0f.wav" % s0)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", SRC,
                        "-ss", "%.1f" % s0, "-t", "%.1f" % dur,
                        "-ac", "1", "-ar", "16000", f], check=True)
        print("\n" + "=" * 78)
        print("【成品 %.0fs ~ %.0fs】员工点名：%s" % (s0, s0 + dur, marks))
        y, sr = sf.read(f, dtype="float32")
        segs, info = m.transcribe(y, language="ko", word_timestamps=True,
                                  beam_size=5, condition_on_previous_text=False)
        print("  语言 %s (p=%.2f)" % (info.language, info.language_probability))
        print("  --- 逐段转写（时间为成品绝对秒）---")
        for s in segs:
            t = "%.2f~%.2f" % (s.start + s0, s.end + s0)
            hit = " <<<点名" if any(x - 1.5 <= s.start + s0 <= x + 1.5 for x in marks) else ""
            print("   %-16s %s%s" % (t, s.text.strip(), hit))
            if s.words:
                for w in s.words:
                    wh = " <<<" if any(x - 1.0 <= w.start + s0 <= x + 1.0 for x in marks) else ""
                    print("        %.2f %s%s" % (w.start + s0, w.word, wh))

        print("\n  --- 文案对照（按 %.2f 字/秒估算字符位置）---" % cps)
        for x in marks:
            c = int(x * cps)
            print("    %.0fs -> 第%d字: …%s…" % (x, c, txt[max(0, c - 25):c + 25]))

        print("\n  --- 能量包络（点名 ±2.5s）---")
        for x in marks:
            print("   ▼ %.0fs 附近" % x)
            print(env_ascii(read16(SRC)[0], 16000, x - 1.5, x + 2.5, x - 1.5))
    print("\nBLIPASR_DONE")


if __name__ == "__main__":
    main()
