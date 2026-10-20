# -*- coding: utf-8 -*-
"""local_cps.py —— 每个反馈点附近的**局部语速**，判定有没有多出/漏掉音节。

全局口径：全篇 cps = 7.50（与 EL 女声锚 7.45 一致）→ 全局没有系统性多字。
但员工报的是**局部**问题，所以逐点算：
    局部 cps = ASR 在该窗口念出的音节数 / 该窗口的净发声时长
- 局部 cps 明显高于 7.5 → 该处多出了音（ASR 把原文没有的音也念出来了）
- 明显低于 7.5 → 该处漏音或拖长
同时给出「原文在该窗口应有的音节数」做双向对照。
"""
import json
import os
import re
import subprocess

import numpy as np
import soundfile as sf

MP3 = "/mnt/tts/output/202610190902/202610190902-1.mp3"
SRC = "/mnt/tts/diag/src_202610190902.txt"
WORK = "/mnt/tts/diag/v0902"
SR = 16000
GLOBAL_CPS = 7.50

POINTS = [
    (0, "开头 怪音"), (5, "磁音"), (14, "音一个字 没了"), (29, "吃字 一个"),
    (44, "额"), (58, "额"), (64, "兹 音"), (73, "吃字"), (83, "(无标注)"),
    (88, "(无标注)"), (100, "啊"), (115, "没音 一次"), (127, "额"),
    (135, "兹 音"), (142, "额"),
]
JUNK = re.compile(r"[^0-9A-Za-z가-힣]")


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

    wj = os.path.join(WORK, "words.json")
    if os.path.exists(wj):
        words = json.load(open(wj))
        words = [tuple(w) for w in words]
    else:
        from faster_whisper import WhisperModel
        m = WhisperModel("Systran/faster-whisper-small", device="cpu",
                         compute_type="int8", local_files_only=True)
        win = 10.0
        n = int(win * sr)
        words = []
        for i in range(0, len(x), n):
            seg = x[i:i + n]
            if len(seg) < sr * 0.5:
                continue
            off = i / sr
            segs, _ = m.transcribe(seg, language="ko", vad_filter=False,
                                   word_timestamps=True)
            for s in segs:
                for w in (s.words or []):
                    if w.start is None:
                        continue
                    words.append((off + w.start, off + w.end, w.word))
        json.dump(words, open(wj, "w"), ensure_ascii=False)

    src = JUNK.sub("", open(SRC, encoding="utf-8").read())
    # 原文音节 → 时间（按全局 cps 线性映射，只用于取「该窗口原文有多少字」）
    total = len(src)

    hop = int(0.01 * sr)
    nb = len(x) // hop
    rms = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12)
                    for i in range(nb)])
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > -40

    def voiced_dur(t0, t1):
        a, b = int(t0 / 0.01), int(t1 / 0.01)
        return float(voiced[max(0, a):b].sum()) * 0.01

    print("全篇 ASR 音节 %d，音频 %.1fs，全局 cps = %.2f"
          % (total, len(x) / sr, GLOBAL_CPS))
    print("\n%-6s %-12s %-7s %-7s %-8s %-8s %-8s %s"
          % ("时间", "员工标注", "ASR字", "发声s", "局部cps", "原文cps", "判定", "附近原文"))
    for t, label in POINTS:
        w = [(st, en, tx) for st, en, tx in words if t - 2.0 <= st <= t + 2.0]
        nsyl = sum(len(JUNK.sub("", tx)) for st, en, tx in w)
        vd = voiced_dur(t - 2.0, t + 2.0)
        lcps = nsyl / vd if vd > 0.3 else float("nan")
        i0 = int((t - 2.0) / (len(x) / sr) * total)
        i1 = int((t + 2.0) / (len(x) / sr) * total)
        nsrc = i1 - i0
        scps = nsrc / vd if vd > 0.3 else float("nan")
        if lcps != lcps:
            verdict = "无足够发声"
        elif lcps > GLOBAL_CPS * 1.35:
            verdict = "⚠ 多字（局部偏快）"
        elif lcps < GLOBAL_CPS * 0.65:
            verdict = "⚠ 少字/拖长"
        else:
            verdict = "正常"
        print("%-6.0f %-12s %-7d %-7.2f %-8.2f %-8.2f %-8s %s"
              % (t, label, nsyl, vd, lcps, scps, verdict,
                 src[max(0, i0):i1][:26]))


if __name__ == "__main__":
    main()
