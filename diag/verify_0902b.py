# -*- coding: utf-8 -*-
"""verify_0902b.py —— 逐点验证 v2：滤掉 whisper 数字规范化假象后重新对齐。

v1 的问题：whisper 把韩语数字（사십사만주）转写成阿拉伯数字（44만주），
字符级 diff 里全是假 insert/delete，把真信号埋了。
v2 做法：**两边都归一化数字**——
  原文侧：连续 2 个以上的韩语数词（일이삼사오육칠팔구십백천만억조）→ "0"
  ASR 侧：连续阿拉伯数字 → "0"
再 diff，剩下的才是真差异。同时打印每个反馈点 ±2s 内 ASR 到底念了什么。
"""
import difflib
import json
import os
import re
import subprocess

import numpy as np
import soundfile as sf

MP3 = "/mnt/tts/output/202610190902/202610190902-1.mp3"
SRC = "/mnt/tts/diag/src_202610190902.txt"
WORK = "/mnt/tts/diag/v0902"
os.makedirs(WORK, exist_ok=True)

POINTS = [
    (0, "开头 怪音"), (5, "磁音"), (14, "音一个字 没了"), (29, "吃字 一个"),
    (44, "额"), (58, "额"), (64, "兹 音"), (73, "吃字"), (83, "(无标注)"),
    (88, "(无标注)"), (100, "啊"), (115, "没音 一次"), (127, "额"),
    (135, "兹 音"), (142, "额"),
]

NUMKO = re.compile(r"[일이삼사오육칠팔구십백천만억조]{2,}")
NUMAR = re.compile(r"[0-9]+")
JUNK = re.compile(r"[^0-9A-Za-z가-힣]")


def norm(s):
    return JUNK.sub("", s or "")


def denum(s, korean):
    if korean:
        return NUMKO.sub("0", s)
    return NUMAR.sub("0", s)


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
    print("ASR 词数 %d" % len(words))

    # 原文字符 → 该字符在音频中的时间（用 ASR 词边界近似）
    char_t, asr_chars = [], []
    for st, en, tx in words:
        c = norm(tx)
        if not c:
            continue
        mid = (st + en) / 2.0
        for ch in c:
            char_t.append(mid)
            asr_chars.append(ch)
    asr_raw = "".join(asr_chars)

    src = open(SRC, encoding="utf-8").read()
    src_n = denum(norm(src), True)
    asr_n = denum(asr_raw, False)
    print("原文 %d 字 / ASR %d 字（数字归一化后 %d / %d）"
          % (len(norm(src)), len(asr_raw), len(src_n), len(asr_n)))

    sm = difflib.SequenceMatcher(None, src_n, asr_n, autojunk=False)
    ins, dele = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("insert", "replace"):
            ts = [char_t[k] for k in range(j1, min(j2, len(char_t)))]
            if ts:
                ins.append((min(ts), max(ts), asr_n[j1:j2], asr_raw[:0]))
        if tag in ("delete", "replace"):
            a = char_t[i1 - 1] if 0 < i1 <= len(char_t) else None
            b = char_t[i1] if i1 < len(char_t) else None
            lo = a if a is not None else (b if b is not None else 0.0)
            hi = b if b is not None else (a if a is not None else 586.0)
            dele.append((lo, hi, src_n[i1:i2]))

    print("\n=== 滤掉数字假象后：音频里多出来的（insert）===")
    for t0, t1, tx, _ in ins:
        print("  %7.2f ~ %7.2f  %s" % (t0, t1, tx))
    print("共 %d 处" % len(ins))
    print("\n=== 滤掉数字假象后：原文漏掉的（delete）===")
    for t0, t1, tx in dele:
        print("  %7.2f ~ %7.2f  %s" % (t0, t1, tx))
    print("共 %d 处" % len(dele))

    # 每点 ±2s 的 ASR 原词
    print("\n=== 每个反馈点 ±2s 内 ASR 实际念到的词 ===")
    for t, label in POINTS:
        near = [(st, en, tx) for st, en, tx in words if t - 2.0 <= st <= t + 2.0]
        print("\n  [%6.1fs] %s" % (t, label))
        print("     " + "  ".join("%.2f:%s" % (st - (t - 2.0), tx)
                                  for st, en, tx in near))
        # 该点前后 0.6s 的能量
        hop = int(0.01 * sr)
        rms = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, mode="same"))
        a, b = int((t - 0.6) * sr), int((t + 0.6) * sr)
        seg = rms[max(0, a):b]
        print("     RMS±0.6s: p50=%.4f  min=%.4f  max=%.4f" %
              (float(np.median(seg)), float(np.min(seg)), float(np.max(seg))))
        ni = [c for c in ins if abs(c[0] - t) <= 2.5]
        nd = [c for c in dele if abs(c[0] - t) <= 2.5]
        if ni:
            print("     ⚠ 附近多出来: " + " | ".join("%.2f %s" % c[:2] for c in ni))
        if nd:
            print("     ⚠ 附近漏掉:   " + " | ".join("%.2f %s" % c[:2] for c in nd))

    json.dump({"ins": ins, "del": dele}, open(os.path.join(WORK, "res2.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
