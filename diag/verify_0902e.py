# -*- coding: utf-8 -*-
"""verify_0902e.py —— v5：修掉 v2 的时间映射 bug，拿准确的增删时间。

v2 的 bug（🔴 会影响结论）：
  char_t 是按**原始 ASR 文本**的字符下标建的，而 diff 是在**数字归一化后**的
  字符串上做的（사십사만 → 0，长度变了）→ 用原始下标去索引归一化后的 opcode，
  时间整体漂移。v2 报的 '그래서' 在 54.4s，而 ASR 词级时间戳明明在 58.2s。
v5 做法：归一化和建 char_t **同步做**，保证下标同构；
          delete 的时间用**夹住空隙的 ASR 字符**（j1-1, j1），不是源文下标。
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

    # 归一化与建 char_t 同步：每个词先去掉标点，再把连续数字替换成 "0"
    char_t, asr_chars = [], []
    for st, en, tx in words:
        c = norm(tx)
        c = NUMAR.sub("0", c)
        if not c:
            continue
        mid = (st + en) / 2.0
        for ch in c:
            char_t.append(mid)
            asr_chars.append(ch)
    asr_txt = "".join(asr_chars)

    src = open(SRC, encoding="utf-8").read()
    src_txt = NUMKO.sub("0", norm(src))
    print("原文 %d 字 / ASR %d 字" % (len(src_txt), len(asr_txt)))

    sm = difflib.SequenceMatcher(None, src_txt, asr_txt, autojunk=False)
    ins, dele = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("insert", "replace"):
            ts = [char_t[k] for k in range(j1, min(j2, len(char_t)))]
            if ts:
                ins.append((min(ts), max(ts), asr_txt[j1:j2]))
        if tag in ("delete", "replace"):
            # 空隙夹在 ASR 的 j1-1 与 j1 之间
            a = char_t[j1 - 1] if 0 < j1 <= len(char_t) else None
            b = char_t[j1] if j1 < len(char_t) else None
            lo = a if a is not None else (b if b is not None else 0.0)
            hi = b if b is not None else (a if a is not None else 586.0)
            dele.append((lo, hi, src_txt[i1:i2]))

    # 只保留「有信息量」的：长度 >= 2 的插入（单字多为 ASR 切分噪声）
    print("\n=== 音频里多出来的（insert，>=2 字）===")
    for t0, t1, tx in ins:
        if len(tx) >= 2:
            print("  %7.2f ~ %7.2f  %s" % (t0, t1, tx))
    print("\n=== 原文里漏掉的（delete，>=2 字）===")
    for t0, t1, tx in dele:
        if len(tx) >= 2:
            print("  %7.2f ~ %7.2f  %s" % (t0, t1, tx))

    print("\n=== 15 个反馈点 ±3s 内的 insert / delete（含单字）===")
    for t, label in POINTS:
        ni = [c for c in ins if abs(c[0] - t) <= 3.0]
        nd = [c for c in dele if abs(c[0] - t) <= 3.0]
        print("\n  [%6.1fs] %s" % (t, label))
        print("     insert: " + (" | ".join("%.2f %s" % c[:2] for c in ni)
                                or "无"))
        print("     delete: " + (" | ".join("%.2f %s" % c[:2] for c in nd)
                                or "无"))
        near = [(st, tx) for st, en, tx in words if t - 1.5 <= st <= t + 1.5]
        print("     附近 ASR 词: " + "  ".join("%.2f:%s" % (st, tx)
                                            for st, tx in near))

    json.dump({"ins": ins, "del": dele},
              open(os.path.join(WORK, "res3.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
