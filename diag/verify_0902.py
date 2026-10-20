# -*- coding: utf-8 -*-
"""verify_0902.py —— 逐点验证员工反馈（标题 202610190902）。

思路（全部走既有口径，不自造）：
1. ASR：10s 窗 + word_timestamps，拿每个词的时间戳 → 建立「字符 → 时间」映射。
2. 与原文（jobs.json 里的 4400 字）做字符级 diff：
   - **insert**（音频里有、原文没有）= 员工说的「额 / 啊 / 怪音」
   - **delete**（原文有、音频里没念）= 员工说的「吃字 / 没音 / 音一个字没了」
3. click 检测用现役口径：重采样 16k、**先归一到峰值 0.70**、`|Δ|>=0.30`。
4. 能量：短时 RMS，找掉字/静音。
用法: python3 verify_0902.py
"""
import difflib
import json
import os
import re
import subprocess
import sys

import numpy as np
import soundfile as sf

MP3 = "/mnt/tts/output/202610190902/202610190902-1.mp3"
SRC = "/mnt/tts/diag/src_202610190902.txt"
WORK = "/mnt/tts/diag/v0902"
os.makedirs(WORK, exist_ok=True)

# 员工反馈点（秒）
POINTS = [
    (0, "开头 怪音"), (5, "磁音"), (14, "音一个字 没了"), (29, "吃字 一个"),
    (44, "额"), (58, "额"), (64, "兹 音"), (73, "吃字"), (83, "(无标注)"),
    (88, "(无标注)"), (100, "啊"), (115, "没音 一次"), (127, "额"),
    (135, "兹 音"), (142, "额"),
]


def norm(s):
    """只留韩文音节 + 字母数字，去掉标点空白，便于对齐。"""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", s or "")


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
    dur = len(x) / sr
    print("音频时长 %.2f s" % dur)

    from faster_whisper import WhisperModel
    m = WhisperModel("Systran/faster-whisper-small", device="cpu",
                     compute_type="int8", local_files_only=True)

    win = 10.0
    n = int(win * sr)
    words = []          # (start, end, text)
    texts = []
    for i in range(0, len(x), n):
        seg = x[i:i + n]
        if len(seg) < sr * 0.5:
            continue
        off = i / sr
        segs, _ = m.transcribe(seg, language="ko", vad_filter=False,
                               word_timestamps=True)
        for s in segs:
            texts.append(s.text)
            for w in (s.words or []):
                if w.start is None:
                    continue
                words.append((off + w.start, off + w.end, w.word))
    print("ASR 词数 %d" % len(words))

    # 字符 → 时间映射
    char_t = []          # 每个归一化字符的时间（取该词中点）
    asr_chars = []
    for st, en, tx in words:
        c = norm(tx)
        if not c:
            continue
        mid = (st + en) / 2.0
        for ch in c:
            char_t.append(mid)
            asr_chars.append(ch)
    asr_txt = "".join(c for c in asr_chars)

    src = open(SRC, encoding="utf-8").read()
    src_txt = norm(src)
    print("原文 %d 字 / ASR %d 字" % (len(src_txt), len(asr_txt)))
    print("ASR 全文: %s" % asr_txt[:200])
    print("原文全文: %s" % src_txt[:200])

    sm = difflib.SequenceMatcher(None, src_txt, asr_txt, autojunk=False)
    ins = []   # (t_start, t_end, text)  音频多出来的
    dele = []  # (t_start, t_end, text)  原文漏掉的
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert" or tag == "replace":
            # ASR 侧 j1..j2 是「多出来的字」
            ts = [char_t[k] for k in range(j1, min(j2, len(char_t)))]
            if ts:
                ins.append((min(ts), max(ts), asr_txt[j1:j2]))
        if tag == "delete" or tag == "replace":
            # 原文侧 i1..i2 是「漏掉的字」，时间夹在两侧相等块之间
            a = char_t[i1 - 1] if i1 - 1 < len(char_t) and i1 > 0 else None
            b = char_t[i1] if i1 < len(char_t) else None
            lo = a if a is not None else (b if b is not None else 0.0)
            hi = b if b is not None else (a if a is not None else dur)
            dele.append((lo, hi, src_txt[i1:i2]))

    print("\n=== 音频里多出来的字（insert，疑似 额/啊/怪音）===")
    for t0, t1, tx in ins:
        print("  %7.2f ~ %7.2f  %s" % (t0, t1, tx))
    print("共 %d 处" % len(ins))

    print("\n=== 原文里漏掉的字（delete，疑似 吃字/没音）===")
    for t0, t1, tx in dele:
        print("  %7.2f ~ %7.2f  %s" % (t0, t1, tx))
    print("共 %d 处" % len(dele))

    # click 检测（现役口径）
    pk = float(np.max(np.abs(x)))
    y = x * (0.70 / pk) if pk > 0 else x
    d = np.abs(np.diff(y))
    idx = np.where(d >= 0.30)[0]
    # 合并相邻（间隔 < 100 样本算同一处）
    groups = []
    for i in idx:
        if groups and i - groups[-1][-1] <= 100:
            groups[-1].append(i)
        else:
            groups.append([i])
    print("\n=== click 检测（峰值归一到 0.70，|Δ|>=0.30，16k）===")
    clicks = []
    for g in groups:
        t0 = g[0] / sr
        mx = float(np.max(d[g[0]:g[-1] + 1]))
        clicks.append((t0, mx))
    for t0, mx in clicks:
        print("  %7.2f  |Δ|=%.3f" % (t0, mx))
    print("共 %d 处" % len(clicks))

    # 短时能量（找掉字/静音）
    print("\n=== 每个反馈点附近的能量与 ASR ===")
    hop = int(0.02 * sr)
    rms = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, mode="same"))
    for t, label in POINTS:
        i = int(t * sr)
        seg = rms[max(0, i - int(1.0 * sr)):i + int(1.0 * sr)]
        seg = seg[seg > 0]
        near_i = [c for c in ins if abs(c[0] - t) <= 3.0]
        near_d = [c for c in dele if abs(c[0] - t) <= 3.0]
        near_c = [c for c in clicks if abs(c[0] - t) <= 2.0]
        print("\n  [%6.1fs] %s" % (t, label))
        print("     RMS(±1s) p50=%.4f min=%.4f max=%.4f" %
              (float(np.median(seg)), float(np.min(seg)), float(np.max(seg))))
        if near_i:
            print("     多出来的字: " + " | ".join("%.2f~%.2f %s" % c for c in near_i))
        if near_d:
            print("     漏掉的字:   " + " | ".join("%.2f~%.2f %s" % c for c in near_d))
        if near_c:
            print("     click:      " + " | ".join("%.2f |Δ|=%.3f" % c for c in near_c))
        # 该时刻正在念的原文片段
        k = None
        for j, ct in enumerate(char_t):
            if ct > t:
                k = j
                break
        if k is not None:
            print("     该时刻原文位置: ...%s【%s】%s..." %
                  (src_txt[max(0, k - 6):k], src_txt[k], src_txt[k + 1:k + 7]))
            print("     该时刻 ASR:     ...%s【%s】%s..." %
                  (asr_txt[max(0, k - 6):k], asr_txt[k], asr_txt[k + 1:k + 7]))

    json.dump({"ins": ins, "del": dele, "clicks": clicks,
               "words": words[:50]}, open(os.path.join(WORK, "res.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
