# -*- coding: utf-8 -*-
"""asr_insert53.py —— 用 ASR 对齐原文案，找出「文案里没有的多余发音」(insert)。

## 为什么（用户 20:02 明确指出）

「最紧要的是清理文案里没有的多余发音嗯」

此前一直用声学特征（zcr / 低频占比 / dB）在**猜**哪些是「嗯」——三轮判据都只能
靠形态，无法回答「这个音是不是文案里的字」。

正解：**ASR 转写成品 → 与原文案字符级对齐 → 找出 insert（ASR 有、文案没有）**。
insert 就是「文案里没有、但被念出来的音」。

⚠️ 必须与 EL 同口径对照： whisper 会把韩语数字规范化（10→십）等，
  会产生**假的 insert**。只有「我们明显多于 EL」的 insert 才是真问题。

## 用法

    python3 asr_insert53.py <文案.txt> <音频1.wav> [音频2.wav ...] [--min N]

输出：每段 insert 的文本、时间区间、声学特征（峰值dB / zcr / 低频占比）。
"""
import argparse
import difflib
import re
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

SR = 16000
FRAME = 0.005


def load16(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    return x


def voiced_quality(a, sr=SR):
    if len(a) < 16:
        return 1.0, 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(a)))) / 2)
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a))) ** 2)
    fr = np.fft.rfftfreq(len(a), 1.0 / sr)
    low = float(sp[fr < 300].sum() / (sp.sum() + 1e-12))
    return zcr, low


def peak_db(a):
    if len(a) == 0:
        return -99.0
    return float(20 * np.log10(np.sqrt((a ** 2).mean()) + 1e-12))


def analyse(ref_txt, wav, model, min_len=2):
    x = load16(wav)
    ref = re.sub(r"\s+", "", ref_txt)
    segs, _ = model.transcribe(x, language="ko",
                               word_timestamps=True, vad_filter=False)
    words = []
    for s in segs:
        for w in (s.words or []):
            ww = re.sub(r"\s+", "", w.word or "")
            if ww:
                words.append({"w": ww, "s": w.start, "e": w.end})
    hyp = "".join(w["w"] for w in words)
    # 字符 -> word 索引
    char2w = []
    for wi, w in enumerate(words):
        for _ in w["w"]:
            char2w.append(wi)

    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    rows = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "insert":
            continue
        seg_txt = hyp[j1:j2]
        if len(seg_txt) < min_len:
            continue
        t0 = words[char2w[j1]]["s"] if j1 < len(char2w) else None
        t1 = words[char2w[j2 - 1]]["e"] if j2 - 1 < len(char2w) else None
        if t0 is None or t1 is None:
            continue
        a = x[int(t0 * SR):int(t1 * SR)]
        zcr, low = voiced_quality(a)
        rows.append({"t": seg_txt, "s": t0, "e": t1, "dur": t1 - t0,
                     "pk": peak_db(a), "zcr": zcr, "low": low})
    return rows, len(x) / SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--min", type=int, default=2)
    a = ap.parse_args()
    ref_txt = open(a.ref, encoding="utf-8").read()
    model = WhisperModel("small", device="cpu", compute_type="int8",
                         local_files_only=True)
    for w in a.wavs:
        rows, dur = analyse(ref_txt, w, model, a.min)
        print("=" * 60)
        print("%s  %.1fs  insert(>=%d字符) = %d 段"
              % (w.split("/")[-1], dur, a.min, len(rows)))
        for r in rows:
            print("  %6.2fs~%6.2fs %4.2fs 峰值%6.1fdB zcr=%.3f 低频=%.2f  「%s」"
                  % (r["s"], r["e"], r["dur"], r["pk"], r["zcr"],
                     r["low"], r["t"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
