#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3v35 长篇**完整性**验收（12 项标准第 12 条 / P0）。

只回答两件能证伪的事：
1. **覆盖率**：文案里的字有没有被真的念出来（漏读 = 覆盖率掉）。
2. **重复**：有没有复读/循环（`whisper-small` 在 >3min 上会循环复读，
   所以必须 30s 窗分段转写后再查，不能整篇喂）。

用法：venv/bin/python longcov53.py
"""
import argparse
import difflib
import glob
import os
import re
import sys
import tempfile

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")
from asr_check import normalize  # noqa: E402


def to16k(p, d):
    import subprocess
    o = os.path.join(d, os.path.basename(p).rsplit(".", 1)[0] + ".wav")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", p,
                    "-ac", "1", "-ar", "16000", o], check=True)
    return o


def read16k(w):
    import soundfile as sf
    x, sr = sf.read(w)
    return x, sr


def win_texts(m, x, sr, win=30.0):
    """30s 窗分段转写，返回每窗文本列表。"""
    import numpy as np
    n = int(win * sr)
    out = []
    for i in range(0, len(x), n):
        seg = x[i:i + n]
        if len(seg) < sr * 0.5:
            continue
        s, _ = m.transcribe(np.asarray(seg, dtype="float32"),
                            language="ko", vad_filter=False)
        out.append(" ".join(t.text.strip() for t in s))
    return out


def coverage(ref, hyp):
    """文案被念出来的比例：对归一化后的串做最长匹配块求和。"""
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return matched / max(len(ref), 1)


def repeats(win_texts, k=24, min_hit=3):
    """复读检测：长度 k 的片段在全篇出现 >=min_hit 次 -> 疑似循环。"""
    full = " ".join(win_texts)
    full_n = re.sub(r"\s+", "", full)
    seen = {}
    for i in range(len(full_n) - k):
        g = full_n[i:i + k]
        seen[g] = seen.get(g, 0) + 1
    hits = {g: c for g, c in seen.items() if c >= min_hit}
    # 相邻窗高度相似 = 原地打转
    adj = []
    for a, b in zip(win_texts, win_texts[1:]):
        r = difflib.SequenceMatcher(None, a[:200], b[:200], autojunk=False).ratio()
        if r > 0.90:
            adj.append(round(r, 3))
    return hits, adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=float, default=30.0)
    a = ap.parse_args()

    JOBS = [
        ("男声", "/mnt/tts/output/L3v35-male/L3v35-male-1.mp3",
         "/mnt/tts/data/cases/1a6c20a60481"),
        ("女声", "/mnt/tts/output/L3v35-female/L3v35-female-1.mp3",
         "/mnt/tts/data/cases/00b6a45ad6aa"),
    ]
    from faster_whisper import WhisperModel
    m = WhisperModel("small", device="cpu", compute_type="int8",
                     local_files_only=True)
    d = tempfile.mkdtemp(prefix="longcov_")
    for name, audio, cdir in JOBS:
        segs = sorted(glob.glob(os.path.join(cdir, "seg*.txt")),
                      key=lambda p: int(re.findall(r"seg(\d+)", p)[0]))
        ref = normalize(" ".join(open(p, encoding="utf-8").read().strip()
                                 for p in segs))
        w = to16k(audio, d)
        x, sr = read16k(w)
        wins = win_texts(m, x, sr, a.win)
        hyp = normalize(" ".join(wins))
        cov = coverage(ref, hyp)
        hits, adj = repeats(wins)
        print("=" * 62)
        print("%s  %s" % (name, os.path.basename(audio)))
        print("  音频 %.1fs  窗数 %d  文案(归一) %d 字  ASR(归一) %d 字"
              % (len(x) / float(sr), len(wins), len(ref), len(hyp)))
        print("  🔴 覆盖率 = %.4f   %s" % (cov, "✅" if cov >= 0.90 else "⚠️ 低于 0.90"))
        nmiss = len(ref) - len(hyp)
        print("  字数差(ASR-文案) = %+d  %s"
              % (-nmiss, "（ASR 更短：可能漏读，也可能是数字归一化差异）"
                 if nmiss > 0 else ""))
        print("  复读片段(>=3 次) = %d %s" % (len(hits), "✅" if not hits else "❌"))
        for g, c in sorted(hits.items(), key=lambda kv: -kv[1])[:5]:
            print("      x%d  %r" % (c, g))
        print("  相邻窗高度相似(>0.90) = %d %s" % (len(adj), "✅" if not adj else "❌ 疑似原地打转 %s" % adj))
        sys.stdout.flush()
    print("=" * 62)


if __name__ == "__main__":
    main()
