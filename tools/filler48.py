# -*- coding: utf-8 -*-
"""filler48.py —— 把「凭空多出来的音」（填充音 / filler）定位到第几秒。

## 起因（员工原话，三次指向同一现象）
- 用户听 `xf5272-prod`：「说完 你 **马上**出来有个'嘎哈'的 0.2 秒音」
- z18 留言（디아）：「**停顿时有 啊的声音**」
- z15 留言（디아）：「第五个问题音频**背景有异声**」

`cerdiff48.py` 已经证明它**不是我的错觉**：ASR diff 里有真正的 insert
（原文没有、音频里却念出来的东西），例如：

    [insert] ref=«»  hyp=«에...»     ← "에" 这种填充音
    [insert] ref=«»  hyp=« 와 2»

🔴 注意这跟记忆里已封存的「句间停顿**往下沉**」是两回事，
   那是能量下沉、这次是**多出来一段音**，不能套旧结论。

## 本脚本做什么
用 faster_whisper 的 `word_timestamps` 拿到词级时间戳，逐词打印，
人工/正则就能定位那些"原文里没有的词"在第几秒。
再配合 `ffmpeg -ss` 切出来听，即可确认是不是员工说的那个音。

用法： python filler48.py <音频> [--grep 에|아|어]
"""
import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")

SR = 16000


def to16k(p):
    if p.endswith(".wav"):
        return p
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--grep", default="", help="只打印含这些字符的词（如 에아어）")
    ap.add_argument("--lang", default="ko")
    a = ap.parse_args()

    import asr_check as A
    m = A._load()
    if m is None:
        print("❌ ASR 模型不可用")
        return 1
    print("=== %s ===" % os.path.basename(a.wav))
    # 🔴 必须传 ndarray 不能传路径：av ≥19 与 faster-whisper 的
    #    decode_audio 参数不兼容（实测 TypeError: metadata_errors）。
    #    A._read_16k() 还会把 24k 成品重采样到 16k —— 直接传数组
    #    faster-whisper 假定 16k，不重采样会导致音高语速全错。
    segs, info = m.transcribe(A._read_16k(a.wav),
                              language=A.lang_code(a.lang),
                              word_timestamps=True,
                              vad_filter=True, beam_size=1)
    hits = 0
    for s in segs:
        for wd in (s.words or []):
            t = (wd.word or "").strip()
            if a.grep and not any(c in t for c in a.grep):
                continue
            hits += 1 if a.grep else 0
            print("  %6.2fs ~ %6.2fs  %s" % (wd.start, wd.end, t))
        if not a.grep:
            print("  [%5.2f ~ %5.2f] %s" % (s.start, s.end, (s.text or "").strip()))
    if a.grep:
        print("\n  含 [%s] 的词共 %d 个" % (a.grep, hits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
