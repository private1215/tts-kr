#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整篇 ASR 回检：模型到底有没有**多念 / 少念**字？

## 为什么跑这个
员工听到「断句处突然蹦出啊/额的短音」。两种可能修法完全不同：
  (a) 模型**多念了一个本来没有的音节** → 内容侧问题，可定位删除
  (b) 念的都是该念的，只是**韵律突兀** → 信号侧问题，要修停顿
只有把整篇转出来跟文案逐字对，才能分开 (a) 和 (b)。

## 🔴 口径修正（踩过的坑）
文案要取**原始前 1500 字符（含空格换行）**再去掉空白做比对。
之前先去空白再取 1500，位置整体后移，导致「转写内容和文案对不上」
的假象（看着像跑飞，其实是自己算错）。

## 输出
- CER（字符错误率）
- 对齐后的差异明细：多念 / 少念 / 读错，按**转写时间戳**定位
用法：/mnt/tts/projects/qwen3-tts/venv/bin/python full_cer.py
"""
import os
import re
import subprocess

import numpy as np
import soundfile as sf

SRC = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
TXT = "/mnt/tts/data/cases/52727ec43154/full.txt"
N_CHARS = 1500          # 🔴 含空格换行的原始字符数，与生成时一致
TMP = "/mnt/tts/diag/_fullcer"
os.makedirs(TMP, exist_ok=True)

KEEP = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]")


def norm(s):
    return "".join(KEEP.findall(s or ""))


def main():
    raw = open(TXT, encoding="utf-8").read()[:N_CHARS]     # 🔴 先截断，再去空白
    ref = norm(raw)
    print("文案：原始 %d 字符 -> 去空白 %d 音节" % (len(raw), len(ref)))

    f = os.path.join(TMP, "full.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", SRC,
                    "-ac", "1", "-ar", "16000", f], check=True)
    y, sr = sf.read(f, dtype="float32")
    print("音频 %.1fs，开始转写（whisper-medium/cpu，约需几分钟）…" % (len(y) / sr), flush=True)

    from faster_whisper import WhisperModel
    m = WhisperModel("Systran/faster-whisper-medium", device="cpu",
                     compute_type="int8", cpu_threads=4, local_files_only=True)
    segs, info = m.transcribe(y, language="ko", word_timestamps=True, beam_size=5,
                              condition_on_previous_text=False)

    hyp = norm("".join(s.text for s in segs))
    print("转写：%d 音节（语言 %s p=%.2f）" % (len(hyp), info.language, info.language_probability))

    # 字符级 CER
    import difflib
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    same = sum(b.size for b in sm.get_matching_blocks())
    cer = 1.0 - same / max(len(ref), 1)
    print("\nCER = %.3f（参考：线上回检阈值 0.35，超了判定内容跑飞）" % cer)
    print("文案 %d 音节 / 转写 %d 音节  差 %+d" % (len(ref), len(hyp), len(hyp) - len(ref)))

    print("\n=== 差异明细（按文案字符位置，换算成大致时间）===")
    cps = len(ref) / (len(y) / sr)           # 音节/秒（用去空白后算，自洽）
    n_show = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        seg_r, seg_h = ref[i1:i2], hyp[j1:j2]
        print("  %-8s 文案[%d:%d] %-14r  转写 %-14r   ≈%.1fs"
              % (tag, i1, i2, seg_r, seg_h, i1 / cps))
        n_show += 1
        if n_show >= 40:
            print("  …（差异过多，只列前 40 条）")
            break
    if n_show == 0:
        print("  无差异 —— 模型念的内容与文案完全一致")
    print("\nFULLCER_DONE")


if __name__ == "__main__":
    main()
