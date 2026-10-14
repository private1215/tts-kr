#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对任意 wav 的任意时间窗跑 faster-whisper，打印分段文本与时间戳。
只用于「听不了音频时确认某处在说什么 / 是否缺字糊字」。
用法: asr_win.py <wav> <start> <end> [标签]
"""
import sys, subprocess, tempfile, os
import numpy as np, soundfile as sf

MODEL_ID = "Systran/faster-whisper-small"


def cut16(path, ss, to):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(ss), "-to", str(to),
                    "-i", path, "-ar", "16000", "-ac", "1", t], check=True)
    y, sr = sf.read(t, dtype="float32")
    os.unlink(t)
    return y


def main():
    from faster_whisper import WhisperModel
    path = sys.argv[1]
    ss = float(sys.argv[2])
    to = float(sys.argv[3])
    label = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(path)
    m = WhisperModel(MODEL_ID, device="cpu", compute_type="int8",
                     cpu_threads=4, local_files_only=True)
    y = cut16(path, ss, to)
    segs, _ = m.transcribe(y, language="ko", beam_size=1,
                           vad_filter=False, word_timestamps=False)
    print(f"\n=== {label} [{ss}~{to}s] ===")
    nch = 0
    for s in segs:
        txt = s.text.strip()
        nch += len(txt.replace(" ", ""))
        print(f"  [{s.start+ss:7.2f}-{s.end+ss:7.2f}] ({s.end-s.start:4.2f}s) {txt}")
    print(f"  -- 识别字符数 {nch}，窗长 {to-ss:.1f}s，"
          f"字符/秒 {nch/(to-ss):.2f}")


main()
