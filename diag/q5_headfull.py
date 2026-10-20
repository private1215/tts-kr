# -*- coding: utf-8 -*-
"""头部定性收官：静音幻觉对照 + 四条问题文件全文词级转写"""
import numpy as np, soundfile as sf
from faster_whisper import WhisperModel

m = WhisperModel("Systran/faster-whisper-small", device="cpu",
                 compute_type="int8", local_files_only=True)

# ① 静音对照：1s 数字静音 + 1s -50dB 噪声底
print("=== 静音对照 ===")
for name, seg in (("digital_silence", np.zeros(16000, dtype="float32")),
                  ("noisefloor_-50dB", (np.random.RandomState(0).randn(16000) * 10 ** (-50 / 20)).astype("float32"))):
    segs, _ = m.transcribe(seg, language="ko", vad_filter=False)
    txt = " ".join(s.text for s in segs).strip()
    print("  %-16s -> %r" % (name, txt))

# ② 全文词级转写
for k in (2, 3, 4, 10):
    p = "/mnt/tts/diag/v0902/headrepro-1019-%d.16k.wav" % k
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    segs, _ = m.transcribe(x, language="ko", vad_filter=False,
                           word_timestamps=True)
    print("=== #%d 全文 ===" % k)
    for s in segs:
        ws = " ".join("%.2f:%s" % (w.start, w.word) for w in (s.words or []))
        print("  %.2f-%.2f %s" % (s.start, s.end, ws))
