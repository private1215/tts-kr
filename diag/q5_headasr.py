# -*- coding: utf-8 -*-
"""文件头 ASR 定性：#2/#3/#4/#10 前 2.6s 逐段转写"""
import numpy as np, soundfile as sf
from faster_whisper import WhisperModel

m = WhisperModel("Systran/faster-whisper-small", device="cpu",
                 compute_type="int8", local_files_only=True)


def envdb(x, hop=160):
    n = len(x) // hop
    return 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)


for k in (2, 3, 4, 10):
    p = "/mnt/tts/diag/v0902/headrepro-1019-%d.16k.wav" % k
    x, sr = sf.read(p, dtype="float32")
    v = envdb(x)
    print("=== #%d  总长 %.2fs ===" % (k, len(x) / sr))
    # 包络（10ms/点）打印前 1.6s，方便对照 ASR 段落
    show = []
    for i in range(0, min(len(v), 160)):
        d = v[i]
        show.append("%.2f:%+.0f" % (i * 0.01, d))
    print("  ENV " + " ".join(show[:80]))
    print("  ENV " + " ".join(show[80:]))
    for a, b in ((0.0, 0.7), (0.6, 1.5), (1.3, 2.6)):
        seg = x[int(a * sr):min(len(x), int(b * sr))]
        if len(seg) < 1600:
            continue
        segs, _ = m.transcribe(seg, language="ko", vad_filter=False,
                               word_timestamps=True)
        words = []
        for s in segs:
            for w in (s.words or []):
                words.append("%.2f:%s" % (a + w.start, w.word))
        print("  [%.1f-%.1fs] %s" % (a, b, " ".join(words) if words else "(无)"))
