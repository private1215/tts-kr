# -*- coding: utf-8 -*-
"""头部碎片段判别：切掉头部后 ASR，看首词是否残缺。
判据：
  持续语音起点 = 首个 >-25dB 且其后 300ms 均值 >-28dB 的位置
  切掉 [0, 起点) 后全文 ASR：
    - 首词仍完整(로보티즈) -> 碎片冗余，可切
    - 首词残缺(보티즈/티즈) -> 碎片含语音，不能切
"""
import numpy as np, soundfile as sf
from faster_whisper import WhisperModel

m = WhisperModel("Systran/faster-whisper-small", device="cpu",
                 compute_type="int8", local_files_only=True)

SR = 16000


def envdb(x, hop=160):
    n = len(x) // hop
    return 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-12)


def speech_onset(v):
    """首个满足 >-25dB 且后 300ms(30 点)均值 >-28dB 的位置"""
    for i in range(len(v) - 30):
        if v[i] > -25 and v[i:i + 30].mean() > -28:
            return i * 0.01
    return None


for k in (1, 2, 3, 4, 10):
    p = "/mnt/tts/diag/v0902/headrepro-1019-%d.16k.wav" % k
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    v = envdb(x)
    on = speech_onset(v)
    # 头部包络摘要
    loud = np.where(v > -35)[0]
    first_loud = loud[0] * 0.01 if len(loud) else None
    head = " ".join("%.2f:%+.0f" % (i * 0.01, v[i])
                    for i in range(0, min(len(v), int((on or 0.6) / 0.01) + 10), 2))
    # 切掉头部后全文 ASR
    cut = x[int(round((on or 0) * sr)):]
    segs, _ = m.transcribe(cut, language="ko", vad_filter=False)
    txt = " ".join(s.text for s in segs).strip()
    print("=== #%d ===" % k)
    print("  first>-35dB=%.2fs  持续语音起点=%.2fs" % (first_loud or -1, on or -1))
    print("  ENV %s" % head)
    print("  切头后 ASR: %s" % txt[:90])
