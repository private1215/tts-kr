# -*- coding: utf-8 -*-
"""check_template.py —— 体检女声模版 EL_female.wav（15.08s / 24kHz）。

模版是本条产出的音色来源，员工选的女声。查四件事：
1. ref_text 与音频是否真对齐（ASR 回读比对）—— 不对齐会直接毁克隆质量；
2. click（现役口径：峰值归一到 0.70、|Δ|>=0.30、16k）；
3. 句末粘连嗯密度（同 tailgrunt53e 判据）；
4. 语速 cps 与 voices.json 里记的 7.62 是否一致。
"""
import os
import re
import subprocess

import numpy as np
import soundfile as sf

TPL = "/mnt/tts/data/voices/EL_female.wav"
SRC = "/mnt/tts/diag/tpl"
os.makedirs(SRC, exist_ok=True)
REF_TEXT = ("벌어질 수밖에 없습니다. 제 멘트를 무조건 맹신하기보다는 아 이 자리 이탈하면 손절이고 "
            "뚫어주면 홀딩이구나 하는 시나리오 매매의 뼈대로 삼으셔야 합니다. 지금 다들 비슷한 "
            "고민으로 밤잠 설치실 겁니다.")
JUNK = re.compile(r"[^0-9A-Za-z가-힣]")


def to16k(path, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", out], check=True)
    return out


def main():
    wav = os.path.join(SRC, "t.wav")
    to16k(TPL, wav)
    x, sr = sf.read(wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    dur = len(x) / sr
    print("模版时长 %.2f s，原采样率 %d Hz" % (dur, sf.info(TPL).samplerate))
    print("ref_text %d 字（含标点 %d）→ cps = %.2f"
          % (len(JUNK.sub("", REF_TEXT)), len(REF_TEXT),
             len(JUNK.sub("", REF_TEXT)) / dur))

    # ASR 回读
    from faster_whisper import WhisperModel
    m = WhisperModel("Systran/faster-whisper-small", device="cpu",
                     compute_type="int8", local_files_only=True)
    segs, _ = m.transcribe(x, language="ko", vad_filter=False)
    txt = " ".join(s.text.strip() for s in segs)
    a, b = JUNK.sub("", txt), JUNK.sub("", REF_TEXT)
    import difflib
    r = difflib.SequenceMatcher(None, b, a, autojunk=False).ratio()
    print("ASR 回读: %s" % txt[:200])
    print("ASR/ref_text 相似度 = %.3f" % r)

    # click
    pk = float(np.max(np.abs(x)))
    y = x * (0.70 / pk) if pk > 0 else x
    d = np.abs(np.diff(y))
    idx = np.where(d >= 0.30)[0]
    print("\nclick（峰值归一到 0.70，|Δ|>=0.30，16k）：%d 处" % len(idx))
    for i in idx[:20]:
        print("   t=%.3f  |Δ|=%.3f" % (i / sr, float(d[i])))

    # 句末粘连嗯（同 tailgrunt53e 判据）
    hop = int(0.005 * sr)
    n = len(x) // hop
    rms = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12)
                    for i in range(n)])
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > -40
    segs2 = []
    i = 0
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            segs2.append((i, j))
            i = j
        else:
            i += 1
    print("\n按 -40dB 分出 %d 个语音段" % len(segs2))
    tails = []
    for k in range(1, len(segs2)):
        pi, pj = segs2[k - 1]
        ci, cj = segs2[k]
        gap = (ci - pj) * 0.005
        if gap >= 0.03:
            continue
        ln = (cj - ci) * 0.005
        if not (0.005 <= ln <= 0.12):
            continue
        if (pj - pi) * 0.005 <= 0.10:
            continue
        s = x[ci * hop:cj * hop]
        if len(s) < 8:
            continue
        pk2 = float(np.max(np.abs(s)))
        pkdb = 20 * np.log10(pk2 + 1e-12)
        if not (-45 <= pkdb <= -20):
            continue
        zcr = float(np.mean(np.abs(np.diff(np.sign(s))) > 0))
        S = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        fr = np.fft.rfftfreq(len(s), 1.0 / sr)
        lowf = float(S[fr < 300].sum() / (S.sum() + 1e-12))
        if zcr < 0.15 and lowf > 0.45:
            tails.append((ci * 0.005, ln, pkdb, zcr, lowf))
    print("句末粘连型嗯：%d 个（%.2f/分钟）" % (len(tails), len(tails) / (dur / 60)))
    for t0, ln, pkdb, zcr, lowf in tails:
        print("   t=%.2f  %.0fms  峰值%.0fdB  zcr=%.3f  低频=%.2f"
              % (t0, ln * 1000, pkdb, zcr, lowf))


if __name__ == "__main__":
    main()
