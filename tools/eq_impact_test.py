# -*- coding: utf-8 -*-
"""验证：0.953(裸推理) -> 0.844(线上成品) 是不是被「为 Qwen 拟合的 EQ」打下来的。

假设：TONE_EQ_FILTER_MALE/FEMALE 是 2026-10-06~08 用 **Qwen 的输出** 拟合出来的
（注释里全是 Qwen 时代的 r13/r18/r20/r25 口径）。10-12 我们把 EL_male/EL_female
切到了 **CosyVoice3**，EQ 却没跟着换。Cosy 裸推理本身对案例已经有 0.953，
再套一层为 Qwen 频谱设计的激进 EQ（-2~+7.6dB），等于把已经贴好的频谱推离案例。

验证：离线把 EQ 叠到裸推理音频上，看音色掉多少。
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = "/mnt/tts/projects/qwen3-tts"
sys.path.insert(0, os.path.join(ROOT, "tools"))
import spkchk  # noqa: E402

CAMPPLUS = os.environ.get(
    "TTS_CAMPPLUS", "/mnt/tts/models/Fun-CosyVoice3-0.5B/campplus.onnx")
_SESS = None


def sess():
    global _SESS
    if _SESS is None:
        import onnxruntime
        _SESS = onnxruntime.InferenceSession(
            CAMPPLUS, providers=["CPUExecutionProvider"])
    return _SESS


def wav16(path):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", tmp.name], check=True)
    return tmp.name


def emb(path):
    w = wav16(path)
    try:
        return spkchk.embed(sess(), spkchk.read16(w))
    finally:
        os.unlink(w)


def cos(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


EQ_M = ("superequalizer=1b=0.800:2b=1.524:3b=0.817:4b=1.138:5b=1.178:6b=1.059:"
        "7b=1.243:8b=1.081:9b=1.331:10b=1.193:11b=1.097:12b=0.951:13b=1.237:"
        "14b=1.048:15b=1.202:16b=1.000:17b=1.000:18b=1.000")
EQ_F = ("superequalizer=1b=2.410:2b=1.460:3b=0.931:4b=1.138:5b=0.907:6b=1.570:"
        "7b=1.501:8b=1.575:9b=1.791:10b=1.786:11b=1.467:12b=1.640:13b=1.354:"
        "14b=1.292:15b=1.357:16b=1.000:17b=1.000:18b=1.000")

JOBS = [
    ("男", "/mnt/tts/data/cases/1a6c20a60481/merged.wav",
     "/mnt/tts/projects/gpt-sovits/out/cosy_full_m.wav", EQ_M, 0.9532, 0.8444),
    ("女", "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav",
     "/mnt/tts/projects/gpt-sovits/out/cosy_full_f.wav", EQ_F, None, 0.8779),
]


def main():
    for g, ref, bare, eq, known_bare, known_online in JOBS:
        if not os.path.exists(bare):
            print("MISSING", bare)
            continue
        e_ref = emb(ref)
        b = emb(bare)
        out = "/tmp/eq_%s.wav" % g
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", bare,
                        "-af", eq, "-ar", "16000", "-ac", "1", out], check=True)
        e_eq = emb(out)
        print("\n=== %s ===" % g)
        print("  裸推理(未加EQ)      %.4f" % cos(e_ref, b))
        print("  裸推理 + 线上EQ     %.4f   ← 掉了 %+.4f"
              % (cos(e_ref, e_eq), cos(e_ref, e_eq) - cos(e_ref, b)))
        print("  （线上成品实测      %s）" % known_online)
        if known_bare:
            print("  （裸推理历史记录    %.4f）" % known_bare)


if __name__ == "__main__":
    main()
