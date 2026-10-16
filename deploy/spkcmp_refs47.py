# -*- coding: utf-8 -*-
"""第47轮：确认 GS 可用的 10s 参考音频与 EL_female 是不是同一个人。

背景：GS（GPT-SoVITS）硬性要求参考音频 3~10 秒，而 voices.json 里
EL_female.wav 是 15.08s → 直接 500。
gpt-sovits/out/ 下有几个现成的 10s prompt，必须先用 campplus 确认
「10s prompt 和 EL_female 同一个人」，否则拿它当 GS 参考 = 换了个说话人，
对照就废了。

口径：tools/spkchk.embed()，campplus 16k，nwin=12，同人阈值 0.75。
"""
import itertools
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")

import numpy as np                      # noqa: E402
import onnxruntime                      # noqa: E402

from tools import spkchk                # noqa: E402

FILES = [
    ("EL_female.wav(15s)", "/mnt/tts/data/voices/EL_female.wav"),
    ("elara.wav", "/mnt/tts/data/voices/elara.wav"),
    ("SOLA_female.wav(15s)", "/mnt/tts/data/voices/SOLA_female.wav"),
    ("prompt_sola_00b6a45a.wav(10s)",
     "/mnt/tts/projects/gpt-sovits/out/prompt_sola_00b6a45a.wav"),
    ("prompt_f10s.wav(10s)",
     "/mnt/tts/projects/gpt-sovits/out/prompt_f10s.wav"),
    ("case_00b6a45ad6aa/merged.wav",
     "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"),
]


def main():
    sess = onnxruntime.InferenceSession(
        spkchk.CAMPPLUS, providers=["CPUExecutionProvider"])
    embs = {}
    for name, p in FILES:
        try:
            w = spkchk.read16(p)
            e = spkchk.embed(sess, w)
            if e is None:
                print("空", name)
                continue
            embs[name] = e
            print("ok  %-34s %6.2fs" % (name, w.shape[1] / float(spkchk.SR)))
        except Exception as ex:                    # noqa: BLE001
            print("fail %-34s %s" % (name, repr(ex)[:110]))
    print()
    names = list(embs)
    print("cosine（同人阈值 %.2f）" % spkchk.SAME_THR)
    for a, b in itertools.combinations(names, 2):
        c = float(np.dot(embs[a], embs[b]))
        flag = "同人" if c >= spkchk.SAME_THR else "不同人"
        print("  %-32s x %-32s = %.4f  %s" % (a, b, c, flag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
