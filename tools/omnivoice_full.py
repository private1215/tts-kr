# -*- coding: utf-8 -*-
"""OmniVoice 整篇候选生成：内部自带分段与 cross-fade 拼接，整篇直接提交。"""
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

BASE = "/mnt/tts/projects/omnivoice"
MODEL = os.path.join(BASE, "models/OmniVoice")
GSOUT = "/mnt/tts/projects/gpt-sovits/out"
CASES = "/mnt/tts/data/cases"
OUT = "/mnt/tts/diag/omnivoice-20261013"
os.makedirs(OUT, exist_ok=True)

JOBS = [
    ("O1_m", "d172b8c4efb3", "male",
     os.path.join(GSOUT, "prompt_yohan_1a6c20a6.wav"),
     os.path.join(GSOUT, "prompt_yohan_1a6c20a6.txt")),
    ("O2_f", "ce1a3fda551a", "female",
     os.path.join(GSOUT, "prompt_sola_00b6a45a.wav"),
     os.path.join(GSOUT, "prompt_sola_00b6a45a.txt")),
]


def case_text(cid):
    d = os.path.join(CASES, cid)
    segs = sorted([f for f in os.listdir(d)
                   if f.startswith("seg") and f.endswith(".txt")])
    return "\n\n".join(open(os.path.join(d, f), encoding="utf-8").read().strip()
                       for f in segs if open(os.path.join(d, f),
                                             encoding="utf-8").read().strip())


def main():
    from omnivoice import OmniVoice
    t0 = time.time()
    model = OmniVoice.from_pretrained(MODEL, device_map="cuda",
                                      dtype=torch.float16)
    print("加载 %.1fs" % (time.time() - t0), flush=True)
    report = []
    for name, cid, gender, rw, rt in JOBS:
        txt = case_text(cid)
        rtext = open(rt, encoding="utf-8").read().strip()
        print("\n==== %s %s %d字 ====" % (name, gender, len(txt)), flush=True)
        t1 = time.time()
        try:
            audio = model.generate(text=txt, ref_audio=rw, ref_text=rtext)
        except Exception as e:
            print("失败：%r" % (e,), flush=True)
            report.append({"name": name, "status": "failed", "err": repr(e)})
            continue
        el = time.time() - t1
        a = np.asarray(audio[0])
        dst = os.path.join(OUT, name + ".wav")
        sf.write(dst, a, 24000, subtype="PCM_16")
        print("-> %s %.1fs 段数=%d 用时 %.1fs"
              % (dst, len(a) / 24000.0, len(audio), el), flush=True)
        report.append({"name": name, "case": cid, "gender": gender,
                       "status": "ok", "chars": len(txt),
                       "segments": len(audio),
                       "audio_seconds": round(len(a) / 24000.0, 2),
                       "elapsed_s": round(el, 1),
                       "ref": os.path.basename(rw), "out": dst})
    json.dump(report, open(os.path.join(OUT, "omni_report.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nDONE", json.dumps(report, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
