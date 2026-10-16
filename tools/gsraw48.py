# -*- coding: utf-8 -*-
"""gsraw48.py —— 直连 gs-tts :8200 拿**裸输出**，分离「引擎本身」与「后处理链」。

用于回答：GS 男声 cps 7.45 是**引擎就这么慢**，还是**后处理链拉慢的**？
- 裸 cps ≈ 8.0 → 是后处理拉慢的 → 改链路即可（GS 已在 NEW_ENGINES 走快路径，
  那就要查快路径里的时长对齐目标）
- 裸 cps ≈ 7.45 → 是引擎本身 → 只能换更快的参考音频

用法：python3 gsraw48.py [--chars 1500]
"""
import json
import os
import sys

import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
import gs_client  # noqa: E402

TXT = open("/mnt/tts/diag/male_1500.txt", encoding="utf-8").read()
VJ = "/mnt/tts/data/voices/voices.json"
OUT = "/mnt/tts/diag/gsraw48"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=1500)
    ap.add_argument("--voice", default="yohan_gs_f8")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    v = json.load(open(VJ, encoding="utf-8"))[a.voice]
    txt = TXT[:a.chars]
    print("参考：%s  %.2fs  ref_text %d 字  隐含 %.2f 字/s"
          % (a.voice, v["dur"], len(v["ref_text"]),
             len(v["ref_text"]) / v["dur"]), flush=True)
    print("文本 %d 字，合成中 ..." % len(txt), flush=True)

    ok, payload = gs_client.synth(text=txt, ref_wav=v["path"],
                                  ref_text=v["ref_text"], spk="yohan",
                                  seg=50)
    if not ok:
        print("失败：", payload)
        return 1
    w = os.path.join(OUT, "%s_raw.wav" % a.voice)
    open(w, "wb").write(payload)
    dur = sf.info(w).duration
    print("\n=== GS 裸输出 ===")
    print("  时长 %.2fs   采样率 %dHz" % (dur, sf.info(w).samplerate))
    print("  **裸 cps = %.2f**" % (len(txt) / dur))
    return 0


if __name__ == "__main__":
    sys.exit(main())
