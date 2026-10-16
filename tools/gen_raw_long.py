#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""裸生成一份长篇原始音频并落盘，供离线调参（第31轮）

为什么要落盘：fix_chain 的变调窗长需要 A/B（`--pitch-win 60/30/15`），
每试一次都重新跑一遍 GPU 太贵。裸输出只依赖模型，存一次就能反复跑后处理。

## 用法
    ./venv/bin/python tools/gen_raw_long.py [voice] [txt] [out.wav]
"""
import os
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
import soundfile as sf  # noqa: E402


def main():
    voice = sys.argv[1] if len(sys.argv) > 1 else "male"
    txt_p = (sys.argv[2] if len(sys.argv) > 2
             else "/mnt/tts/data/cases/6cdd25bc75f2/full.txt")
    out = (sys.argv[3] if len(sys.argv) > 3
           else "/mnt/tts/diag/raw_long_male.wav")
    if os.path.exists(out):
        print("已存在，跳过生成：%s" % out)
        return 0

    text = open(txt_p, encoding="utf-8").read().strip()
    print("voice=%s chars=%d" % (voice, len(text)), flush=True)

    import vox_client
    audio, sr, meta = vox_client.synth(text=text, voice=voice)
    if audio is None:
        print("合成失败:", meta)
        return 1
    sf.write(out, audio, sr)
    print("✅ %.1fs @ %d Hz -> %s" % (len(audio) / float(sr), sr, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
