# -*- coding: utf-8 -*-
"""临时注册一个「GS 能用」的女声参考音色（参考音频必须 3~10 秒）。

为什么需要它
------------
GS（GPT-SoVITS）硬性要求参考音频 **3~10 秒**；voices.json 里现役女声
`EL_female.wav` 是 15.08s → 走 /api/jobs 一律 500
「Reference audio is outside the 3-10 second range」。

`gpt-sovits/out/prompt_sola_00b6a45a.wav` 是现成的 10s 片段，
campplus 实测与 EL_female 余弦 **0.7953**（同人阈值 0.75）→ 同一个人，
可以安全地当 GS 侧的参考（换的不是说话人，只是片段长度）。

用法（zheng，不需要 sudo）：
    python3 /mnt/tts/diag/gs_voice_add.py add
    python3 /mnt/tts/diag/gs_voice_add.py del
    然后调 POST /api/voices/reload（脚本里会顺手 curl）
"""
import json
import os
import shutil
import subprocess
import sys

V = "/mnt/tts/data/voices/voices.json"
NAME = "SOLA_gs10"
REF = "/mnt/tts/projects/gpt-sovits/out/prompt_sola_00b6a45a.wav"
TXT = "/mnt/tts/projects/gpt-sovits/out/prompt_sola_00b6a45a.txt"
TOKEN = "smoke-8f3a2c91"


def reload_voices():
    try:
        out = subprocess.run(
            ["curl", "-s", "-X", "POST",
             "-H", "Authorization: Bearer " + TOKEN,
             "http://127.0.0.1:8000/api/voices/reload"],
            capture_output=True, text=True, timeout=30)
        print("reload ->", out.stdout.strip()[:300])
    except Exception as e:                     # noqa: BLE001
        print("reload 失败:", repr(e)[:160])


def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "add").strip()
    d = json.load(open(V, encoding="utf-8"))

    if mode == "add":
        if NAME in d:
            print("已存在，先删再建")
            del d[NAME]
        ref_text = open(TXT, encoding="utf-8").read().strip()
        d[NAME] = {
            "path": REF,
            "ref_text": ref_text,
            "sr": 24000,
            "dur": 10.0,
            "engine": "gptsovits",
            "gender": "female",
            "bench_case": "00b6a45ad6aa",
            "cps": 7.45,
            "cps_lock": 7.45,
            "clip": None,
            "origin": "diag-temp-47",
            "label": {"zh": "SOLA 10s（GS 对照用，临时）",
                      "en": "SOLA 10s (GS A/B, temp)"},
        }
    elif mode == "del":
        if NAME not in d:
            print("不存在，无需删")
            return 0
        del d[NAME]
    else:
        print("用法: gs_voice_add.py add|del")
        return 2

    shutil.copy2(V, V + ".bak-gs47")
    with open(V, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    print("%s 完成，当前音色数 %d，含 %s: %s"
          % (mode, len(d), NAME, NAME in d))
    reload_voices()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
