# -*- coding: utf-8 -*-
"""refswap_full53.py —— 生成长篇「EL 参考」版（男女各一篇），供 z25 出题。

优化点：**换参考音频**。
  现役参考（prompt_yohan_1a6c20a6.wav / prompt_sola_00b6a45a.wav）是
  gpt-sovits 时代的 EL 切片，F0 男 196.1 / 女 ~?。
  实测 omni 对参考的 F0 有「压缩」：yohan 参考 196.1 -> 输出 176.7（-10%）。
  改用**同案例 EL 分段**（seg0.wav 的 5~15s，F0 男 ~200）做参考后：
      男声 F0 176.7 -> 201.8（EL 200.7，完全命中）
      男声音色 0.9477 -> 0.9689
      男声 click 0.0287 -> 0.0238
  女声 F0/音色不变（266.3 / 0.9427），仅 click 0.0094 -> 0.0017。

🔴 参考音频硬约束（踩过的坑）：
  · ref_text 的**字数必须与音频时长配比**（10s 参考约 75~90 字，即 7.5~9 字/s）。
    首版按句边界向后扩展取到 174 字/10s = 17.4 字/s，输出只出 46s（语速崩），
    正是「ref_text 比音频长 -> 语速崩」的老坑。
  · 只向前找句号（起点对齐），**不向后扩展**（终点按字符数硬切）。
  · 生成后必须把 /voices 的参考**改回原值**，否则污染后续生成。
"""
import json
import os
import re
import subprocess
import time
import urllib.request

OMNI = "http://127.0.0.1:8300"
REF_BASE = "/mnt/tts/projects/gpt-sovits/out"
# (案例, voice, 现役参考 wav, 现役参考 txt)
JOBS = [
    ("d172b8c4efb3", "male", "prompt_yohan_1a6c20a6.wav",
     "prompt_yohan_1a6c20a6.txt"),
    ("00b6a45ad6aa", "female", "prompt_sola_00b6a45a.wav",
     "prompt_sola_00b6a45a.txt"),
]
START, CLIP = 5.0, 10.0


def dur_of(p):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1", p], capture_output=True, text=True
    ).stdout.strip().split("=")[1])


def case_text(case):
    d = "/mnt/tts/data/cases/%s" % case
    if os.path.exists(os.path.join(d, "full.txt")):
        return open(os.path.join(d, "full.txt"), encoding="utf-8").read().strip()
    return "\n\n".join(
        open(os.path.join(d, f), encoding="utf-8").read().strip()
        for f in sorted(os.listdir(d))
        if f.startswith("seg") and f.endswith(".txt"))


def set_voice(voice, wav, txt):
    body = json.dumps({"voice": voice, "wav": wav, "txt": txt}).encode()
    urllib.request.urlopen(urllib.request.Request(
        OMNI + "/voices", data=body,
        headers={"Content-Type": "application/json"}), timeout=60)


def synth(voice, text, rid):
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(
        OMNI + "/synth",
        data=json.dumps({"text": text, "voice": voice, "rid": rid}).encode(),
        headers={"Content-Type": "application/json"}), timeout=3600)
    raw = r.read()
    return raw, r.headers.get("x-audio-seconds"), time.time() - t0


def main():
    for case, voice, ref_wav, ref_txt in JOBS:
        d = "/mnt/tts/data/cases/%s" % case
        seg = os.path.join(d, "seg0.wav")
        seg_txt = os.path.join(d, "seg0.txt")
        sd = dur_of(seg)
        txt = open(seg_txt, encoding="utf-8").read().strip()
        ch = len(txt)
        a = int(START / sd * ch)
        b = int((START + CLIP) / sd * ch)
        # 起点向前找句号；终点按字符数硬切，绝不向后扩展
        s = txt.rfind(".", 0, a) + 1 or 0
        ref_text = txt[s:b].strip()
        rw = "/tmp/ref_%s_53.wav" % voice
        rt = "/tmp/ref_%s_53.txt" % voice
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", seg,
                        "-ss", str(START), "-t", str(CLIP),
                        "-ar", "24000", "-ac", "1", rw], check=True)
        open(rt, "w", encoding="utf-8").write(ref_text)
        print("%s 参考 %d 字/%.0fs = %.1f 字/s" %
              (voice, len(ref_text), CLIP, len(ref_text) / CLIP), flush=True)

        set_voice(voice, rw, rt)
        full = case_text(case)
        raw, dur, el = synth(voice, full, "refswapfull53_%s" % voice)
        dst = "/mnt/tts/diag/refswapfull53_%s.wav" % voice
        open(dst, "wb").write(raw)
        print("  %s -> %s  %ss  用时%.0fs" % (voice, dst, dur, el), flush=True)

        # 立刻恢复现役参考
        set_voice(voice, os.path.join(REF_BASE, ref_wav),
                  os.path.join(REF_BASE, ref_txt))
        print("  已恢复 %s 参考到 %s" % (voice, ref_wav), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
