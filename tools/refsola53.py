# -*- coding: utf-8 -*-
"""refsola53.py —— 女声 SOLA 参考版（对照用），同文本同长度。"""
import json
import os
import re
import time
import urllib.request

CASE = "/mnt/tts/data/cases/00b6a45ad6aa"
full = "\n\n".join(
    open(os.path.join(CASE, f), encoding="utf-8").read().strip()
    for f in sorted(os.listdir(CASE))
    if f.startswith("seg") and f.endswith(".txt"))
parts = re.split(r"(?<=[.!?。！？])\s+", full)
out = ""
for x in parts:
    if len(out) + len(x) + 1 > 785 and out:
        break
    out = (out + " " + x).strip()
t0 = time.time()
r = urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8300/synth",
    data=json.dumps({"text": out, "voice": "female",
                     "rid": "refsola53"}).encode(),
    headers={"Content-Type": "application/json"}), timeout=2400)
raw = r.read()
print("女声 SOLA 参考版 dur", r.headers.get("x-audio-seconds"),
      "time", round(time.time() - t0, 1), flush=True)
open("/mnt/tts/diag/refsola53_raw.wav", "wb").write(raw)
