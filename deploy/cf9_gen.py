#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用员工真实文案(cf9d187252f8, 3881字, 员工跑在 cosyvoice3 上判"机器人")
跑一遍我们的最优链路 VoxCPM2 + fix_chain(字/秒锚 8.37)。"""
import json
import os
import time
import urllib.request

D = "/mnt/tts/diag/cf9"
os.makedirs(D, exist_ok=True)

d = json.load(open("/mnt/tts/data/jobs.json"))
v = d["cf9d187252f8"]
text = v["text"]
open(D + "/text.txt", "w").write(text)
print("chars=%d" % len(text), flush=True)

body = json.dumps({"text": text, "voice": "male", "rid": "cf9male"}).encode("utf-8")
req = urllib.request.Request("http://127.0.0.1:8400/synth", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
resp = urllib.request.urlopen(req, timeout=2400)
raw = resp.read()
open(D + "/vox_male_raw.wav", "wb").write(raw)
print("gen_seconds=%.1f bytes=%d hdr_chunks=%s hdr_sec=%s"
      % (time.time() - t0, len(raw), resp.headers.get("X-Chunks"),
         resp.headers.get("X-Audio-Seconds")), flush=True)
