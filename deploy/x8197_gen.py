#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨文案验证音高塌陷：用案例 8197（有同文本 EL 音频）跑 VoxCPM2。

cf9（员工那条）已证明 F0 167.8 vs EL ~198。这里换一篇确认不是个案，
并且因为 8197 自带 EL 全篇，可以做 --el 逐窗对齐 + 和 EL 直接 A/B。
"""
import os
import time
import urllib.request

D = "/mnt/tts/diag/x8197"
os.makedirs(D, exist_ok=True)
text = open("/mnt/tts/data/cases/8197d9ac8a35/full.txt").read()
open(D + "/text.txt", "w").write(text)
print("chars=%d" % len(text), flush=True)

body = json_ = None
import json
body = json.dumps({"text": text, "voice": "male", "rid": "x8197"}).encode("utf-8")
req = urllib.request.Request("http://127.0.0.1:8400/synth", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
resp = urllib.request.urlopen(req, timeout=2400)
raw = resp.read()
open(D + "/vox_male_raw.wav", "wb").write(raw)
print("gen_seconds=%.1f bytes=%d chunks=%s sec=%s"
      % (time.time() - t0, len(raw), resp.headers.get("X-Chunks"),
         resp.headers.get("X-Audio-Seconds")), flush=True)
