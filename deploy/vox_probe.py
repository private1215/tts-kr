#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VoxCPM2 分块受控测试：逐块加大，定位「卡死/变慢」发生在哪个规模。

用法: vox_probe.py [max_chars_list...]   默认 600 1200 2400
"""
import json
import os
import sys
import time

import requests

URL = "http://127.0.0.1:8400"
OUT = "/mnt/tts/diag/vox_probe"
os.makedirs(OUT, exist_ok=True)


def text():
    jobs = json.load(open("/mnt/tts/data/jobs.json"))
    items = jobs if isinstance(jobs, list) else jobs.get("jobs", list(jobs.values()))
    j = [x for x in items if isinstance(x, dict) and x.get("id") == "cf9d187252f8"][0]
    return j["text"].replace("\n", " ")


def main():
    sizes = [int(x) for x in sys.argv[1:]] or [600, 1200, 2400]
    t = text()
    for n in sizes:
        seg = t[:n]
        tag = f"vox_{n}"
        print(f"\n=== {tag}: {len(seg)}字 ===", flush=True)
        t0 = time.time()
        try:
            r = requests.post(URL + "/synth",
                              json={"text": seg, "voice": "male", "rid": tag},
                              timeout=900)
        except Exception as e:
            print(f"  ✗ 请求异常 {e!r}", flush=True)
            break
        el = time.time() - t0
        if r.status_code != 200:
            print(f"  ✗ HTTP {r.status_code}: {r.text[:200]}", flush=True)
            break
        dst = os.path.join(OUT, tag + ".wav")
        open(dst, "wb").write(r.content)
        dur = float(r.headers.get("X-Audio-Seconds", 0))
        print(f"  ✓ {dur:.1f}s 音频, 用时 {el:.1f}s, RTF={el/max(dur,1e-6):.3f}, "
              f"块={r.headers.get('X-Chunks')} -> {dst}", flush=True)
        h = requests.get(URL + "/health", timeout=10).json()
        print(f"    显存 allocated={h['allocated_GiB']} reserved={h['reserved_GiB']}",
              flush=True)


if __name__ == "__main__":
    main()
