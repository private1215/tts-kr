#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端冒烟：拿员工真实提交的文案，分别打 OmniVoice(8300) 与 VoxCPM2(8400)。
用法: test_services.py [job_id]
"""
import json
import os
import sys
import time

import requests

OUT = "/mnt/tts/diag/prod-20261014"
os.makedirs(OUT, exist_ok=True)


def job_text(jid=None):
    jobs = json.load(open("/mnt/tts/data/jobs.json"))
    items = jobs if isinstance(jobs, list) else jobs.get("jobs", list(jobs.values()))
    rows = [j for j in items if isinstance(j, dict) and j.get("text")]
    if jid:
        rows = [j for j in rows if j.get("id") == jid]
    rows.sort(key=lambda j: j.get("created_at") or 0)
    return rows[-1]


def call(port, text, voice, tag):
    url = f"http://127.0.0.1:{port}/synth"
    t = time.time()
    r = requests.post(url, json={"text": text, "voice": voice, "rid": tag},
                      timeout=3600)
    el = time.time() - t
    if r.status_code != 200:
        print(f"  ✗ {tag} HTTP {r.status_code}: {r.text[:300]}")
        return None
    dst = os.path.join(OUT, f"{tag}.wav")
    open(dst, "wb").write(r.content)
    dur = float(r.headers.get("X-Audio-Seconds", 0))
    print(f"  ✓ {tag}: {len(text)}字 -> {dur:.1f}s 音频, "
          f"{len(r.content)/1024/1024:.1f}MB, 用时 {el:.1f}s, "
          f"RTF={el/max(dur,1e-6):.3f}, 块={r.headers.get('X-Chunks','-')}")
    print(f"    -> {dst}")
    return dst


def main():
    jid = sys.argv[1] if len(sys.argv) > 1 else None
    j = job_text(jid)
    text = j["text"]
    print(f"任务 {j.get('id')} 引擎(旧)={j.get('engine')} 音色={j.get('voice')} "
          f"字数={len(text)}")
    print("\n=== OmniVoice 女声 (8300) ===")
    call(8300, text, "female", "omni_female")
    print("\n=== VoxCPM2 男声 (8400) ===")
    call(8400, text, "male", "vox_male")


if __name__ == "__main__":
    main()
