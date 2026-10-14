#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用**当前生产链路**跑一条女声成品，用于跟 EL 女声基线比 σ。

为什么选案例 52727ec43154 的原文：
    第四轮盲听里，员工对 5272 这篇的评价是「OmniVoice + 线上全都不像」，
    这是女声最难看的一次成绩。拿它当基准，量出来是什么水平就是什么水平。

用法：
    python gen_female_prod.py [字数] [voice]
默认 1500 字 / EL_female（OmniVoice）。
"""
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"
TOK = os.environ.get("TTS_DEBUG_TOKEN", "smoke-8f3a2c91")
CASE = os.environ.get("CASE", "52727ec43154")
CASE_TXT = "/mnt/tts/data/cases/%s/full.txt" % CASE
CHARS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
VOICE = sys.argv[2] if len(sys.argv) > 2 else "EL_female"
OUT_DIR = "/mnt/tts/diag/xf5272"


def req(path, data=None):
    if data is None:
        r = urllib.request.Request(BASE + path,
                                   headers={"Authorization": "Bearer " + TOK})
    else:
        body = urllib.parse.urlencode(data).encode("utf-8")
        r = urllib.request.Request(
            BASE + path, data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "Authorization": "Bearer " + TOK})
    return json.loads(urllib.request.urlopen(r, timeout=300).read())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    text = open(CASE_TXT, encoding="utf-8").read().strip()[:CHARS]
    print("提交 %d 字（EL 女声案例 %s 原文）voice=%s"
          % (len(text), CASE, VOICE), flush=True)
    r = req("/api/jobs", {"voice": VOICE, "text": text, "mode": "clone",
                          "language": "Korean", "title": "xf5272-prod",
                          "emotion": "standard"})
    jid = r.get("id") or r.get("job", {}).get("id")
    print("job id =", jid, flush=True)
    d = None
    for _ in range(120):
        time.sleep(5)
        d = req("/api/jobs/" + jid)
        if d.get("status") == "done":
            break
        if d.get("status") == "error":
            print("生成失败:", d.get("error"))
            return 1
    else:
        print("超时未完成")
        return 1

    st = d.get("stats") or {}
    files = d.get("files") or []
    print("stats: engine=%s cps=%s chunks=%s"
          % (st.get("plan", {}).get("engine"), st.get("cps"),
             st.get("chunks")), flush=True)
    if not files:
        print("无成品")
        return 1
    rel = files[0].get("path") or files[0].get("name")
    mp3 = os.path.join(OUT_DIR, "prod_%s_%s.mp3" % (VOICE, CASE))
    with urllib.request.urlopen(
            urllib.request.Request(
                "%s/api/download/%s" % (BASE, rel.lstrip("/")),
                headers={"Authorization": "Bearer " + TOK}),
            timeout=600) as resp:
        open(mp3, "wb").write(resp.read())
    wav = mp3.replace(".mp3", ".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp3,
                    "-ar", "24000", "-ac", "1", wav], check=True)
    print("下载 -> %s\n转 wav -> %s" % (mp3, wav), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
