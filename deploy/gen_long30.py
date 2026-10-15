#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第30轮：走真实 API 生成一条 ≥5 分钟长篇，用于 L3 验收 + 出题素材。

用法： ./venv/bin/python deploy/gen_long30.py <voice> <txt> <tag>
与 deploy/smoke_p1.py 用同一套提交方式（form-urlencoded + Bearer token）。
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"
TOK = os.environ.get("TTS_DEBUG_TOKEN", "smoke-8f3a2c91")


def post(path, data):
    body = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Authorization": "Bearer " + TOK})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def get(path):
    req = urllib.request.Request(
        BASE + path, headers={"Authorization": "Bearer " + TOK})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def main():
    voice, txt, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    text = open(txt, encoding="utf-8").read().strip()
    print("voice=%s chars=%d bytes=%d tag=%s"
          % (voice, len(text), len(text.encode()), tag), flush=True)
    r = post("/api/jobs", {"voice": voice, "text": text,
                           "mode": "clone", "language": "Korean",
                           "title": tag, "emotion": "standard"})
    jid = r.get("id") or r.get("job", {}).get("id")
    print("jid =", jid, flush=True)
    if not jid:
        print(json.dumps(r)[:600])
        return 1
    t0 = time.time()
    last = ""
    while time.time() - t0 < 1800:
        j = get("/api/jobs/" + jid)
        st = j.get("status")
        if st != last:
            print("  %.0fs status=%s stage=%s" % (time.time() - t0, st,
                                                  j.get("stage")), flush=True)
            last = st
        if st == "done":
            print("✅ done in %.0fs" % (time.time() - t0), flush=True)
            print(json.dumps(j, ensure_ascii=False)[:4000], flush=True)
            return 0
        if st in ("error", "failed"):
            print("🔴 failed:", json.dumps(j, ensure_ascii=False)[:2000],
                  flush=True)
            return 1
        time.sleep(10)
    print("🔴 超时")
    return 1


if __name__ == "__main__":
    sys.exit(main())
