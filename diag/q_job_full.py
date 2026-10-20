# -*- coding: utf-8 -*-
"""打印指定 job 的完整记录（含 chunks 文本）。用法: python3 q_job_full.py <id>"""
import json
import sys
import urllib.parse
import urllib.request
import http.cookiejar

BASE = "http://127.0.0.1:8000"
jid = sys.argv[1] if len(sys.argv) > 1 else "bf17cf796a40"
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
op.open(urllib.request.Request(
    BASE + "/api/login",
    data=urllib.parse.urlencode({"username": "admin", "password": "qwer1234"}).encode()
), timeout=30).read()
j = json.load(op.open(BASE + "/api/jobs?scope=all&q=202610190902", timeout=30))
items = j.get("items") or j.get("jobs") or []
it = items[0]
for k in ("id", "title", "voice", "speaker", "owner", "status", "language", "mode",
          "batch_size", "elapsed_seconds", "started_at", "created_at", "mp3", "wav",
          "files", "preview", "stats", "error"):
    if k in it:
        print("%s = %s" % (k, json.dumps(it[k], ensure_ascii=False)[:2000]))
print("--- chunks ---")
for c in (it.get("chunks") or []):
    print(json.dumps(c, ensure_ascii=False)[:1500])
