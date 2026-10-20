# -*- coding: utf-8 -*-
"""按标题查 job 记录（含音色、引擎、参数）。用法: python3 q_job.py <标题片段>"""
import json
import sys
import urllib.parse
import urllib.request
import http.cookiejar

BASE = "http://127.0.0.1:8000"
q = sys.argv[1] if len(sys.argv) > 1 else "202610190902"
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
data = urllib.parse.urlencode({"username": "admin", "password": "qwer1234"}).encode()
op.open(urllib.request.Request(BASE + "/api/login", data=data), timeout=30).read()
j = json.load(op.open(BASE + "/api/jobs?scope=all&q=%s" % urllib.parse.quote(q),
                      timeout=30))
items = j.get("items") or j.get("jobs") or []
print("命中 %d 条" % len(items))
for it in items:
    keep = ("id", "title", "status", "voice", "speaker", "engine", "gender",
            "text", "user", "created_at", "finished_at", "duration", "params",
            "emotion", "speed", "cps", "output", "out_dir", "owner")
    slim = {k: it[k] for k in keep if k in it}
    print(json.dumps(slim, ensure_ascii=False)[:3000])
    print("  全部字段:", sorted(it.keys()))
    break
