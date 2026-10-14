#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给两位试听同事各开一个独立账号。

🔴 为什么必须分开：votes.jsonl 里 25 条票 **全部 user="admin"**，
   两个人如果都登 admin，我拿到的是混在一起的一坨，既算不出
   「两人是否一致」（重测信度），也没法发现「某一个人其实在敷衍」。
   分开账号后，同一份题库能给出 2 个独立样本点，还能互相校验。
"""
import json
import os
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"
TOK = os.environ.get("TTS_DEBUG_TOKEN", "smoke-8f3a2c91")
NEW = [("listener1", "ting1-2026"), ("listener2", "ting2-2026")]


def post(path, data):
    body = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Authorization": "Bearer " + TOK})
    try:
        return 200, json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "ignore")[:200]
    except Exception as e:
        return 0, repr(e)[:200]


if __name__ == "__main__":
    for name, pw in NEW:
        code, r = post("/api/users", {"username": name, "password": pw,
                                      "role": "user"})
        print("%-12s -> HTTP %s  %s" % (name, code, r))
    code, r = post("/api/users", {})
    users = json.loads(urllib.request.urlopen(
        urllib.request.Request(BASE + "/api/users",
                               headers={"Authorization": "Bearer " + TOK}),
        timeout=30).read())
    print("\n现有账号:")
    for u in users:
        print("  %-12s role=%s must_change=%s" % (u.get("username"),
                                                   u.get("role"),
                                                   u.get("must_change")))
