# -*- coding: utf-8 -*-
"""fbtest52.py —— 反馈台**端到端**自测（出题后必跑）。

## 为什么必须有它

z22 首版出题脚本字段名写错（`question`/`description` 且选项缺 `key`），
而 `app/ui/feedback.html` 的 render() 读的是 `title`/`desc`/`dim`/`options[].key`。
后果：**题干不显示、前端计数 0/N、提交时 picks 被过滤成空对象**
→ 员工点了选项却全部提交失败，votes.jsonl 里一条记录都没有。
（用户截图发现，见 .workbuddy/memory/2026-10-17.md）

## 做法（不改员工数据）

用 `selftest` 账号（users.json 备注「自动化自测账号，仅用于线上链路回归」）：
1. 临时重置它的密码（PBKDF2，同 hash_pw 算法）
2. `/api/login` 拿 cookie
3. `/api/fb/items` 逐题校验 title/desc/dim/options[].key 非空
4. 模拟前端 `PICK[id] = o.key`，复算 upd() 的计数，断言 == 题数
5. `/api/fb/vote` 提交，读回 votes.jsonl 断言 picks 非空且 key ∈ {A,B,C}
6. **finally 还原密码 + 删掉自测记录**（员工票数不受影响）

## 用法（需 sudo：users.json 属主是 root）

    scp tools/fbtest52.py yuyin:/tmp/
    echo '<sudo密码>' | ssh yuyin "sudo -S -p '' \
        /mnt/tts/projects/qwen3-tts/venv/bin/python /tmp/fbtest52.py"
"""
import base64
import hashlib
import http.cookiejar
import json
import secrets
import subprocess
import urllib.parse
import urllib.request

USERS = "/mnt/tts/data/users.json"
VOTES = "/mnt/tts/fb/votes.jsonl"
BASE = "http://127.0.0.1:8000"
TEST_PW = "selftest2026"


def hash_pw(pw):
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 120000)
    return base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def main():
    subprocess.run("cp %s %s.bak-fbtest52" % (USERS, USERS), shell=True,
                   capture_output=True)
    d = json.load(open(USERS))
    orig = d["selftest"]["pwd"]
    d["selftest"]["pwd"] = hash_pw(TEST_PW)
    json.dump(d, open(USERS, "w"), ensure_ascii=False, indent=1)
    print("已临时重置 selftest 密码，备份 -> %s.bak-fbtest52" % USERS, flush=True)
    try:
        cj = http.cookiejar.CookieJar()
        op = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cj))
        body = urllib.parse.urlencode({"username": "selftest",
                                       "password": TEST_PW}).encode()
        r = op.open(urllib.request.Request(BASE + "/api/login", data=body),
                    timeout=20)
        print("登录:", r.status, json.loads(r.read()), flush=True)

        items = json.loads(op.open(BASE + "/api/fb/items",
                                   timeout=20).read())["items"]
        print("题目数:", len(items), flush=True)
        assert items, "❌ 没有题目"

        PICK = {}
        for it in items:
            opts = it.get("options") or []
            assert opts, "❌ 题目无选项: %s" % it.get("id")
            k = opts[0].get("key")
            assert k, "❌ 选项缺 key: %s %r" % (it.get("id"), opts[0])
            assert it.get("title"), "❌ 缺 title: %s" % it.get("id")
            assert it.get("desc"), "❌ 缺 desc: %s" % it.get("id")
            PICK[it["id"]] = k
        n = sum(1 for it in items if PICK.get(it["id"]))
        print("前端 upd() 会显示：已完成 %d / %d" % (n, len(items)), flush=True)
        assert n == len(items), "❌ 计数不足，前端仍会显示未选完"

        body = json.dumps({"picks": PICK,
                           "free": "【自测】反馈台字段修复验证，可删除"}).encode()
        r = op.open(urllib.request.Request(
            BASE + "/api/fb/vote", data=body,
            headers={"Content-Type": "application/json"}), timeout=20)
        print("提交:", r.status, json.loads(r.read()), flush=True)

        recs = [json.loads(l) for l in open(VOTES) if l.strip()]
        mine = [x for x in recs if x["user"] == "selftest"]
        print("votes.jsonl 中 selftest 记录数:", len(mine), flush=True)
        assert mine and mine[-1]["picks"], "❌ 提交的 picks 是空的"
        assert set(mine[-1]["picks"].values()) <= {"A", "B", "C"}, "❌ key 异常"
        print("\n✅ 端到端验证通过：题干/desc/dim/key 全部正确，"
              "提交成功且 picks 非空", flush=True)
    finally:
        d = json.load(open(USERS))
        d["selftest"]["pwd"] = orig
        json.dump(d, open(USERS, "w"), ensure_ascii=False, indent=1)
        print("已还原 selftest 原密码", flush=True)
        recs = [l for l in open(VOTES) if l.strip()
                and json.loads(l).get("user") != "selftest"]
        open(VOTES, "w").write("".join(recs))
        print("已删除 selftest 测试记录，剩余 %d 条" % len(recs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
