# -*- coding: utf-8 -*-
"""L3 长篇生成（v3.6 验收 + z36 盲听素材）。

同一个脚本跑两遍，靠 PREFIX 区分版本：
  PREFIX=L3z36   → v3.6（开了数字零填充）—— L3 验收 + z36 的 A 面
  PREFIX=L3v35c  → v3.5（切回 .bak-before-zerofill53）—— z36 的 B 面（对照组）
文案/音色/case/seed 两次完全一致，所以 A/B 是逐秒可对齐的同一句话。

服务端执行：
  cd /mnt/tts/projects/qwen3-tts && venv/bin/python /tmp/gen_z36_long.py
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import http.cookiejar

BASE = "http://127.0.0.1:8000"
USER = os.environ.get("TTS_USER", "admin")
PASS = os.environ.get("TTS_PASS", "qwer1234")
CASES = "/mnt/tts/data/cases"
OUT = "/mnt/tts/output"
PREFIX = os.environ.get("PREFIX", "L3z36")

cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))


def login():
    d = urllib.parse.urlencode({"username": USER, "password": PASS}).encode()
    op.open(urllib.request.Request(BASE + "/api/login", data=d), timeout=30)


def post(path, form, timeout=180):
    d = urllib.parse.urlencode(form).encode()
    return json.loads(op.open(urllib.request.Request(BASE + path, data=d),
                              timeout=timeout).read().decode())


def get(path, timeout=60):
    return json.loads(op.open(urllib.request.Request(BASE + path),
                              timeout=timeout).read().decode())


def case_text(cid):
    dd = os.path.join(CASES, cid)
    full = os.path.join(dd, "full.txt")
    if os.path.exists(full):
        return open(full, encoding="utf-8").read()
    segs = sorted(glob.glob(os.path.join(dd, "seg*.txt")),
                  key=lambda p: int(re.findall(r"seg(\d+)", p)[0]))
    return "\n".join(open(p, encoding="utf-8").read().strip() for p in segs)


def wait(jid, timeout=5400, poll=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            st = get("/api/jobs/" + jid)
        except Exception as e:
            print("   poll err", repr(e)[:80], flush=True)
            time.sleep(poll)
            continue
        s = str(st.get("status", ""))
        if s in ("done", "completed", "success", "finished", "error", "failed"):
            return st
        time.sleep(poll)
    return {"status": "timeout"}


JOBS = [
    ("male", "1a6c20a60481", "EL_male"),
    ("female", "00b6a45ad6aa", "EL_female"),
]


def main():
    login()
    report = {"prefix": PREFIX}
    for g, cid, voice in JOBS:
        title = "%s-%s" % (PREFIX, g)
        txt = case_text(cid)
        n = len(txt.replace("\n", " "))
        print("[%s] submit chars=%d case=%s voice=%s" % (title, n, cid, voice),
              flush=True)
        t0 = time.time()
        r = post("/api/jobs", {"title": title, "text": txt, "mode": "clone",
                               "voice": voice, "language": "Korean",
                               "speed": "1.0", "case_id": cid})
        jid = r.get("id") or r.get("job_id") or r.get("job")
        print("   job=%s" % jid, flush=True)
        st = wait(jid)
        el = time.time() - t0
        print("   status=%s elapsed=%.0fs" % (st.get("status"), el), flush=True)
        fs = sorted(glob.glob(os.path.join(OUT, title + "*/", "*.mp3")))
        durs = []
        for f in fs:
            try:
                d = float(subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries",
                     "format=duration", "-of", "csv=p=0", f],
                    capture_output=True, text=True).stdout.strip())
            except Exception:
                d = -1
            durs.append((f, d))
        report[title] = {"jid": jid, "status": st.get("status"), "chars": n,
                         "elapsed_s": round(el), "files": durs,
                         "num_warnings": st.get("num_warnings"),
                         "cer": (st.get("stats") or {}).get("cer")
                         if isinstance(st.get("stats"), dict) else None}
        print("   files=%s" % durs, flush=True)
        print("   num_warnings=%s" % (st.get("num_warnings"),), flush=True)
    json.dump(report, open("/mnt/tts/diag/gen_%s.json" % PREFIX, "w"),
              ensure_ascii=False, indent=1)
    print("DONE", flush=True)


main()
