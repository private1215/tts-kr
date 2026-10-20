# -*- coding: utf-8 -*-
"""z35 长语音重生：男(1a6c20a60481) + 女(00b6a45ad6aa) 全篇，走生产 API。

链路 = v3.5（本轮无音频链路改动：numcheck53 第五闸门只影响标注，不碰音频）。
重生成目的：z35 反馈轮用全新取样，顺带对冲单次取样波动。
"""
import os, sys, json, time, glob, re, subprocess
import urllib.request, urllib.parse, http.cookiejar

BASE = "http://127.0.0.1:8000"
USER = os.environ.get("TTS_USER", "admin")
PASS = os.environ.get("TTS_PASS", "qwer1234")
CASES = "/mnt/tts/data/cases"
OUT = "/mnt/tts/output"

cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

def login():
    d = urllib.parse.urlencode({"username": USER, "password": PASS}).encode()
    op.open(urllib.request.Request(BASE + "/api/login", data=d), timeout=30)

def post(path, form, timeout=180):
    d = urllib.parse.urlencode(form).encode()
    return json.loads(op.open(urllib.request.Request(BASE + path, data=d), timeout=timeout).read().decode())

def get(path, timeout=60):
    return json.loads(op.open(urllib.request.Request(BASE + path), timeout=timeout).read().decode())

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
            print("   poll err", repr(e)[:80]); time.sleep(poll); continue
        s = str(st.get("status", ""))
        if s in ("done", "completed", "success", "finished", "error", "failed"):
            return st
        time.sleep(poll)
    return {"status": "timeout"}

JOBS = [
    ("L3z35-male",   "1a6c20a60481", "EL_male",   "male"),
    ("L3z35-female", "00b6a45ad6aa", "EL_female", "female"),
]

def main():
    login()
    report = {}
    for title, cid, voice, gender in JOBS:
        txt = case_text(cid)
        n = len(txt.replace("\n", " "))
        print("[%s] submit chars=%d case=%s voice=%s" % (title, n, cid, voice), flush=True)
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
                d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                          "format=duration", "-of", "csv=p=0", f],
                                         capture_output=True, text=True).stdout.strip())
            except Exception:
                d = -1
            durs.append((f, d))
        report[title] = {"jid": jid, "status": st.get("status"), "chars": n,
                         "elapsed_s": round(el), "files": durs,
                         "num_warnings": st.get("num_warnings"),
                         "dir": st.get("dir") or st.get("out_dir")}
        print("   files=%s" % durs, flush=True)
        print("   num_warnings=%s" % (st.get("num_warnings"),), flush=True)
    json.dump(report, open("/mnt/tts/diag/gen_z35.json", "w"), ensure_ascii=False, indent=1)
    print("DONE", flush=True)

main()
