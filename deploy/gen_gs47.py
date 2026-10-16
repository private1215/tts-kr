#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_gs47.py —— 走 /api/jobs 生成一条成品并落盘（男女 / 多引擎通用）。

第47轮用途：GS 已加入快路径（NEW_ENGINES 含 gptsovits），需要重新生成
GS 成品与现役成品做**同链**对照。

用法：
  python3 gen_gs47.py --title gs47_m_gptsovits \
      --voice yohan_gs10 --engine gptsovits \
      --text-file /mnt/tts/data/cases/6cdd25bc75f2/seg0.txt --chars 1500
"""
import argparse
import glob
import os
import time

import requests

BASE = "http://127.0.0.1:8000"
USER = os.environ.get("TTS_USER", "admin")


def login():
    s = requests.Session()
    for pw in ("admin", "smoke-8f3a2c91", "123456"):
        r = s.post(f"{BASE}/api/login",
                   json={"username": USER, "password": pw})
        if r.status_code == 200:
            print(f"登录 OK（{USER}）")
            return s
    # 令牌也可能走 header
    s.headers.update({"Authorization": "Bearer smoke-8f3a2c91"})
    r = s.get(f"{BASE}/api/me")
    print(f"改用令牌，/api/me -> {r.status_code}")
    return s


def find_out_dir(title):
    """产物目录可能被 _unique_dir 加 `-N` 后缀：挑 mp3 最多且最新的。"""
    cands = sorted(glob.glob(os.path.join("/mnt/tts/output", title + "*")))
    best, best_key = None, None
    for d in cands:
        fs = glob.glob(os.path.join(d, "*.mp3"))
        if not fs:
            continue
        key = (len(fs), max(os.path.getmtime(f) for f in fs))
        if best_key is None or key > best_key:
            best, best_key = d, key
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--chars", type=int, default=1500)
    ap.add_argument("--wait", type=int, default=1800,
                    help="最长等待秒数")
    a = ap.parse_args()

    with open(a.text_file, encoding="utf-8") as f:
        text = f.read().strip()[:a.chars]
    print(f"文本 {len(text)} 字  <- {a.text_file}")

    s = login()
    r = s.post(f"{BASE}/api/jobs", data={
        "title": a.title, "text": text, "voice": a.voice,
        "engine": a.engine, "language": "Korean",
        "mode": "clone", "max_chars": 5000,
    })
    print("提交 ->", r.status_code, r.text[:300])
    if r.status_code != 200:
        return 1
    jid = r.json().get("id") or r.json().get("job_id")
    print("job id =", jid)

    t0 = time.time()
    while time.time() - t0 < a.wait:
        j = s.get(f"{BASE}/api/jobs/{jid}").json()
        st = j.get("status")
        prog = j.get("progress") or j.get("percent") or ""
        print(f"  [{int(time.time()-t0)}s] {st} {prog}", flush=True)
        if st in ("done", "finished", "completed", "success"):
            d = find_out_dir(a.title)
            print("✅ 成品目录:", d)
            for f in sorted(glob.glob(os.path.join(d or "", "*.mp3"))):
                print("   ", f)
            return 0
        if st in ("failed", "error"):
            print("❌ 失败:", str(j)[:600])
            return 1
        time.sleep(15)
    print("⏰ 超时")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
