# -*- coding: utf-8 -*-
"""L3 全篇验收：生成全篇裸输出 + 跑各后处理组合 + 与 EL 同口径对比。

🔴 规范 S5：**L3（≥3~5 分钟全篇）是唯一质量验收**。
   片段只用来判断「改动有没有生效」，不能外推全篇（第47轮栽过两次）。

用法（服务器）：
    /mnt/tts/projects/qwen3-tts/venv/bin/python -u /tmp/l3_gen52.py
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

FIX_CHAIN = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"
CASE = "d172b8c4efb3"          # 男声案例（EL merged.wav 555s）
EL_WAV = "/mnt/tts/data/cases/%s/merged.wav" % CASE
OUT = "/mnt/tts/diag/l3_52"
TARGET_CPS = 8.37

EQ_MILD = ("highpass=f=110:poles=2,"
           "equalizer=f=90:t=q:w=2.0:g=3.0,"
           "equalizer=f=350:t=q:w=1.5:g=1.5,"
           "equalizer=f=1400:t=q:w=1.5:g=1.5,"
           "highshelf=f=4000:g=-2,lowpass=f=14000")

os.makedirs(OUT, exist_ok=True)


def full_text():
    p = "/mnt/tts/data/cases/%s/full.txt" % CASE
    return open(p, encoding="utf-8").read().strip()


def synth_raw(text, dst):
    body = json.dumps({"text": text, "voice": "male",
                       "rid": "l352"}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8400/synth", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=3600)
    raw = r.read()
    open(dst, "wb").write(raw)
    print("裸输出 %s  %.1fs  用时 %.0fs" %
          (dst, float(r.headers.get("x-audio-seconds", 0)), time.time() - t0),
          flush=True)


def post(in_wav, out_wav, chars, no_clean=False, no_adaptive=False, eq=None):
    cmd = [sys.executable, FIX_CHAIN, in_wav, out_wav,
           "--gender", "m", "--target-cps", str(TARGET_CPS),
           "--chars", str(chars)]
    if no_clean:
        cmd.append("--no-clean")
    if no_adaptive:
        cmd.append("--no-adaptive-hf")
    if eq:
        cmd += ["--eq", eq]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print("FAIL %s: %s" % (out_wav, r.stderr[-300:]), flush=True)
        return False
    print("后处理 %s  用时 %.0fs" % (out_wav, time.time() - t0), flush=True)
    return True


def main():
    txt = full_text()
    chars = len(txt)
    print("全篇 %d 字" % chars, flush=True)
    raw = os.path.join(OUT, "m_raw.wav")
    if not os.path.exists(raw):
        synth_raw(txt, raw)

    variants = [
        ("A_baseline", False, False, None),
        ("B_mild_eq", False, True, EQ_MILD),
        ("C_no_clean", True, False, None),
        ("D_mild_no_clean", True, True, EQ_MILD),
    ]
    report = {"chars": chars, "case": CASE, "rows": []}
    for name, nc, na, eq in variants:
        dst = os.path.join(OUT, "%s.wav" % name)
        ok = post(raw, dst, chars, nc, na, eq)
        report["rows"].append({"name": name, "ok": ok, "out": dst})
    json.dump(report, open(os.path.join(OUT, "l3_report.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nDONE", json.dumps(report, ensure_ascii=False)[:400], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
