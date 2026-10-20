#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""shorthead53.py —— 短首句实验：#7 型崩坏能不能安全切除？

## 要回答的问题
headfix 不切 #7 型（起音抛多组碎片，占产出 9%）的唯一理由：
**短首句**（如「네. 반갑습니다…）本身就是多音节 + 句间停顿 ≥250ms，
看起来也是「多组 + 长间隙」，会被整段吃掉。

→ 那就**直接生成一批短首句**，量它们的头型。如果短首句的头型和 #7 明显不同，
#7 就可以切；如果一样，就不能切。

## 口径
- 走生产 API（不是 CLI 裸推理），音色 EL_female / EL_male
- 头型算法 = headfix 自己那套（headtype53.head_shape）
- 再让 ASR 听头部区域：听到真词 = 不能切；听到垃圾 = 可切

用法：venv/bin/python tools/shorthead53.py
"""
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.parse
import http.cookiejar

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")
sys.path.insert(0, "/mnt/tts/diag")
from headtype53 import head_shape  # noqa: E402
from asr_check import normalize, transcribe  # noqa: E402

BASE = "http://127.0.0.1:8000"
OUT = "/mnt/tts/output"

# 短首句样本：刻意选「多音节 + 句间有停顿」的，这是最容易被误判成 #7 的一类
TEXTS = [
    ("네. 반갑습니다.", "네+반갑"),
    ("안녕하세요. 반갑습니다.", "안녕+반갑"),
    ("네. 안녕하세요.", "네+안녕"),
    ("반갑습니다. 오늘도 함께하겠습니다.", "반갑+오늘"),
    ("오늘은 삼성전자에 대해 말씀드리겠습니다.", "오늘은 삼성"),
]

cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))


def login():
    d = urllib.parse.urlencode({"username": "admin", "password": "qwer1234"}).encode()
    op.open(urllib.request.Request(BASE + "/api/login", data=d), timeout=30)


def post(path, form, timeout=180):
    d = urllib.parse.urlencode(form).encode()
    return json.loads(op.open(urllib.request.Request(BASE + path, data=d),
                              timeout=timeout).read().decode())


def get(path, timeout=60):
    return json.loads(op.open(urllib.request.Request(BASE + path),
                              timeout=timeout).read().decode())


def wait(jid, timeout=600, poll=8):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            st = get("/api/jobs/" + jid)
        except Exception:
            time.sleep(poll); continue
        if st.get("status") in ("done", "completed", "success", "finished",
                                "error", "failed"):
            return st
        time.sleep(poll)
    return {"status": "timeout"}


def head_end_of(y, sr):
    """与 head_shape **完全同算法**（含 3s 窗、-30dB 阈值）。

    🔴 2026-10-20 修死循环：旧版只回退静音段，遇到「S 前一帧就是浊音」
    时 j==i → i=j 原地不动 → 无限循环（实测卡住 5 分钟不退）。
    正确做法照 headtype53：浊音段整段跳过，只对静音段判 >=150ms。
    """
    hop = int(0.01 * sr)
    n = min(len(y) // hop, 300)
    db = 20 * np.log10(np.array(
        [np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2) + 1e-12)
         for i in range(n)]) + 1e-12)
    v = db > -30.0
    si = None
    for k in range(len(db) - 30):
        if db[k] > -25 and db[k:k + 30].mean() > -28:
            si = k
            break
    if si is None:
        return None, None
    i = si - 1
    head_end = None
    while i >= 0:
        if v[i]:
            j = i
            while j >= 0 and v[j]:
                j -= 1
            i = j
        else:
            j = i
            while j >= 0 and not v[j]:
                j -= 1
            if (i - j) >= 15:            # >=150ms
                head_end = j + 1
                break
            i = j
    if head_end is None or head_end <= 0:
        return si * 0.01, 0.0
    return si * 0.01, head_end * 0.01


def main():
    login()
    rows = []
    for i, (text, tag) in enumerate(TEXTS):
        for voice in ("EL_female",):
            title = "sh53_%s_%s" % (tag, voice)
            print("[%d/%d] %s  %s" % (i + 1, len(TEXTS), tag, voice), flush=True)
            r = post("/api/jobs", {"title": title, "text": text, "mode": "clone",
                                   "voice": voice, "language": "Korean",
                                   "speed": "1.0"})
            jid = r.get("id") or r.get("job_id") or r.get("job")
            st = wait(jid)
            fs = glob.glob(os.path.join(OUT, title + "*", "*.mp3"))
            if not fs:
                print("   ❌ 无产物 status=%s" % st.get("status")); continue
            mp3 = sorted(fs)[0]
            wav = "/tmp/sh53.wav"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", mp3,
                            "-ac", "1", "-ar", "16000", wav], check=True)
            y, sr = sf.read(wav)
            kind, fdur, fgap, ng = head_shape(y, sr)
            S, he = head_end_of(y, sr)
            if S is None:
                print("   ❌ 找不到起音点"); continue
            # 头部区域组数（与 head_shape 同口径：3s 窗内 >-30dB）
            hop = int(0.01 * sr)
            n = min(len(y) // hop, 300)
            db = 20 * np.log10(np.array(
                [np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2) + 1e-12)
                 for i in range(n)]) + 1e-12)
            vv = db > -30.0
            hi = int(he * 100)
            gs, k = [], 0
            while k < hi:
                if vv[k]:
                    m = k
                    while m < hi and vv[m]:
                        m += 1
                    gs.append((k * 0.01, (m - k) * 0.01)); k = m
                else:
                    k += 1
            speech = sum(b - a for a, b in gs)
            hwav = "/tmp/sh53_head.wav"
            sf.write(hwav, y[:int(he * sr)], sr)
            try:
                txt = normalize(transcribe(hwav, language="Korean"))
            except Exception as e:
                txt = "(失败 %r)" % repr(e)[:30]
            print("   文案=%r" % text)
            print("   kind=%-12s S=%.2f 头区=%.2f 组数=%d 头区语音=%.2fs ASR=%r"
                  % (kind, S or -1, he, len(gs), speech, txt[:40]))
            print("   组: %s" % " ".join("%.2f-%.2f(%.2fs)" % (a, b, b - a)
                                          for a, b in gs[:8]))
            rows.append({"tag": tag, "text": text, "kind": kind, "S": S,
                         "head": he, "ngroups": len(gs), "speech": speech,
                         "asr": txt})
            sys.stdout.flush()
    json.dump(rows, open("/mnt/tts/diag/shorthead53.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n=== 汇总 ===")
    for r in rows:
        print("  %-12s kind=%-12s 组数=%d 头区语音=%.2fs ASR=%r"
              % (r["tag"], r["kind"], r["ngroups"], r["speech"], r["asr"][:30]))


if __name__ == "__main__":
    main()
