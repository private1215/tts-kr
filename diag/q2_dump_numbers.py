# -*- coding: utf-8 -*-
"""q2_dump_numbers.py —— Q2 第一步：把原文与 ASR 的数字表达并排打出来。

目的：在写自动审计前，先看清这条 4400 字里到底有哪些数字、两种文本各自长什么样。
"""
import json
import re

j = json.load(open("/mnt/tts/data/jobs.json"))
jobs = j if isinstance(j, list) else j.get("jobs", j)
if isinstance(jobs, dict):
    jobs = list(jobs.values())
src = ""
for it in jobs:
    if isinstance(it, dict) and it.get("title") == "202610190902":
        src = it.get("text") or ""
        break

words = json.load(open("/mnt/tts/diag/v0902/words.json"))
if isinstance(words, dict):
    words = words.get("words", [])

asr_txt = "".join(w[2] for w in words)

# 汉字词数字串（含 쩜 小数点）
NUMKO = re.compile(r"[일이삼사오육칠팔구영공십백천만억조]{1,}")
NUMAR = re.compile(r"[0-9][0-9,\.]*")

print("=" * 70)
print("原文数字表达（含前后 12 字上下文）")
print("=" * 70)
for m in re.finditer(r"[\d일이삼사오육칠팔구영공십백천만억조쩜퍼센트%\.\,원주월일시분초년개명건배달러조억만점대여불무]+", src):
    s = m.group()
    if not re.search(r"[0-9일이삼사오육칠팔구영공십백천만억조]", s):
        continue
    a, b = max(0, m.start() - 14), min(len(src), m.end() + 14)
    print("  [%3d] %-22s | ...%s【%s】%s..." % (m.start(), s, src[a:m.start()], s, src[m.end():b]))

print()
print("=" * 70)
print("ASR 数字表达（含前后 12 字上下文）")
print("=" * 70)
for m in re.finditer(r"[\d일이삼사오육칠팔구영공십백천만억조쩜퍼센트%\.\,원주월일시분초년개명건배달러조억만점대여불무]+", asr_txt):
    s = m.group()
    if not re.search(r"[0-9일이삼사오육칠팔구영공십백천만억조]", s):
        continue
    a, b = max(0, m.start() - 14), min(len(asr_txt), m.end() + 14)
    print("  [%3d] %-22s | ...%s【%s】%s..." % (m.start(), s, asr_txt[a:m.start()], s, asr_txt[m.end():b]))

print()
print("原文 %d 字，ASR %d 字" % (len(src), len(asr_txt)))
