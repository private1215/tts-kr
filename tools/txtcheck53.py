# -*- coding: utf-8 -*-
import re
from collections import Counter

t = open("/mnt/tts/diag/female_full.txt", encoding="utf-8").read()
sents = [s.strip() for s in re.split(r"[.\n]", t) if len(s.strip()) > 10]
c = Counter(sents)
dups = {k: v for k, v in c.items() if v > 1}
print("sentence count:", len(sents), "| duplicated sentences:", len(dups))
for k, v in list(dups.items())[:5]:
    print("  x%d: %s" % (v, k[:50]))
ok = set(" .,?!~()[]:;-–—…·'\"")
weird = {ch: n for ch, n in Counter(t).items() if not (ch.isalnum() or ch in ok)}
print("weird chars:", dict(list(weird.items())[:10]))
print("total chars:", len(t))
print("head:", t[:80])
print("mid:", t[len(t) // 2 - 40:len(t) // 2 + 40])
print("tail:", t[-80:])
# 最长句
longest = max(sents, key=len) if sents else ""
print("longest sentence:", len(longest), longest[:60])
