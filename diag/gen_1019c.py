# -*- coding: utf-8 -*-
"""gen_1019b.py —— 用**当前生产链**重跑男声 EL 同文本案例 d172b8c4efb3。

目的：文档 P0「男声接入清理链」的依据是**旧链路**（voxcpm 时代 + 老后处理）
的实测。现役男声已换成 omnivoice + 极简后处理，基线可能已经变了 ——
不看清楚就照搬旧方案，等于拿旧地图找新路。所以先走 /api/jobs 出一版
当前真实产物，再与 EL 原声（cases/d172b8c4efb3/merged.wav）同口径对比。

用法: cd /mnt/tts/projects/qwen3-tts && venv/bin/python -u /mnt/tts/diag/gen_1019b.py
"""
import glob
import os
import shutil
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from tools.eval5 import login, submit_job, wait_job  # noqa: E402

login()
TXT = "/mnt/tts/data/cases/d172b8c4efb3/full.txt"
OUT = "/mnt/tts/diag/mcur1019p.mp3"

text = open(TXT, encoding="utf-8").read().strip()
print("chars:", len(text), flush=True)

j = submit_job(text, "EL_male", "mcur1019p")
jid = j.get("id")
print("job:", jid, flush=True)
st = wait_job(jid, timeout=5400)
print("status:", st.get("status"), "| engine=", st.get("engine"),
      "| stats=", st.get("stats"), flush=True)

cands = []
for d in glob.glob("/mnt/tts/output/mcur1019p*"):
    cands += (glob.glob(os.path.join(d, "*.mp3"))
              + glob.glob(os.path.join(d, "*.wav")))
print("产物:", cands, flush=True)
if cands:
    src = cands[0]
    shutil.copy(src, OUT)
    print("copied %s -> %s" % (src, OUT), flush=True)
