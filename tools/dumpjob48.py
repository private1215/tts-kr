# -*- coding: utf-8 -*-
"""dumpjob48.py —— 把一个 job 的原文 / 成品路径落盘，供 cernum48.py 等复用。

用法： python dumpjob48.py <job_id> [out_txt]
输出：
  <out_txt>      原文（默认 /tmp/job_<id>.txt）
  并打印成品 mp3 的绝对路径
"""
import json
import os
import sys

JOBS = "/mnt/tts/data/jobs.json"
OUT = "/mnt/tts/output"


def main():
    if len(sys.argv) < 2:
        print("用法: dumpjob48.py <job_id> [out_txt]")
        return 1
    jid = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else "/tmp/job_%s.txt" % jid[:12]
    d = json.load(open(JOBS, encoding="utf-8"))
    v = d.get(jid) if isinstance(d, dict) else None
    if v is None and isinstance(d, dict):
        for k, x in d.items():
            if isinstance(x, dict) and (x.get("id") == jid
                                        or str(x.get("title")) == jid):
                v = x
                jid = x.get("id") or jid
                break
    if v is None:
        print("❌ 找不到 job", sys.argv[1])
        return 1
    t = v.get("text") or ""
    open(dst, "w", encoding="utf-8").write(t)
    print("TXT=%s (%d 字)" % (dst, len(t)))
    for f in (v.get("files") or []):
        p = os.path.join(OUT, f.get("path", ""))
        print("MP3=%s  exists=%s  seconds=%s"
              % (p, os.path.exists(p), f.get("seconds")))
    st = v.get("stats") or {}
    print("engine=%s chars=%s audio_seconds=%s"
          % (st.get("plan", {}).get("engine"), st.get("chars"),
             st.get("audio_seconds")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
