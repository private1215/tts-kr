# -*- coding: utf-8 -*-
"""gen_gs48f_eq.py —— 重新生成女声 GS 成品，验证 GS_EQ 定案参数是否端到端生效。

复用 job `2b920b3400c3`（gs48f_sola）的**同一份原文**（3415 字，case 00b6a45ad6aa），
voice=`SOLA_gs10`，只改 title。🔴 必须同文案才能和 EL 标杆
`cases/00b6a45ad6aa/merged.wav` 直接比音色。

用法： python gen_gs48f_eq.py [title]
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"
TOK = os.environ.get("TTS_DEBUG_TOKEN", "smoke-8f3a2c91")
SRC_JOB = "2b920b3400c3"
TITLE = sys.argv[1] if len(sys.argv) > 1 else "gs48f_eq"


def req(path, data=None):
    if data is None:
        r = urllib.request.Request(BASE + path,
                                   headers={"Authorization": "Bearer " + TOK})
    else:
        body = urllib.parse.urlencode(data).encode("utf-8")
        r = urllib.request.Request(
            BASE + path, data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "Authorization": "Bearer " + TOK})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())


def main():
    d = req("/api/jobs/" + SRC_JOB)
    text = d.get("text") or ""
    if not text:
        print("❌ 源 job 没有 text")
        return 1
    print("复用原文 %d 字，voice=SOLA_gs10，title=%s" % (len(text), TITLE),
          flush=True)
    r = req("/api/jobs", {"voice": "SOLA_gs10", "text": text, "mode": "clone",
                          "language": "Korean", "title": TITLE,
                          "emotion": "standard"})
    jid = r.get("id") or r.get("job", {}).get("id")
    print("job id =", jid, flush=True)
    for _ in range(240):
        time.sleep(5)
        d = req("/api/jobs/" + jid)
        if d.get("status") == "done":
            break
        if d.get("status") == "error":
            print("生成失败:", d.get("error"))
            return 1
    else:
        print("超时未完成")
        return 1

    st = d.get("stats") or {}
    files = d.get("files") or []
    chars = st.get("chars")
    secs = st.get("audio_seconds")
    print("engine=%s chars=%s audio_seconds=%s cps=%.3f"
          % (st.get("plan", {}).get("engine"), chars, secs,
             (chars / secs) if chars and secs else float("nan")), flush=True)
    for f in files:
        print("  file:", f.get("path"), f.get("seconds"), "cer=", f.get("cer"),
              flush=True)
    print("JOB_ID=%s" % jid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
