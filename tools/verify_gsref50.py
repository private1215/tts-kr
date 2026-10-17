#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证第50轮补丁：员工默认音色（15 秒参考音频）走 GS 引擎不再 500。

背景：
    GS 只接受 3~10 秒的参考音频。而 BEST_CLONE = EL_female(15.08s) /
    EL_male(14.98s) 都是 15 秒 —— 打补丁前，用默认音色 + engine=gptsovits
    提交必然 HTTP 500，也就是「换 GS 上线」这件事根本做不到。

判据：
    - 打补丁前：EL_female + gptsovits -> 失败（GS 报推理失败/500）
    - 打补丁后：EL_female + gptsovits -> done，且日志里出现
      「GS 参考音频替换：EL_female.wav(15.08s) -> 音色 SOLA_gs10」

用法：
    python3 tools/verify_gsref50.py            # 男女各跑一条短稿（约 1~3 分钟）
    python3 tools/verify_gsref50.py --chars 120
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"
TOK = os.environ.get("TTS_DEBUG_TOKEN", "smoke-8f3a2c91")
CHARS = 200
for i, a in enumerate(sys.argv):
    if a == "--chars" and i + 1 < len(sys.argv):
        CHARS = int(sys.argv[i + 1])

CASES = {
    "EL_female": "52727ec43154",   # 女声案例原文
    "EL_male": "6cdd25bc75f2",     # 男声案例原文
}


def req(path, data=None, timeout=300):
    h = {"Authorization": "Bearer " + TOK}
    if data is None:
        r = urllib.request.Request(BASE + path, headers=h)
    else:
        body = urllib.parse.urlencode(data).encode("utf-8")
        r = urllib.request.Request(
            BASE + path, data=body, headers=dict(
                h, **{"Content-Type": "application/x-www-form-urlencoded"}))
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def run(voice, case):
    txt_path = "/mnt/tts/data/cases/%s/full.txt" % case
    if not os.path.exists(txt_path):
        print("  ❌ 缺原文 %s" % txt_path)
        return False
    text = open(txt_path, encoding="utf-8").read().strip()[:CHARS]
    print("\n=== voice=%s  engine=gptsovits  %d 字 ===" % (voice, len(text)))
    try:
        r = req("/api/jobs", {
            "voice": voice, "text": text, "mode": "clone",
            "language": "Korean", "title": "verify_gsref50_%s" % voice,
            "emotion": "standard",
            "engine": "gptsovits", "engine_force": "1",
        })
    except Exception as e:
        print("  ❌ 提交就失败: %r" % e)
        return False
    jid = r.get("id") or (r.get("job") or {}).get("id")
    print("  job id =", jid)
    for _ in range(90):
        time.sleep(5)
        d = req("/api/jobs/" + jid, timeout=60)
        st = d.get("status")
        if st == "done":
            f = (d.get("files") or [])
            print("  ✅ done  文件=%s  时长=%s" % (
                f, (d.get("stats") or {}).get("audio_seconds")))
            return True
        if st == "error":
            print("  ❌ error: %s" % str(d.get("error"))[:300])
            return False
    print("  ⏱ 超时")
    return False


def main():
    ok = True
    for voice, case in CASES.items():
        ok = run(voice, case) and ok
    print("\n==== 结论：%s ====" % (
        "全部通过（默认音色走 GS 不再 500）" if ok else "仍有失败，见上"))
    print("核对日志：grep 'GS 参考音频替换' "
          "/mnt/tts/projects/qwen3-tts/service.log | tail -5")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
