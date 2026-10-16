# -*- coding: utf-8 -*-
"""cerlat48.py —— 在**同一对素材**上切 4 种归一化组合，量各自的 CER。

只转写一次（最贵的一步），然后切开关重算 CER，所以四个数字严格可比。

组合：
  A 全关      （v1.1 及更早）
  B 只数字    （v1.2）
  C 只拉丁    （v1.3 新增）
  D 数字+拉丁 （v1.3 定案）

用法： python cerlat48.py --wav <音频> --txt <原文> [--chars N]
"""
import argparse
import difflib
import os
import re
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from app import asr_check as A  # noqa: E402

_LAT = re.compile(r"[A-Za-z]+(?:\s*&\s*[A-Za-z]+)*")
_DIG = re.compile(r"[0-9]")


def blocks(a, b, limit=12):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ra, rb = a[i1:i2], b[j1:j2]
        out.append((tag, ra, rb, i2 - i1 + j2 - j1))
    out.sort(key=lambda r: -r[3])
    return out[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--txt", default="")
    ap.add_argument("--job", default="")
    ap.add_argument("--chars", type=int, default=0)
    ap.add_argument("--lang", default="Korean")
    a = ap.parse_args()

    if a.job:
        import json
        import urllib.request
        r = json.loads(urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8000/api/jobs/" + a.job,
            headers={"Authorization": "Bearer " + os.environ.get(
                "TTS_DEBUG_TOKEN", "smoke-8f3a2c91")}), timeout=60).read())
        ref = r.get("text") or ""
    else:
        ref = open(a.txt, encoding="utf-8").read()
    if a.chars:
        ref = ref[:a.chars]
    print("原文 %d 字" % len(ref), flush=True)

    print("转写中...", flush=True)
    hyp = A.transcribe(a.wav, language=a.lang)
    if not hyp:
        print("❌ 转写失败")
        return 1
    print("ASR 假设 %d 字" % len(hyp), flush=True)
    print()

    res = {}
    for num in (False, True):
        for lat in (False, True):
            A._NUM_NORM = num
            A._LAT_NORM = lat
            res[(num, lat)] = A.cer(ref, hyp)

    print("=== CER 对照（同一对素材，只切开关；阈值 %.2f）===" % A.CER_THRESHOLD)
    for (num, lat), v in sorted(res.items()):
        tag = ("A 全关        " if not num and not lat else
               "B 只数字(v1.2)" if num and not lat else
               "C 只拉丁(v1.3)" if lat and not num else
               "D 数字+拉丁   ")
        print("  %s  CER = %.4f  %s"
              % (tag, v, "重掷" if v > A.CER_THRESHOLD else "通过"))
    print()
    print("  数字归一化增益 = %.4f" % (res[(False, False)] - res[(True, False)]))
    print("  拉丁归一化增益 = %.4f" % (res[(False, False)] - res[(False, True)]))
    print("  合计增益       = %.4f"
          % (res[(False, False)] - res[(True, True)]))

    A._NUM_NORM, A._LAT_NORM = True, True
    n_ref, n_hyp = A.normalize(ref), A.normalize(hyp)
    print()
    print("=== v1.3 定案口径下剩余最大差异块 ===")
    for tag, ra, rb, n in blocks(n_ref, n_hyp):
        print("  %-8s ref=«%s»  hyp=«%s»  (%d)"
              % (tag, ra[:50], rb[:50], n))
    print()
    print("  拉丁 token：ref=%s  hyp=%s"
          % (_LAT.findall(ref)[:10], _LAT.findall(hyp)[:10]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
