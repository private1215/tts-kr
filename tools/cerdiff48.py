# -*- coding: utf-8 -*-
"""cerdiff48.py —— dump CER 残余差异块并分类，判断「영문약어 假错」值不值得修。

## 为什么先量再改
`app/asr_check.py` 的 CER 只用来决定**要不要重掷**（阈值 0.35）。
女声全篇实测 CER 0.0527，离阈值差 6.6 倍 —— 就算把全部假错都消掉，
也只是 0.05 → 0.03，**一个重掷都省不下来**。
所以先 dump 出真实构成：如果 영문약어 只占 0.02，那这个优化就是纯自嗨
（记忆教训：z44 证明了「物理上量到的差异 ≠ 员工能听到的差异」）。

## 分类
- `latin`：原文侧是拉丁字母串（MLCC / AI / RD），ASR 侧是韩语字母名 → 英缩假错
- `digit`：两侧都含阿拉伯数字，只是写法不同（已由 v1.2 数字归一化部分处理）
- `hangul`：纯韩语差异 → 真错（念错/漏字/多字）
"""
import argparse
import difflib
import os
import re
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from app import asr_check as A  # noqa: E402

LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"[0-9]")


def blocks(a, b, limit=40):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ra, rb = a[i1:i2], b[j1:j2]
        if not ra and not rb:
            continue
        if LATIN.search(ra) or LATIN.search(rb):
            k = "latin"
        elif DIGIT.search(ra) or DIGIT.search(rb):
            k = "digit"
        else:
            k = "hangul"
        out.append((k, tag, ra, rb, i2 - i1 + j2 - j1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--txt", default="")
    ap.add_argument("--chars", type=int, default=0)
    ap.add_argument("--lang", default="Korean")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--job", default="", help="从 /api/jobs/<id> 取原文"
                                             "（优先于 --txt）")
    a = ap.parse_args()

    if a.job:
        import json
        import urllib.request
        r = json.loads(urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8000/api/jobs/" + a.job,
            headers={"Authorization": "Bearer " + os.environ.get(
                "TTS_DEBUG_TOKEN", "smoke-8f3a2c91")}), timeout=60).read())
        ref = r.get("text") or ""
        print("从 job %s 取原文 %d 字" % (a.job[:12], len(ref)), flush=True)
    else:
        ref = open(a.txt, encoding="utf-8").read()
    if a.chars:
        ref = ref[:a.chars]
    print("原文 %d 字（含空格）" % len(ref), flush=True)

    print("转写中（whisper-small，全篇约 2-4 分钟）...", flush=True)
    hyp = A.transcribe(a.wav, language=a.lang)
    if not hyp:
        print("❌ 转写失败")
        return 1
    print("ASR 假设 %d 字" % len(hyp), flush=True)
    print("CER = %.4f（阈值 %.2f，%s）"
          % (A.cer(ref, hyp), A.CER_THRESHOLD,
             "重掷" if A.cer(ref, hyp) > A.CER_THRESHOLD else "通过"),
          flush=True)

    bs = blocks(ref, hyp)
    for k in ("latin", "digit", "hangul"):
        sub = [x for x in bs if x[0] == k]
        cost = sum(x[4] for x in sub)
        print()
        print("=== %s：%d 块，代价 %d 字符（占原文 %.4f，阈值 %.2f）==="
              % (k, len(sub), cost, cost / max(len(ref), 1),
                 A.CER_THRESHOLD))
        for _, tag, ra, rb, n in sorted(sub, key=lambda r: -r[4])[:a.limit]:
            print("  [%s] ref=«%s»  hyp=«%s»  (%d)"
                  % (tag, ra[:60], rb[:60], n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
