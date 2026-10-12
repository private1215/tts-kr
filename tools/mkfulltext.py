# -*- coding: utf-8 -*-
"""把某条案例的**全部片段文案**拼成一篇完整稿，供长文案对照生成。

背景（🔴 2026-10-12 用户纠正）：
    之前所有对照都只跑 60 秒片段，而真实业务是**单篇 3000 字左右、
    成片 3~8 分钟**。短片段上"好听"不代表长音频不出问题
    —— 长文本发散 / 吞字 / 韵律漂移，全都是在几分钟的尺度上才暴露的。
    所以对照必须用完整长文案。

用法：
    python tools/mkfulltext.py <case_id> [输出txt]
    # 默认写到 /mnt/tts/projects/gpt-sovits/out/full_<cid>.txt
"""
import json
import os
import sys

BASE = os.environ.get("TTS_BASE", "").strip() or "/mnt/tts"
CASES_JSON = os.path.join(BASE, "data", "cases.json")
CASES_DIR = os.path.join(BASE, "data", "cases")
OUT = "/mnt/tts/projects/gpt-sovits/out"


def _natkey(s):
    import re
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]


def main():
    cid = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT, "full_%s.txt" % cid)
    allc = json.load(open(CASES_JSON, encoding="utf-8"))
    meta = allc.get(cid)
    if not meta:
        raise SystemExit("案例 %s 不在 %s" % (cid, CASES_JSON))
    segs = sorted(meta.get("segments") or [],
               key=lambda s: _natkey(s.get("text") or ""))
    buf = []
    for s in segs:
        tp = os.path.join(CASES_DIR, cid, s.get("text") or "")
        if os.path.exists(tp):
            t = open(tp, encoding="utf-8").read().strip()
            if t:
                buf.append(t)
    txt = " ".join(buf)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(txt)
    print("写 %s  段数=%d  字数=%d" % (dst, len(buf), len(txt)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
