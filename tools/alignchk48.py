# -*- coding: utf-8 -*-
"""alignchk48.py —— 跨成品 A/B 出题前必做：**确认两个切片真的是同一句**。

🔴 铁律：跨成品 A/B 必须先按文本对齐。
若 A 的第 120s 和 B 的第 121.6s 说的不是同一句话，
员工听到的「差别」其实只是内容不同，整轮结论作废。

做法：两段各跑 ASR，取前 40 个字符比对（只留韩文/数字）。
"""
import difflib
import os
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
from asr_check import normalize, transcribe  # noqa: E402


def main():
    pairs = sys.argv[1:]
    if len(pairs) % 2:
        print("用法：alignchk48.py A1 B1 [A2 B2 ...]")
        return 1
    for i in range(0, len(pairs), 2):
        a, b = pairs[i], pairs[i + 1]
        ha = normalize(transcribe(a, language="Korean"))
        hb = normalize(transcribe(b, language="Korean"))
        sm = difflib.SequenceMatcher(None, ha[:120], hb[:120], autojunk=False)
        r = sm.ratio()
        tag = "✅ 同一句" if r >= 0.55 else "❌ 不同句！"
        print("\n=== %s vs %s  相似度 %.2f  %s" % (
            os.path.basename(a), os.path.basename(b), r, tag))
        print("  A: %s" % ha[:90])
        print("  B: %s" % hb[:90])
    return 0


if __name__ == "__main__":
    sys.exit(main())
