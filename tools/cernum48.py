# -*- coding: utf-8 -*-
"""cernum48.py —— 数字归一化补丁的**客观化验证**（S2）。

同一对（原文 / 真实 ASR 假设文本），分别用开关关 / 开算 CER，
看假错消掉多少，并 dump 掉剩余的差异块确认「剩下的不是数字假错」。

用法：
  python3 cernum48.py --wav /tmp/m48_xxx.wav --txt /mnt/tts/diag/female_full.txt --chars 3415
"""
import argparse
import difflib
import os
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from app import asr_check as A  # noqa: E402


def dump_diff(a, b, limit=25):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        out.append((tag, a[i1:i2], b[j1:j2], i2 - i1 + j2 - j1))
    out.sort(key=lambda r: -r[3])
    return out[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--txt", required=True)
    ap.add_argument("--chars", type=int, default=3415)
    ap.add_argument("--lang", default="Korean")
    a = ap.parse_args()

    ref = open(a.txt, encoding="utf-8").read()[:a.chars]
    print("原文 %d 字（含空格）" % len(ref))

    print("转写中（whisper-small，453s 音频可能要 2-4 分钟）...", flush=True)
    hyp = A.transcribe(a.wav, language=a.lang)
    if not hyp:
        print("❌ 转写失败")
        return 1
    print("ASR 假设 %d 字" % len(hyp), flush=True)

    # 关 = 旧口径；开 = v1.2 新口径
    A._NUM_NORM = False
    old = A.cer(ref, hyp)
    A._NUM_NORM = True
    new = A.cer(ref, hyp)

    print()
    print("=== CER 对照（同一对素材，只切开关）===")
    print("  TTS_ASR_NUM_NORM=0（旧口径）  CER = %.4f" % old)
    print("  TTS_ASR_NUM_NORM=1（v1.2）    CER = %.4f" % new)
    print("  假错消掉 %.4f（阈值 %.2f）" % (old - new, A.CER_THRESHOLD))
    print("  重掷判定：旧口径 %s / 新口径 %s" % (
        "重掷" if old > A.CER_THRESHOLD else "通过",
        "重掷" if new > A.CER_THRESHOLD else "通过"))

    print()
    print("=== 新口径下剩余最大的差异块（确认不是数字假错）===")
    a2, b2 = A.normalize(ref), A.normalize(hyp)
    for tag, x, y, w in dump_diff(a2, b2):
        print("  %-8s 原文=%r ASR=%r" % (tag, x[:40], y[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
