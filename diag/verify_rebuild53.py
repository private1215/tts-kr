# -*- coding: utf-8 -*-
"""verify_rebuild53.py —— 校验「从案例库反推的 elara / yo han 切片窗口」对不对。

## 背景
2026-10-18 员工误删 elara / yo han 两个音色（连 sources/*__full.wav 一起）。
用户纠正：**它们不是员工上传的录音**，是从案例库语音里切出来的。

## 已锁定的来源（双证据）
| 音色 | 源 | 证据 |
|---|---|---|
| elara | `a3d39a095bac/seg0.wav` | ① 时长 **173.087s** = 备份 src_dur 173.09 精确吻合 ② 独有词 `겉모습만` 全库唯一命中 |
| yo han | `799e4d7060ce/seg0.wav` | ① 时长 **261.878s** = 备份 src_dur 261.88 精确吻合 ② ref_text 命中 full.txt/seg1.txt |

## 这一步干什么
在备份记录的 clip 时间点（elara 14.36~22.50 / yo han 136.02~144.68）
切窗转写，与备份 ref_text 比对相似度，确认窗口没错位。

🔴 顺带解决一个大坑：yo han 备份里的 ref_text 是 **191 字 / 8.66s = 22 字每秒**，
物理上念不出来（GS 硬坑 3：ref_text 比音频长 → 语速崩 + 把参考文本当正文反复念）。
→ **重建时不照抄旧 ref_text，用窗内 ASR 结果重新生成对齐的文本**。

用法: venv/bin/python -u /mnt/tts/diag/verify_rebuild53.py
"""
import difflib
import re
import subprocess
import sys

import faster_whisper

CASES = "/mnt/tts/data/cases"
# (音色, 源相对路径, start, end, 备份里记录的 ref_text 前段)
JOBS = [
    ("elara",
     "a3d39a095bac/seg0.wav", 14.36, 22.50,
     "위아래로 마구 흔들면서 따라붙는 개미와 차익을 내려는 개미를 동시에 "
     "털어내는 과정이 나올 수밖에 없어요. 주가가 저항선을 뚫는 겉모습만"),
    ("yo han",
     "799e4d7060ce/seg0.wav", 136.02, 144.68,
     "바로 전날인 구월 삼십일에는, 개미들 이만삼천주 털렸고, 외인이 "
     "이천팔백주, 기관이 이만주 싹 쓸어 담았거든요?"),
]


def norm(s):
    return re.sub(r"[\s\.,!?\"'·…\-\–—()\[\]【】]", "", s or "")


def main():
    w = faster_whisper.WhisperModel("small", device="cuda",
                                    compute_type="float16")
    for name, rel, s, e, expect in JOBS:
        src = "%s/%s" % (CASES, rel)
        out = "/tmp/vr_%s.wav" % name.replace(" ", "")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                        "-ss", str(s), "-t", str(e - s),
                        "-ar", "16000", "-ac", "1", out], check=True)
        segs, _ = w.transcribe(out, language="ko", beam_size=5)
        txt = " ".join(x.text for x in segs).strip()
        dur = e - s
        sc = difflib.SequenceMatcher(None, norm(expect), norm(txt)).ratio()
        print("== %s ==", flush=True)
        print("   源      : %s  %.2fs ~ %.2fs (%.2fs)" % (rel, s, e, dur))
        print("   转写    : %s" % txt, flush=True)
        print("   期望    : %s" % expect)
        print("   相似度  : %.3f" % sc)
        print("   转写 %d 字 → %.2f 字每秒" % (len(txt), len(txt) / dur))
        print("   ✅ 窗口正确" if sc >= 0.55 else "   ❌ 窗口可能错位", flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
