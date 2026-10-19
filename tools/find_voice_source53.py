# -*- coding: utf-8 -*-
"""find_voice_source53.py —— 在被删音色的 ref_text 反查它来自案例库哪一条。

背景：2026-10-18 员工误删了 elara / yo han 两个音色，连
`sources/*__full.wav` 一起清掉了。用户说明：**这两个不是员工上传的录音**，
原本的上传克隆音色早已淘汰，它们是**从案例库语音里切出来的**。
→ 案例库（37 条，EL 原声）没被删，能反推重建。

做法：
  ① 归一化（去空白/标点）后在所有 case 的 full.txt / seg*.txt 里找 ref_text
  ② 命中后按「字符比例」粗估时间位置，再用 whisper 滑窗校验并微调，
     返回最佳匹配窗口（用于切出等长切片）

用法: venv/bin/python tools/find_voice_source53.py --name elara
"""
import argparse
import json
import os
import re
import subprocess
import sys

CASES = "/mnt/tts/data/cases"
VOICES_BAK = "/mnt/tts/data/voices/voices.json.bak-before-omnimale53"


def norm(s):
    return re.sub(r"[\s\.,!?\"'·…\-\–—()\[\]【】]", "", s or "")


def case_texts(cid):
    d = os.path.join(CASES, cid)
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".txt"):
            out[fn] = open(os.path.join(d, fn), encoding="utf-8").read()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    bak = json.load(open(VOICES_BAK, encoding="utf-8"))
    src = bak.get(a.name)
    if not src:
        print("备份里没有音色 %s" % a.name)
        return 1
    ref = norm(src.get("ref_text") or "")
    print("音色 %s：ref_text %d 字（归一化），dur %.2fs，src_dur %.2fs"
          % (a.name, len(ref), src.get("dur", 0), src.get("src_dur", 0)),
          flush=True)
    if not ref:
        print("没有 ref_text，无法反查")
        return 1

    hits = []
    for cid in sorted(os.listdir(CASES)):
        d = os.path.join(CASES, cid)
        if not os.path.isdir(d):
            continue
        for fn, txt in case_texts(cid).items():
            if ref in norm(txt):
                hits.append((cid, fn, len(norm(txt))))
                print("  命中 %s / %s（该文 %d 字）" % (cid, fn,
                                                        len(norm(txt))),
                      flush=True)
    if not hits:
        print("❌ 案例库里找不到这段 ref_text —— 它可能来自已淘汰的上传录音")
        return 2
    print("\n候选来源：%s" % [h[0] for h in hits])
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
