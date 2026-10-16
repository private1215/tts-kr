# -*- coding: utf-8 -*-
"""第47轮：给 A/B 素材找**文本对齐**的切点。

为什么必须有这一步
------------------
三条音频是同一条文案，但**时长不同**：
    EL 原声      459.13s
    现役 omnivoice 458.83s   （与 EL 差 0.30s，几乎重合）
    GS 微调 SOLA  450.38s   （比 EL 快 8.75s）
按同一个绝对秒切，GS 的第 400s 和 EL 的第 400s 根本不是同一句话
→ 员工听到的是不同内容，比的就是「哪句好念」，正是用户反复纠正的
「逗人玩」陷阱。所以必须先用 ASR 找到**三方同一句话**的时刻。

做法：在 EL 上取候选时刻，按各自时长比例换算到另两条，
各切 15s 窗口 ASR，比对文字。选出三方一致的点当 A/B 切点。
"""
import difflib
import os
import subprocess

import numpy as np

SR = 16000
TMP = "/mnt/tts/diag/_align47"
os.makedirs(TMP, exist_ok=True)

FILES = [
    ("EL", "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav", 459.13),
    ("现役", "/mnt/tts/diag/sola47_f_omnivoice.wav", 458.83),
    ("GS", "/mnt/tts/diag/sola47_f_gptsovits.wav", 450.38),
]
# EL 上的候选时刻（秒）
CANDS = [12, 30, 55, 85, 115, 150, 190, 230, 275, 320, 365, 410, 445]
DUR = 15.0
M = None


def asr(y):
    global M
    if M is None:
        from faster_whisper import WhisperModel
        M = WhisperModel("medium", device="cpu", compute_type="int8",
                         local_files_only=True)
    s, _ = M.transcribe(y, language="ko", beam_size=5, vad_filter=False,
                        condition_on_previous_text=False)
    return " ".join(x.text.strip() for x in s)


def seg(src, tag, t0):
    o = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ss", "%.2f" % t0, "-t", "%.2f" % DUR,
                    "-ac", "1", "-ar", str(SR), o], check=True)
    import soundfile as sf
    y = sf.read(o, dtype="float32")[0]
    return ((y - y.mean()) / (np.abs(y).max() + 1e-9)).astype(np.float32)


def norm(s):
    return "".join(ch for ch in s if ch.isalnum())


def main():
    rows = []
    for t in CANDS:
        got = {}
        for nm, p, dur in FILES:
            scale = dur / FILES[0][2]
            tt = t * scale
            if tt + DUR > dur:
                got[nm] = None
                continue
            got[nm] = asr(seg(p, "%s_%d" % (nm, t), tt))
        if any(v is None for v in got.values()):
            continue
        a, b, c = (norm(got["EL"]), norm(got["现役"]), norm(got["GS"]))
        s1 = difflib.SequenceMatcher(None, a, b).ratio()
        s2 = difflib.SequenceMatcher(None, a, c).ratio()
        s3 = difflib.SequenceMatcher(None, b, c).ratio()
        rows.append((t, s1, s2, s3, got))
        print("t=%4ds  现役vsEL=%.3f  GSvsEL=%.3f  GSvs现役=%.3f"
              % (t, s1, s2, s3))
        print("     EL  : %s" % got["EL"][:110])
        print("     现役: %s" % got["现役"][:110])
        print("     GS  : %s" % got["GS"][:110])
        print()
    good = [r for r in rows if r[1] > 0.85 and r[2] > 0.85]
    print("=== 三方都对齐的切点（相似度 > 0.85）===")
    for t, s1, s2, s3, _ in good:
        print("  t=%4ds  现役=%.3f GS=%.3f" % (t, s1, s2))
    if not good:
        print("  （无）→ 需要放宽阈值或改用词级时间戳")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
