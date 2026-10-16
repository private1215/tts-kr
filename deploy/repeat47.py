# -*- coding: utf-8 -*-
"""第47轮：验证员工说的「第三题 11 秒重复说话」。

## 背景
员工 z45 第 3 题写「11 秒 重复说话和杂音」。第 3 题的 ② = GS 微调段
（素材 z45_b_s3，从 GS 全篇 313.90s 起切 20s）。
**重复念字正是 GPT-SoVITS 自回归解码的已知硬伤**，而且客观可测 ——
本项目判定阈值就是「冗余率 > 0.30 = 重复念」（engine_score_all.py RED_MAX）。
全篇层面 GS 的 cov=0.928 是过了的，但**局部**重复不影响全篇指标，
所以必须单独查这一段。

## 做法
1. 对员工实际听到的两段（A=现役 319.79s 起 / B=GS 313.90s 起）做 whisper ASR。
2. 逐句打印，肉眼 + 程序一起找重复片段。
3. 程序化检测：找长度 >= 4 音节的**连续重复子串**（同一串在文本里出现 >= 2 次
   且两次出现间隔 < 6 秒）。
4. 顺带给出该段的时长/语速，判断「重复」是不是因为语速失控。
"""
import os
import subprocess
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")

import soundfile as sf                                    # noqa: E402
from faster_whisper import WhisperModel                   # noqa: E402

CUR = "/mnt/tts/diag/sola47_f_omnivoice.wav"     # 458.83
GS = "/mnt/tts/diag/sola47_f_gptsovits.wav"      # 450.38
DUR = 20.0
M = None


def asr(src, t0, dur, tag):
    global M
    if M is None:
        M = WhisperModel("medium", device="cpu", compute_type="int8",
                         local_files_only=True)
    o = "/mnt/tts/diag/_rp_%s.wav" % tag
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ss", "%.2f" % t0, "-t", "%.1f" % dur,
                    "-ac", "1", "-ar", "16000", o], check=True)
    segs, _ = M.transcribe(sf.read(o, dtype="float32")[0], language="ko",
                           beam_size=5, vad_filter=False,
                           condition_on_previous_text=False)
    return [(float(s.start), float(s.end), s.text.strip()) for s in segs]


def find_repeat(items):
    """找连续重复：相邻若干句的文本，在后面又几乎一字不差地出现一遍。"""
    txts = [t for _, _, t in items]
    out = []
    n = len(txts)
    for i in range(n):
        for L in (2, 3, 4):
            if i + 2 * L > n:
                continue
            a = " ".join(txts[i:i + L]).replace(" ", "")
            b = " ".join(txts[i + L:i + 2 * L]).replace(" ", "")
            if len(a) < 8:
                continue
            if a == b:
                out.append((items[i][0], L, a[:60]))
    return out


def show(tag, src, t0):
    items = asr(src, t0, DUR, tag)
    print("=== %s（%s 第 %.2fs 起 %.0fs）===" % (tag, os.path.basename(src), t0, DUR))
    for s, e, t in items:
        print("  %6.2f~%6.2f  %s" % (s, e, t))
    rep = find_repeat(items)
    if rep:
        print("  🔴 检测到连续重复：")
        for at, L, a in rep:
            print("     第 %.2fs 起，连续 %d 句重复：%s" % (t0 + at, L, a))
    else:
        print("  ✅ 未检测到连续重复")
    # 语速
    body = "".join(t for _, _, t in items).replace(" ", "")
    voiced = sum(e - s for s, e, _ in items)
    if voiced > 0.5:
        print("  发声 %.1fs / 字数 %d → %.2f 字/s"
              % (voiced, len(body), len(body) / voiced))
    print()
    return items


def main():
    # 员工第三题听到的两段
    show("A_现役_s3", CUR, 319.79)
    show("B_GS_s3", GS, 313.90)
    # 员工说 11 秒（在段内），单独把 8~14s 的窗口再放大看一次
    print("### 员工点名的「第 11 秒」窗口（段内 8~14s，更贴近点名位置）###")
    show("A_现役_s3_win", CUR, 319.79 + 8.0)
    show("B_GS_s3_win", GS, 313.90 + 8.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
