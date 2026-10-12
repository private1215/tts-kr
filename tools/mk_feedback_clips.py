# -*- coding: utf-8 -*-
"""为反馈页盲听题库切音频片段（**统一响度**，否则盲听结果不可信）。

🔴 为什么必须响度归一
----------------------
三引擎产出的"出身"不一样：
  · Qwen 的音频走的是线上完整 pipeline（响度闭环 + peakfit）
  · GS / Cosy 的长文案产出是**裸推理结果**，没过后处理
不归一的话，员工会觉得"声音大的那段更清楚/更好"—— 这是音量偏差，
不是质量偏差。盲听实验里这是头号污染源，必须掐掉。

统一口径：EBU R128 `loudnorm I=-16 TP=-1.5 LRA=11`，44.1kHz 单声道 128k mp3。

🔴 片段为什么取「开头 40 秒」
------------------------------
三引擎语速不同（GS 4.86 / 参照 5.35 字每秒），直接按时间点切中段会对不齐。
但**所有引擎都从文案第一个字开始念**，所以 [10s, 50s] 区间三者对应同一段文案
（跳过前 10 秒是避开各家起势差异）。中段片段只做辅助题，且偏移在容忍范围内。

用法：
    python tools/mk_feedback_clips.py            # 切全部
    python tools/mk_feedback_clips.py --dry-run  # 只看会生成什么
"""
import argparse
import os
import subprocess
import sys

OUT_DIR = os.environ.get("FB_CLIP_DIR", "/mnt/tts/data/fb-new")
GSOUT = "/mnt/tts/projects/gpt-sovits/out"
DIAG = "/mnt/tts/diag"
CASES = "/mnt/tts/data/cases"

# (片段名前缀, 音频路径, 切片起点秒)
# 🔴 三引擎一律用长文案产出（用户 2026-10-12 要求：用最新版做选择题）
GROUPS = [
    ("f", [
        ("ref", "%s/00b6a45ad6aa/merged.wav" % CASES),
        ("qwen", "%s/qwen_fullf_EL_female.wav" % DIAG),
        ("cosy", "%s/cosy_full_f.wav" % GSOUT),
        ("gsft", "%s/full-f_full_00b6a45ad6aa_ft_e40_seg50.wav" % GSOUT),
        ("gszs", "%s/full-f_full_00b6a45ad6aa_zs_e40_seg50.wav" % GSOUT),
    ]),
    ("m", [
        ("ref", "%s/1a6c20a60481/merged.wav" % CASES),
        ("qwen", "%s/qwen_fullm_EL_male.wav" % DIAG),
        ("cosy", "%s/cosy_full_m.wav" % GSOUT),
        ("gsft", "%s/full-m_full_1a6c20a60481_ft_e40_seg50.wav" % GSOUT),
        ("gszs", "%s/full-m_full_1a6c20a60481_zs_e40_seg50.wav" % GSOUT),
    ]),
]

# (片段名后缀, 起点秒, 时长秒)
CLIPS = [("head", 10.0, 40.0), ("mid", 240.0, 40.0)]


def clip(src, dst, ss, dur):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y",
           "-ss", str(ss), "-t", str(dur), "-i", src,
           "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
           "-ar", "44100", "-ac", "1",
           "-codec:a", "libmp3lame", "-b:a", "128k", dst]
    subprocess.run(cmd, check=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    made, missing = [], []
    for g, items in GROUPS:
        for tag, src in items:
            if not os.path.exists(src):
                missing.append(src)
                continue
            for suffix, ss, dur in CLIPS:
                dst = os.path.join(OUT_DIR, "fb_%s_%s_%s.mp3" % (g, tag, suffix))
                if a.dry_run:
                    print("  会生成", dst, "<-", src, "[%gs+%gs]" % (ss, dur))
                    continue
                clip(src, dst, ss, dur)
                made.append(dst)
                print("  ->", dst)
    if missing:
        print("⚠️ 缺失源文件 %d 个：" % len(missing))
        for m in missing:
            print("   ", m)
    print("生成 %d 个片段；缺失 %d 个源" % (len(made), len(missing)))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
