# -*- coding: utf-8 -*-
"""三引擎统一打分：Qwen3-TTS / CosyVoice3 / GPT-SoVITS。

🔴 本项目已经因为"只看单项"翻过两次车（adapter、句尾下降），
   所以这里**不做总分**，四项列出来让人自己权衡：
     1. 说话人相似度  —— campplus 余弦，vs 案例原声（沿用既有口径，结果可和旧表对齐）
     2. CER           —— 内容有没有跑飞/吞字
     3. 韵律贴合       —— 句尾降幅 / 停顿时长 / 停顿密度 / 句中位Hz 四项，
                         全部报**带符号的偏差**（对照案例原声），不合成一个数
     4. 时长/语速     —— 相对案例的速度比（1.0 = 和真人一样快）

用法：
    python tools/three_engine_score.py
"""
import os
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")
os.environ.setdefault("HF_HOME", "/mnt/tts/models/asr")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from spkchk import read16, embed, CAMPPLUS  # noqa: E402
from prosody_target import measure_wav  # noqa: E402
from app.asr_check import transcribe, cer  # noqa: E402

GT = open("/mnt/tts/diag/hh_short.txt", encoding="utf-8").read().strip()
FB = "/mnt/tts/data/tone-ab"
GSO = "/mnt/tts/projects/gpt-sovits/out"

REF = os.path.join(FB, "hh_ref.mp3")

CAND = [
    ("参照（案例原声）",            REF),
    ("Qwen EL_female（当前默认）",  os.path.join(FB, "hh_el.mp3")),
    ("Qwen SOLA_female",            os.path.join(FB, "hh_sola.mp3")),
    ("CosyVoice3 零样本",           os.path.join(GSO, "cosy_zs.wav")),
    ("CosyVoice3 激昂",             os.path.join(GSO, "cosy_exc.wav")),
    ("GPT-SoVITS 零样本",           os.path.join(GSO, "gs_zeroshot.wav")),
    ("GPT-SoVITS 微调 e30",         os.path.join(GSO, "gs_e30_s480_s42.wav")),
]

# 韵律四项：脚本里的 key -> 展示名 -> 单位
PROS = [("tone_amp_p50", "句尾降幅", "半音"),
        ("pause_dur_p50", "停顿时长", "s"),
        ("pause_n_per_min", "停顿密度", "次/分"),
        ("pitch_med", "句中位音高", "Hz")]


def main():
    print("目标文案 %d 字 | 参照 %s" % (len(GT), os.path.basename(REF)))
    print()
    sess = ort.InferenceSession(CAMPPLUS, providers=["CPUExecutionProvider"])

    rows = []
    ref = None
    for name, p in CAND:
        if not os.path.exists(p):
            print("⚠️ 缺文件", p)
            continue
        m = measure_wav(p)
        e = embed(sess, read16(p))
        sim = float(np.dot(embed(sess, read16(REF)), e)) if e is not None else float("nan")
        if ref is None:
            ref = m
            ref_sim = sim
        try:
            asr = transcribe(p, "ko")
            c = cer(GT, asr)
        except Exception as ex:
            print("ASR 失败", name, str(ex)[:80])
            c = None
        rows.append((name, m, sim, c))

    r0 = rows[0][1]
    print("%-26s %8s %8s %8s %8s" % ("候选", "相似度", "CER", "时长s", "速度比"))
    print("-" * 62)
    for name, m, sim, c in rows:
        ratio = m["dur"] / r0["dur"]
        print("%-26s %8.4f %8s %8.1f %8.3f"
              % (name, sim, ("%.4f" % c) if c is not None else "-", m["dur"], ratio))

    print()
    print("韵律贴合度 —— 全部是**相对案例原声的带符号偏差**，越接近 0 越贴合：")
    print("%-26s %s" % ("候选", "".join("%14s" % n for _, n, _ in PROS)))
    print("-" * 84)
    print("%-26s %s" % ("参照（案例原声）",
                        "".join("%14s" % ("%s %s" % (r0[k], u)) for k, _, u in PROS)))
    for name, m, sim, c in rows[1:]:
        cells = []
        for k, _, u in PROS:
            a, b = m.get(k), r0.get(k)
            if a is None or b is None:
                cells.append("%14s" % "-")
            else:
                dv = a - b
                mark = "" if abs(dv) < abs(b) * 0.15 + 1e-9 else "*"
                cells.append("%14s" % ("%+.2f%s" % (dv, mark)))
        print("%-26s %s" % (name, "".join(cells)))
    print()
    print("（* = 偏离参照超过 15%，值得注意；带符号便于判断方向，不要只看绝对值）")


if __name__ == "__main__":
    main()
