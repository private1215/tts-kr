# -*- coding: utf-8 -*-
"""串联方案：千问打底 + Cosy 加情绪 —— **不是**拼接/混音。

定义
----
用千问整篇输出的开头片段作为 CosyVoice3 的**参考音频**，用该片段的
**真实转写**作 ref_text，再让 CosyVoice3 以这个参考把**同一篇完整稿件**
重新念一遍。instruct 用项目预设里的 excited（股市直播语气）——
这才是「加情绪」的实际落点，参数留痕写进 manifest。

⚠️ 合规要点
----------
1. ref_text 必须用 ASR 真实转写，不能拿案例原文前几句冒充（会错对齐）。
2. 转写后与案例原文开头做 CER 校验，异常要记录，不静默。
3. 只用 GPU 通过现有 cosy 服务（HTTP），本脚本自身不申请显存。
4. 不改任何生产配置、不改服务。

用法
----
    python -u tools/series_qwen_to_cosy.py
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")

DIAG = "/mnt/tts/diag/longsubmit-20261013"
OUT = "/mnt/tts/diag/series-20261013"
os.makedirs(OUT, exist_ok=True)
CASES = "/mnt/tts/data/cases"

# (输出名, 千问整篇源, 案例id, 性别, instruct)
JOBS = [
    ("S1_m", os.path.join(DIAG, "Q1.wav"), "d172b8c4efb3", "male", "excited"),
    ("S2_f", os.path.join(DIAG, "Q2.wav"), "ce1a3fda551a", "female", "excited"),
]

REF_SEC = 13.0
SIL_DB = -45.0


def case_text(cid):
    d = os.path.join(CASES, cid)
    segs = sorted([f for f in os.listdir(d)
                   if f.startswith("seg") and f.endswith(".txt")])
    return "\n\n".join(open(os.path.join(d, f), encoding="utf-8").read().strip()
                       for f in segs if open(os.path.join(d, f),
                                             encoding="utf-8").read().strip())


def first_voiced(x, sr):
    """跳过开头静音，返回起始样本。"""
    n = int(0.02 * sr)
    for i in range(0, max(len(x) - n, 1), n):
        rms = float(np.sqrt(np.mean(x[i:i + n] ** 2)) + 1e-12)
        if 20.0 * np.log10(rms) > SIL_DB:
            return i
    return 0


def main():
    from app.asr_check import transcribe, cer, normalize
    from app.cosy_client import synth

    report = []
    for name, src, cid, gender, instruct in JOBS:
        print("\n==== %s %s ====" % (name, gender), flush=True)
        if not os.path.exists(src):
            print("缺源文件", src)
            continue
        x, sr = sf.read(src, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        s = first_voiced(x, sr)
        seg = x[s:s + int(REF_SEC * sr)]
        ref_path = os.path.join(OUT, name + "_ref.wav")
        sf.write(ref_path, seg, sr, subtype="PCM_16")
        print("参考片段 %.1fs -> %.1fs (%.1fs)" % (s / sr, (s + len(seg)) / sr,
                                                   len(seg) / sr), flush=True)

        hyp = transcribe(ref_path, language="ko")
        txt = case_text(cid)
        # 对齐校验：拿原文开头同样长度的字符数做粗比对
        est = max(20, int(len(txt) * (REF_SEC / 530.0)))
        c = cer(normalize(txt[:est]), normalize(hyp))
        print("ASR: %s" % hyp[:160], flush=True)
        print("与原文开头 %d 字 CER=%.4f（仅对齐校验，不是成品质量）" % (est, c),
              flush=True)
        if not hyp.strip():
            print("❌ 转写为空，跳过（不拿原文冒充）")
            continue

        t0 = time.time()
        audio, osr, meta = synth(text=txt, ref_wav=ref_path, ref_text=hyp,
                                 instruct=instruct, speed=1.0, seed=42)
        el = time.time() - t0
        if audio is None:
            print("❌ 生成失败:", meta)
            report.append({"name": name, "status": "failed", "meta": str(meta)})
            continue
        dst = os.path.join(OUT, name + ".wav")
        sf.write(dst, np.asarray(audio, dtype="float32"), osr, subtype="PCM_16")
        print("-> %s %.1fs @%dHz  用时 %.1fs" % (dst, len(audio) / osr, osr, el),
              flush=True)
        report.append({
            "name": name, "gender": gender, "case": cid, "status": "ok",
            "source_qwen": src, "ref_segment_start_s": round(s / sr, 2),
            "ref_segment_sec": round(len(seg) / sr, 2),
            "ref_asr_text": hyp, "ref_align_cer": c,
            "instruct": instruct, "chars": len(txt),
            "audio_seconds": round(len(audio) / osr, 2), "sr": osr,
            "elapsed_s": round(el, 1), "out": dst,
        })

    json.dump(report, open(os.path.join(OUT, "series_report.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nDONE", json.dumps(report, ensure_ascii=False)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
