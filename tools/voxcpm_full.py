# -*- coding: utf-8 -*-
"""VoxCPM2 整篇候选生成：与案例 EL 同稿比较，供员工盲听。

策略
----
1. 首选**整篇一次提交**（观察长文稳定性/漂移）。
2. 失败则按句子边界切块重试，硬拼接（记录分块事实，不冒充整篇一次成型）。
3. 参考用官方推荐的最大相似度做法：同一段参考音频同时给
   reference_wav_path 和 prompt_wav_path+prompt_text。

用法（服务器）：
    CUDA_VISIBLE_DEVICES=GPU-... python -u tools/voxcpm_full.py
"""
import json
import os
import re
import time

import numpy as np
import soundfile as sf

BASE = "/mnt/tts/projects/voxcpm"
MODEL = os.path.join(BASE, "models/VoxCPM2")
GSOUT = "/mnt/tts/projects/gpt-sovits/out"
CASES = "/mnt/tts/data/cases"
OUT = "/mnt/tts/diag/voxcpm-20261013"
os.makedirs(OUT, exist_ok=True)

# (输出名, 案例id, 性别, 参考wav, 参考txt)
JOBS = [
    ("V1_m", "d172b8c4efb3", "male",
     os.path.join(GSOUT, "prompt_yohan_1a6c20a6.wav"),
     os.path.join(GSOUT, "prompt_yohan_1a6c20a6.txt")),
    ("V2_f", "ce1a3fda551a", "female",
     os.path.join(GSOUT, "prompt_sola_00b6a45a.wav"),
     os.path.join(GSOUT, "prompt_sola_00b6a45a.txt")),
]

CHUNK_CHARS = 600          # 回退分块大小（句边界对齐）
# 🔴 2026-10-13 实测：女声 3948 字走整篇模式只出 218.4s 就「成功」返回
#    （EL 525s / 男声分块 487s）——**静默截断**，比报错更危险。
#    默认强制分块；只有显式允许时才尝试整篇。
FORCE_CHUNK = os.environ.get("VOXCPM_FORCE_CHUNK", "1") != "0"
ONLY = os.environ.get("VOXCPM_ONLY", "")          # 只跑指定输出名


def case_text(cid):
    d = os.path.join(CASES, cid)
    segs = sorted([f for f in os.listdir(d)
                   if f.startswith("seg") and f.endswith(".txt")])
    return "\n\n".join(open(os.path.join(d, f), encoding="utf-8").read().strip()
                       for f in segs if open(os.path.join(d, f),
                                             encoding="utf-8").read().strip())


def split_sentences(text, limit):
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 > limit and cur:
            chunks.append(cur.strip())
            cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def main():
    from voxcpm.core import VoxCPM
    t0 = time.time()
    model = VoxCPM(voxcpm_model_path=MODEL, enable_denoiser=False,
                   optimize=True, device="cuda")
    sr = int(model.tts_model.sample_rate)
    print("加载 %.1fs sr=%d" % (time.time() - t0, sr), flush=True)

    report = []
    for name, cid, gender, rw, rt in JOBS:
        if ONLY and name != ONLY:
            continue
        txt = case_text(cid)
        ptext = open(rt, encoding="utf-8").read().strip()
        print("\n==== %s %s %d字 ====" % (name, gender, len(txt)), flush=True)

        def gen(text, **kw):
            return model.generate(
                text=text, reference_wav_path=rw,
                prompt_wav_path=rw, prompt_text=ptext,
                cfg_value=2.0, inference_timesteps=10, normalize=True)

        mode, audio = "full", None
        t1 = time.time()
        if not FORCE_CHUNK:
            try:
                audio = gen(txt)
            except Exception as e:
                print("整篇失败：%r，回退分块" % (e,), flush=True)
        else:
            mode = "chunked"
        el = time.time() - t1
        nchunks = 1
        if audio is None or mode == "chunked":
            mode = "chunked"
            pieces, t1 = [], time.time()
            for c in split_sentences(txt, CHUNK_CHARS):
                pieces.append(np.asarray(gen(c)))
                print("  块%d/%d %.1fs" % (
                    len(pieces), 0, len(pieces[-1]) / sr), flush=True)
            nchunks = len(pieces)
            audio = np.concatenate(pieces) if pieces else None
            el = time.time() - t1
        if audio is None or len(audio) == 0:
            report.append({"name": name, "status": "failed"})
            continue
        audio = np.asarray(audio)
        dst = os.path.join(OUT, name + ".wav")
        sf.write(dst, audio, sr, subtype="PCM_16")
        print("-> %s %.1fs 模式=%s 块=%d 用时 %.1fs"
              % (dst, len(audio) / sr, mode, nchunks, el), flush=True)
        report.append({
            "name": name, "case": cid, "gender": gender, "status": "ok",
            "chars": len(txt), "mode": mode, "chunks": nchunks,
            "audio_seconds": round(len(audio) / sr, 2), "sr": sr,
            "elapsed_s": round(el, 1),
            "ref": os.path.basename(rw), "out": dst,
        })

    json.dump(report, open(os.path.join(OUT, "voxcpm_report.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nDONE", json.dumps(report, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
