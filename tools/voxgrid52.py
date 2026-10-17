# -*- coding: utf-8 -*-
"""M1 离线网格：voxcpm 参数（STEPS × CFG）对**裸输出**质量的影响。

实验性质 —— 不改生产配置、不动 voices.json、不递增版本。
目的：现役 VOX_STEPS=10 正好是官方文档「标准质量」的最低档
（官方：高质量 20-30、最高 40+；CFG 长篇建议 1.5「更自然」）。
三年一直在后处理上打补丁，从没把引擎本身调到高质量档，本网格补上这一课。

设计要点
--------
1. **同一段文本**跑所有组合，唯一变量是 (steps, cfg)。
2. 分块大小固定 **VOX_CHUNK_CHARS=300**（= 生产值，见 vox-tts.service 第35轮结论；
   不是 vox_server.py 里的默认值 600）。
3. **裸输出**：不做任何后处理，直接落盘。测的是引擎本身。
4. 同时记 RTF —— STEPS 翻倍推理就翻倍，必须量化这个代价。

用法（服务器，zheng 可直接写 projects）：
    # 先让服务释放显存（峰值会冲到 10.9GiB）
    curl -X POST http://127.0.0.1:8400/unload
    CUDA_VISIBLE_DEVICES=GPU-b67eee2c-c672-6915-910d-617c62abebee \
      /mnt/tts/projects/voxcpm/venv/bin/python -u \
      /mnt/tts/projects/voxcpm/voxgrid52.py
"""
import json
import os
import re
import sys
import time

import numpy as np
import soundfile as sf

BASE = "/mnt/tts/projects/voxcpm"
if BASE not in sys.path:
    sys.path.insert(0, BASE)

MODEL = os.path.join(BASE, "models/VoxCPM2")
GSOUT = "/mnt/tts/projects/gpt-sovits/out"
CASE = os.environ.get("VOXGRID_CASE", "d172b8c4efb3")     # 男声案例
OUT = os.environ.get("VOXGRID_OUT", "/mnt/tts/diag/voxgrid52")
# 目标文本长度（含空格字符数）。~800 字 ≈ 100s，click/停顿统计样本足够。
TARGET_CHARS = int(os.environ.get("VOXGRID_CHARS", "830"))
CHUNK_CHARS = int(os.environ.get("VOX_CHUNK_CHARS", "300"))  # 生产值

REF_WAV = os.path.join(GSOUT, "prompt_yohan_1a6c20a6.wav")
REF_TXT = os.path.join(GSOUT, "prompt_yohan_1a6c20a6.txt")

# (steps, cfg)：steps 现役 10 / 官方高质量 20-30；cfg 现役 2.0 / 官方长篇建议 1.5
GRID = [(10, 2.0), (20, 2.0), (30, 2.0),
        (10, 1.5), (20, 1.5), (30, 1.5)]
ONLY = os.environ.get("VOXGRID_ONLY", "")   # 形如 "20_1.5"，只跑一组

os.makedirs(OUT, exist_ok=True)


def split_sentences(text, limit):
    """与 vox_server.py 完全一致的分块逻辑 —— 保证实验可复现生产行为。"""
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


def case_text():
    """取案例 full.txt 前 TARGET_CHARS 字，切在句边界。"""
    p = os.path.join("/mnt/tts/data/cases", CASE, "full.txt")
    raw = open(p, encoding="utf-8").read().strip()
    parts = re.split(r"(?<=[.!?。！？])\s+", raw)
    out = ""
    for s in parts:
        if len(out) + len(s) + 1 > TARGET_CHARS and out:
            break
        out = (out + " " + s).strip()
    return out


def main():
    from voxcpm.core import VoxCPM
    import torch

    txt = case_text()
    ptext = open(REF_TXT, encoding="utf-8").read().strip()
    chunks = split_sentences(txt, CHUNK_CHARS)
    print("文本 %d 字 / %d 块(limit=%d)  参考=%s"
          % (len(txt), len(chunks), CHUNK_CHARS, os.path.basename(REF_WAV)),
          flush=True)

    t0 = time.time()
    model = VoxCPM(voxcpm_model_path=MODEL, enable_denoiser=False,
                   optimize=True, device="cuda")
    sr = int(model.tts_model.sample_rate)
    print("模型加载 %.1fs sr=%d" % (time.time() - t0, sr), flush=True)

    report = []
    for steps, cfg in GRID:
        tag = "%d_%.1f" % (steps, cfg)
        if ONLY and tag != ONLY:
            continue
        print("\n==== steps=%d cfg=%.1f ====" % (steps, cfg), flush=True)
        pieces = []
        t1 = time.time()
        try:
            for i, c in enumerate(chunks):
                a = model.generate(text=c, reference_wav_path=REF_WAV,
                                   prompt_wav_path=REF_WAV, prompt_text=ptext,
                                   cfg_value=cfg, inference_timesteps=steps,
                                   normalize=True)
                pieces.append(np.asarray(a))
                torch.cuda.empty_cache()
                print("  块 %d/%d %.1fs" % (i + 1, len(chunks),
                                            len(pieces[-1]) / sr), flush=True)
        except Exception as e:
            print("  失败：%r" % (e,), flush=True)
            report.append({"tag": tag, "steps": steps, "cfg": cfg,
                           "status": "failed", "error": repr(e)[:200]})
            continue
        el = time.time() - t1
        audio = np.concatenate(pieces) if pieces else np.zeros(1, np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur = len(audio) / sr
        dst = os.path.join(OUT, "m_%s.wav" % tag)
        sf.write(dst, audio.astype(np.float32), sr, subtype="PCM_16")
        print("-> %s  %.1fs  RTF=%.3f  用时%.1fs" % (dst, dur, el / dur, el),
              flush=True)
        report.append({
            "tag": tag, "steps": steps, "cfg": cfg, "status": "ok",
            "chars": len(txt), "chunks": len(chunks), "chunk_chars": CHUNK_CHARS,
            "audio_seconds": round(dur, 2), "sr": sr,
            "elapsed_s": round(el, 1), "rtf": round(el / dur, 3),
            "out": dst,
        })
        json.dump(report, open(os.path.join(OUT, "grid_report.json"), "w"),
                  ensure_ascii=False, indent=2)

    json.dump({"case": CASE, "chars": len(txt), "chunk_chars": CHUNK_CHARS,
               "el_baseline": {"click_per_s": 1.10, "pause_median": None},
               "rows": report},
              open(os.path.join(OUT, "grid_report.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nDONE", json.dumps(report, ensure_ascii=False)[:800], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
