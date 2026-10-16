#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""maleref47.py —— 男声 GS 参考音频：从 15s 标杆切出 10s，并做身份核验。

🔴 GS 硬坑：参考音频必须在 **3~10 秒**。现役男声参考
   `EL_male.wav` 14.98s / `yohan_male.wav` 15.00s → 走 /api/jobs 一律
   500 `Reference audio is outside the 3-10 second range`。
   女声已用 `prompt_sola_00b6a45a.wav`(10s) 解决（与 EL_female 余弦 0.7953 同人）。

本脚本：
  ① 从 yohan_male.wav（15s）切若干候选 10s 片段
  ② campplus 对每个候选打分，与 EL_male（标杆）比对，阈值 0.75
  ③ 挑最像的那个，写到 /mnt/tts/data/voices/yohan_gs10.wav

⚠️ 绝不能拿 elara.wav 当男声参考（与 EL_male 只有 0.3424，不同人）。
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")

OUT = "/mnt/tts/data/voices"
SRC = "/mnt/tts/data/voices/yohan_male.wav"
REF = "/mnt/tts/data/voices/EL_male.wav"
# 起点候选：避开最开头（可能有静音/噪声），也避开最尾
STARTS = [0.0, 1.0, 2.0, 3.0, 4.0]


def dur(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "default=nw=1:nk=1", p],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def main():
    print(f"源 {SRC}  {dur(SRC):.2f}s")
    print(f"标杆 {REF}  {dur(REF):.2f}s")

    import onnxruntime
    import spkchk

    opt = onnxruntime.SessionOptions()
    opt.graph_optimization_level = \
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.intra_op_num_threads = 2
    sess = onnxruntime.InferenceSession(
        spkchk.CAMPPLUS, sess_options=opt,
        providers=["CPUExecutionProvider"])

    def emb(p):
        return spkchk.embed(sess, spkchk.read16(p))

    ref_emb = emb(REF)
    base_emb = emb(SRC)
    cos = float(ref_emb @ base_emb)
    print(f"\nyohan_male(15s) × EL_male 余弦 = {cos:.4f}  "
          f"{'✅ 同人' if cos >= 0.75 else '❌ 不同人'}")

    best, best_cos = None, -1.0
    print(f"\n{'起点':>6}{'余弦':>10}  判定")
    for s in STARTS:
        if s + 10.0 > dur(SRC) + 1e-6:
            continue
        p = f"/mnt/tts/diag/male47/cand_{int(s)}.wav"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", SRC,
                        "-ss", str(s), "-t", "10.0", "-ar", "16000",
                        "-ac", "1", p], check=True)
        c = float(emb(p) @ ref_emb)
        print(f"{s:>6.1f}{c:>10.4f}  {'✅ 同人' if c >= 0.75 else '❌'}")
        if c > best_cos:
            best, best_cos = p, c

    if best is None or best_cos < 0.75:
        print("\n❌ 没有同人候选，别硬上")
        return 1

    dst = os.path.join(OUT, "yohan_gs10.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", best,
                    "-ar", "24000", "-ac", "1", dst], check=True)
    print(f"\n✅ 选中 {best}（{best_cos:.4f}）-> {dst}  {dur(dst):.2f}s")

    # 注册到 voices.json
    vp = "/mnt/tts/data/voices/voices.json"
    d = json.load(open(vp, encoding="utf-8"))
    d["yohan_gs10"] = {
        "path": dst,
        "ref_text": d.get("yohan_male", {}).get("ref_text", "")[:200],
        "sr": 24000,
        "dur": round(dur(dst), 2),
        "engine": "gptsovits",
        "gender": "m",
        "bench_case": "6cdd25bc75f2",
        "note": "第47轮：GS 专用 10s 参考（GS 只接受 3~10s，15s 会 500）",
    }
    json.dump(d, open(vp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"✅ 已注册音色 yohan_gs10 -> {vp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
