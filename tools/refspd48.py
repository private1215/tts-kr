# -*- coding: utf-8 -*-
"""refspd48.py —— 找一段**隐含语速更快**的男声 GS 参考音频。

## 为什么
男声 voxcpm 现役 click 1.175%（EL 的 4.5 倍）、静音段 4-8k 18.201（EL 10.968）；
男声 GS（gs47m8_final）静音 4-8k **11.423 ≈ EL 10.968**、click **0.517%**（好一倍多），
质量全面占优。但 GS 成品 cps 只有 **6.92**，EL 男声是 **8.23**，**慢 16%**。

## 依据（已验证的规律）
**GS 输出语速 ≈ 参考音频的隐含语速（ref_text 字数 ÷ 音频时长）**
  参考 16.0 字/s → 输出 1.98；参考 7.34 → 6.45；参考 6.75 → 6.75（1:1）

现役男声参考 `yohan_gs10` = 7.55s / 51 字 = 6.75 字/s → 输出 6.92 ✅ 规律成立。
所以要把输出提到 ~8.3，需要参考隐含语速 ≈ 8.3 字/s（按 6.75→6.92 的比例约 ×1.025 折算）。

## 约束（三条都不能破）
1. 时长 **3~10 秒**（GS 硬性限制，超出一律 500）
2. 必须切在**句边界**（不能按固定秒数硬切，否则 ref_text 与音频对不上）
3. campplus × EL_male **≥ 0.75**（同一人）

用法：python3 refspd48.py [--target 8.3] [--max-sec 180]
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from asr_check import _load, _read_16k  # noqa: E402

EL_MALE = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"
OUTD = "/mnt/tts/diag/refspd48"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=8.3, help="目标隐含语速 字/s")
    ap.add_argument("--max-sec", type=float, default=180.0)
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)

    # 只转前 max-sec 秒，省时间
    x = _read_16k(EL_MALE)
    x = x[:int(16000 * a.max_sec)]
    m = _load()
    print("ASR 转录前 %.0fs ..." % a.max_sec, flush=True)
    segs, _ = m.transcribe(x, language="ko", vad_filter=True, beam_size=5)
    segs = [{"s": s.start, "e": s.end, "t": s.text.strip()} for s in segs
            if s.text.strip()]
    for s in segs[:6]:
        print("  %6.2f~%6.2f  %s" % (s["s"], s["e"], s["t"][:50]))
    print("  共 %d 句" % len(segs), flush=True)

    # 从每个句子起点开始，向后累积到 3~10s，算 字数/时长
    cands = []
    for i, s0 in enumerate(segs):
        for j in range(i, len(segs)):
            dur = segs[j]["e"] - s0["s"]
            if dur < 3.0:
                continue
            if dur > 10.0:
                break
            nch = sum(len(segs[k]["t"]) for k in range(i, j + 1))
            spd = nch / dur
            # 必须整句收尾，且时长留余量
            cands.append({"s": s0["s"], "e": segs[j]["e"], "dur": dur,
                          "nch": nch, "spd": spd,
                          "txt": " ".join(segs[k]["t"] for k in range(i, j + 1))})
    cands.sort(key=lambda c: abs(c["spd"] - a.target))
    print("\n=== 最接近 %.1f 字/s 的候选 ===" % a.target)
    for c in cands[:a.top]:
        print("  %.2f~%.2fs (%5.2fs) %3d字  %.2f字/s" %
              (c["s"], c["e"], c["dur"], c["nch"], c["spd"]))
        print("       %s" % c["txt"][:80])

    # campplus 验证 + 落盘
    import onnxruntime
    import spkchk
    opt = onnxruntime.SessionOptions()
    opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.intra_op_num_threads = 2
    sess = onnxruntime.InferenceSession(spkchk.CAMPPLUS, sess_options=opt,
                                        providers=["CPUExecutionProvider"])
    emb = lambda p: spkchk.embed(sess, spkchk.read16(p))
    cos = lambda u, v: float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    ref = emb("/mnt/tts/data/voices/EL_male.wav")
    print("\n=== 切片 + 音色验证（×EL_male，阈值 0.75）===")
    for k, c in enumerate(cands[:3]):
        w = os.path.join(OUTD, "cand%d.wav" % k)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", EL_MALE,
                        "-ss", "%.3f" % c["s"], "-to", "%.3f" % c["e"],
                        "-ar", "24000", "-ac", "1", w], check=True)
        sc = cos(emb(w), ref)
        print("  cand%d  %.2fs %d字 %.2f字/s  音色 %.4f  %s"
              % (k, sf.info(w).duration, c["nch"], c["spd"], sc,
                 "同人 OK" if sc >= 0.75 else "不同人!!"))
        c["wav"] = w
        c["cos"] = sc

    json.dump(cands[:3], open(os.path.join(OUTD, "cands.json"), "w",
                              encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n已写 %s/cands.json" % OUTD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
