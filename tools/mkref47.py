# -*- coding: utf-8 -*-
"""按 ASR 时间戳在**句边界**重新切男声 GS 参考音频，并同步写对 ref_text。
候选：
  P1 = [0.00, 5.42]  제1句 40字  -> 7.38 字/秒
  P2 = [5.42, 12.94] 제2句 51字  -> 6.78 字/秒
对照组 SOLA_gs10 = 10s / 73字 = 7.3 字/秒（已知 CER 0.052 合格）
"""
import os, sys, json, shutil
import numpy as np, soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
import onnxruntime, spkchk

SRC = "/mnt/tts/data/voices/yohan_male.wav"
OUTD = "/mnt/tts/data/voices"
VOICES = os.path.join(OUTD, "voices.json")

x, sr = sf.read(SRC, dtype="float32", always_2d=False)
if x.ndim > 1:
    x = x.mean(axis=1)
print(f"源 {SRC} {len(x)/sr:.2f}s sr={sr}")

V = json.load(open(VOICES, encoding="utf-8"))
full = V["yohan_male"]["ref_text"]
import re
parts = [p for p in re.split(r"(?<=[.?!])\s*", full) if p.strip()]
print("句子:")
for i, p in enumerate(parts):
    print(f"  [{i}] {len(p):3d}字  {p}")

# ASR 测得句边界
BOUND = [(0.00, 5.42, 0), (5.42, 12.94, 1)]

opt = onnxruntime.SessionOptions()
opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
opt.intra_op_num_threads = 2
sess = onnxruntime.InferenceSession(spkchk.CAMPPLUS, sess_options=opt,
                                    providers=["CPUExecutionProvider"])
def emb(p):
    return spkchk.embed(sess, spkchk.read16(p))

ref_el = emb("/mnt/tts/data/voices/EL_male.wav")

made = {}
for t0, t1, si in BOUND:
    # 尾端留 30ms 静音尾巴，避免硬切
    seg = x[int(t0*sr): int(min(t1 + 0.03, len(x)/sr)*sr)]
    name = f"yohan_p{si+1}"
    path = os.path.join(OUTD, f"{name}.wav")
    sf.write(path, seg, sr)
    txt = parts[si].strip()
    dur = len(seg)/sr
    cos = float(np.dot(emb(path), ref_el) /
                (np.linalg.norm(emb(path))*np.linalg.norm(ref_el)))
    made[name] = dict(path=path, dur=dur, text=txt, cps=len(txt)/dur, cos=cos)
    print(f"\n{name}: {dur:.2f}s  {len(txt)}字  {len(txt)/dur:.2f}字/秒  "
          f"campplus×EL_male={cos:.4f} ({'同人' if cos>=0.75 else '不同人!!'})")
    if dur < 3 or dur > 10:
        print(f"   ⚠️ 时长 {dur:.2f}s 超出 GS 参考要求的 3~10s")

# 对照：SOLA_gs10
try:
    d = sf.info("/mnt/tts/data/voices/SOLA_gs10.wav").duration
    t = V["SOLA_gs10"]["ref_text"]
    print(f"\n对照 SOLA_gs10: {d:.2f}s {len(t)}字 {len(t)/d:.2f}字/秒 (已知合格)")
except Exception as e:
    print(f"\n对照 SOLA_gs10 读取失败: {e}")

json.dump(made, open("/mnt/tts/diag/mkref47.json", "w"), ensure_ascii=False, indent=1)
print("\n已写出 /mnt/tts/diag/mkref47.json")
