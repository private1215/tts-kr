# -*- coding: utf-8 -*-
"""omnichunk_ab53.py —— 验证「极短嗯」是否为 OmniVoice 内部分块伪影。

背景（2026-10-17 检索 OmniVoice 官方文档得到）：
`docs/generation-parameters.md` 明确写着：
    audio_chunk_duration = 15.0   目标分块时长(秒)
    audio_chunk_threshold = 30.0  预估时长超过它就激活分块
→ **长文本会被 generate() 内部按 15 秒切开**，不是整篇一次生成。
   omni_server.py 的 `m.generate(text=..., ref_audio=..., ref_text=...)`
   没传任何参数，全走默认值 → 4594 字(约550s)被切成约 37 块。

而 omni_server.py 注释里的「必须整篇一次提交，分块会换韵律」是**误解**：
客户端传整篇 ≠ 模型整篇生成，内部照样按 15s 切。

假设：员工听到的「特别特别短的嗯，超级短」是**分块边界的伪影**。
验证：同一文本、同一参考，只改 audio_chunk_duration，
      若「嗯」数随块变长而显著下降 → 假设成立，且这是参数级修复。
"""
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

MODEL_DIR = "/mnt/tts/projects/omnivoice/models/OmniVoice"
REF_WAV = "/mnt/tts/projects/gpt-sovits/out/prompt_yohan_1a6c20a6.wav"
REF_TXT = "/mnt/tts/projects/gpt-sovits/out/prompt_yohan_1a6c20a6.txt"
CASE = "/mnt/tts/data/cases/d172b8c4efb3"
OUT = "/mnt/tts/diag/omnichunk53"
# 只测前 N 字，控制实验时长（约 100s 音频）；100000 = 全篇
CHARS = int(os.environ.get("OMNICHUNK_CHARS", "100000"))

os.makedirs(OUT, exist_ok=True)


def grunt_count(x, sr=16000, frame=0.005):
    """极短孤立发声段（员工说的『特别特别短的嗯』）数量。"""
    n = int(frame * sr)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > np.percentile(db, 35)
    segs, i = [], 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            segs.append((i * frame, (j - i) * frame))
            i = j
        else:
            i += 1
    out = []
    for k in range(1, len(segs) - 1):
        s, d = segs[k]
        if 0.02 <= d <= 0.12:
            pe = segs[k - 1][0] + segs[k - 1][1]
            nx = segs[k + 1][0]
            if (s - pe) > 0.15 and (nx - (s + d)) > 0.15:
                out.append((s, d))
    return out


def main():
    import torch
    from omnivoice import OmniVoice

    model = OmniVoice.from_pretrained(MODEL_DIR, device_map="cuda",
                                      dtype=torch.float16)
    sr = 24000

    txt = open(os.path.join(CASE, "full.txt"), encoding="utf-8").read().strip()
    parts = txt.split(". ")
    out, n = "", 0
    for p in parts:
        if n + len(p) + 1 > CHARS and out:
            break
        out = (out + ". " + p).strip()
        n = len(out)
    rtext = open(REF_TXT, encoding="utf-8").read().strip()
    print("测试文本 %d 字" % len(out), flush=True)

    results = []
    for chunk in (15.0, 60.0):
        t0 = time.time()
        audio = model.generate(text=out, ref_audio=REF_WAV, ref_text=rtext,
                               audio_chunk_duration=chunk)
        a = np.asarray(audio[0] if isinstance(audio, (list, tuple)) else audio)
        if a.ndim > 1:
            a = a.mean(axis=1)
        dst = os.path.join(OUT, "chunk%d.wav" % int(chunk))
        sf.write(dst, a.astype(np.float32), sr, subtype="PCM_16")
        dur = len(a) / sr
        g = grunt_count(a.astype(np.float32))
        print("audio_chunk_duration=%-5s  %.1fs  用时%.0fs  极短嗯 %d 个 (%.2f/分钟)"
              % (chunk, dur, time.time() - t0, len(g), len(g) / dur * 60), flush=True)
        for s, d in g:
            print("      t=%.2fs  %.0fms" % (s, d * 1000), flush=True)
        results.append({"chunk": chunk, "dur": dur, "grunts": len(g),
                        "detail": [(round(s, 2), round(d * 1000)) for s, d in g]})
        torch.cuda.empty_cache()

    json.dump(results, open(os.path.join(OUT, "ab.json"), "w"), indent=2)
    a, b = results[0], results[1]
    print("\n结论：15s 块 %d 个 vs 60s 块 %d 个" % (a["grunts"], b["grunts"]), flush=True)
    if b["grunts"] < a["grunts"] * 0.6:
        print("✅ 「嗯」随块变长显著下降 → 是分块边界伪影，"
              "增大 audio_chunk_duration 是可用的参数级修复", flush=True)
    else:
        print("❌ 「嗯」与分块无明显关系 → 是模型自发生成，需另找方向", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
