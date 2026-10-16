# -*- coding: utf-8 -*-
"""用 ASR 带时间戳转录男声参考音频，确定「前 N 秒到底念了哪些字」。
这是根治 ref_text 与 ref_audio 不匹配的唯一可靠办法（包络只能估）。
"""
import sys, json
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from app import asr_check

PATHS = ["/mnt/tts/data/voices/yohan_male.wav",     # 15s 原始
         "/mnt/tts/data/voices/yohan_gs10.wav",     # 10s 切片
         "/mnt/tts/data/voices/SOLA_gs10.wav"]      # 女声对照(已知好)

m = asr_check._load()
print("ASR model:", m)
for p in PATHS:
    x = asr_check._read_16k(p)
    sr = 16000
    print(f"\n===== {p}  ({len(x)/sr:.2f}s) =====")
    segs, info = m.transcribe(x, language="ko", vad_filter=True, beam_size=5,
                              word_timestamps=False)
    for s in segs:
        print(f"  {s.start:6.2f} ~ {s.end:6.2f}s  {s.text.strip()}")
    full = " ".join(s.text for s in segs).strip()
    print(f"  -- 全文 {len(full)}字: {full}")
    # 10s 处念到哪
    acc = 0
    for s in segs:
        if s.start >= 10.0:
            break
        acc += len(s.text.strip())
    print(f"  -- 0~10s 内累计字数(粗) = {acc}")

print("\n=== voices.json 里写的 ref_text ===")
v = json.load(open("/mnt/tts/data/voices/voices.json"))
for k in ("yohan_male", "yohan_gs10", "SOLA_gs10"):
    t = v.get(k, {}).get("ref_text", "")
    print(f"  {k}: {len(t)}字  {t[:70]}")
