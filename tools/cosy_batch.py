# -*- coding: utf-8 -*-
"""CosyVoice3 批量对照样本：同文案 × 长短 × 多 seed × 男女。

为什么重点跑 Cosy：上一轮三引擎打分里，CosyVoice3 零样本音色 0.9074，
明显高于当前线上 Qwen 的 0.8441，而 CER 0.0583 对 0.0563 基本持平。
更关键的是 **Cosy 已经作为独立服务常驻在 3090 上，启用它对主服务零显存代价**
（Qwen 常驻 3060 6.89G，两边不抢），所以它是最便宜的一档提升。

但本项目因为"单样本就下结论"翻过两次车，所以这一轮一律配：
  - 两种长度：短 304 字 / 长 2902 字（长文本是 Cosy 从未测过的场景）
  - 两个 seed：42 / 2026
  - 男女各一套参照

🔴 参照音频统一规则：参照**必须取自案例自己**，且和文案对应：
  - 女声 = 韩华 c3117b7b0fb1（이민정），文案 hh_short.txt
  - 男声 = 산일전기 799e4d7060ce（정민철），文案取 gs_long.txt 前 400 字
  两边都不能拿别人的录音去克隆另一个人。

用法：
    python tools/cosy_batch.py
"""
import os
import subprocess
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
os.chdir("/mnt/tts/projects/qwen3-tts")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from app.cosy_client import synth  # noqa: E402

OUT = "/mnt/tts/projects/gpt-sovits/out"
CASE_F = "/mnt/tts/data/cases/c3117b7b0fb1/merged.wav"
CASE_M = "/mnt/tts/data/cases/799e4d7060ce/merged.wav"

SHORT = open("/mnt/tts/diag/hh_short.txt", encoding="utf-8").read().strip()
LONG = open("/mnt/tts/diag/gs_long.txt", encoding="utf-8").read().strip()


def make_prompt(case, tag, a=5.0, b=15.0):
    """从案例取一段参照 + 转写。⚠️ 只用案例**开头**的一小段，
    其余留着当测试区，避免"用要考的那段去当提示"。"""
    wav = os.path.join(OUT, "prompt_%s.wav" % tag)
    txt = os.path.join(OUT, "prompt_%s.txt" % tag)
    if os.path.exists(wav) and os.path.exists(txt):
        return wav, open(txt, encoding="utf-8").read().strip()
    x, sr = sf.read(case, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    seg = x[int(a * sr):int(b * sr)]
    sf.write(wav, seg, sr, subtype="PCM_16")
    # 🔴 不能把文件路径直接交给 faster_whisper：这台机器的 av/PyAV 版本和
    #    whisper 不兼容，会抛 TypeError: open() got an unexpected keyword
    #    argument 'metadata_errors'（2026-10-12 踩到）。
    #    改成自己读成 16k 的 ndarray 再喂进去，顺带用 CPU int8 避开 GPU 争用。
    import librosa
    from faster_whisper import WhisperModel
    os.environ.setdefault("HF_HOME", "/mnt/tts/models/asr")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    y, sr0 = sf.read(wav, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    y16 = librosa.resample(y, orig_sr=sr0, target_sr=16000)
    m = WhisperModel("medium", device="cpu", compute_type="int8")
    segs, _ = m.transcribe(y16, language="ko", beam_size=1)
    t = " ".join(s.text for s in segs).strip()
    open(txt, "w", encoding="utf-8").write(t)
    return wav, t


# (输出名, 参照tag, 文案, seed, instruct)
JOBS = [
    ("cosy_f_short_s2026", "f10s", SHORT, 2026, "neutral"),
    ("cosy_f_long_s42",    "f10s", LONG,  42,   "neutral"),
    ("cosy_f_long_s2026",  "f10s", LONG,  2026, "neutral"),
    ("cosy_m_short_s42",   "m799", LONG[:400], 42, "neutral"),
    ("cosy_m_short_s2026", "m799", LONG[:400], 2026, "neutral"),
]


def main():
    pf, tf = make_prompt(CASE_F, "f10s")
    pm, tm = make_prompt(CASE_M, "m799")
    print("女参照:", tf, flush=True)
    print("男参照:", tm, flush=True)
    prom = {"f10s": (pf, tf), "m799": (pm, tm)}

    for name, tag, text, seed, ins in JOBS:
        wav, txt = prom[tag]
        print("=== %s  seed=%d  字数=%d ===" % (name, seed, len(text)), flush=True)
        audio, sr, meta = synth(text=text, ref_wav=wav, ref_text=txt,
                                instruct=ins, speed=1.0, seed=seed)
        if audio is None:
            print("   失败:", meta.get("error"), flush=True)
            continue
        audio = np.asarray(audio, dtype=np.float32)
        p = os.path.join(OUT, name + ".wav")
        sf.write(p, audio, sr, subtype="PCM_16")
        print("   -> %s  %.1fs @%d" % (p, len(audio) / sr, sr), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
