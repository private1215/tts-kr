# -*- coding: utf-8 -*-
"""用**同一条短文案**横比多个参考音频，看参考切片到底能把音色拉开多少。

背景（2026-10-13）：
  男声案例 1a6c20a60481，同一条文案、同一参照：
    线上（用 15s 的 EL_male.wav）        音色 0.8444
    裸推理（用 10s 的 prompt_yohan_1a6c20a6.wav） 音色 0.9532
  差 0.109。而 tools/gap_step_test.py 已证明：后处理（atempo / mp3 / slowdyn）
  加起来只吃 0.0013 —— 所以差在**生成端**，生成端唯一变量就是参考音频。

本脚本把这段 0.109 拆到参考音频头上：
  同一条 320 字文案 × N 个候选参考 → 直接比 campplus vs 案例原声。

用法：
    python -u tools/ref_ab_short.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

ROOT = "/mnt/tts/projects/qwen3-tts"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import spkchk  # noqa: E402

CAMPPLUS = "/mnt/tts/models/Fun-CosyVoice3-0.5B/campplus.onnx"
OUT = "/mnt/tts/projects/gpt-sovits/out"
TMP = tempfile.mkdtemp(prefix="refab_")

CASE = "1a6c20a60481"
REF = f"/mnt/tts/data/cases/{CASE}/merged.wav"
TEXT_FILE = os.path.join(OUT, f"full_{CASE}.txt")
N_CHARS = 320          # = COSY_MAX_CHARS，单块，不引入分块差异

# (名字, 参考wav, 参考txt or None从voices.json取)
CANDS = [
    ("EL_male(现役15s)", "/mnt/tts/data/voices/EL_male.wav", None, "EL_male"),
    ("prompt_yohan_1a6c20a6(10s)", os.path.join(OUT, "prompt_yohan_1a6c20a6.wav"),
     os.path.join(OUT, "prompt_yohan_1a6c20a6.txt"), None),
    ("prompt_m799(10s)", os.path.join(OUT, "prompt_m799.wav"),
     os.path.join(OUT, "prompt_m799.txt"), None),
    ("yohan_male(voices)", "/mnt/tts/data/voices/yohan_male.wav", None,
     "yohan_male"),
]

_SESS = None


def sess():
    global _SESS
    if _SESS is None:
        import onnxruntime
        _SESS = onnxruntime.InferenceSession(
            CAMPPLUS, providers=["CPUExecutionProvider"])
    return _SESS


def wav16(path):
    out = os.path.join(TMP, "x16.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", out], check=True)
    return out


def emb(path):
    return spkchk.embed(sess(), spkchk.read16(wav16(path)))


def cos(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def ref_text_of(vname):
    import json
    d = json.load(open("/mnt/tts/data/voices/voices.json"))
    return d[vname].get("ref_text") or ""


def main():
    txt = open(TEXT_FILE, encoding="utf-8").read().strip()[:N_CHARS]
    print(f"测试文案：{len(txt)} 字（单块，不分段）\n")
    e_ref = emb(REF)

    from app.cosy_client import synth
    rows = []
    for name, pwav, ptxt, vname in CANDS:
        if not os.path.exists(pwav):
            print(f"{name}: 参考不存在，跳过")
            continue
        ptx = open(ptxt, encoding="utf-8").read().strip() if ptxt else \
            ref_text_of(vname)
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", pwav],
            capture_output=True, text=True).stdout.strip() or 0)
        try:
            audio, sr, meta = synth(text=txt, ref_wav=pwav, ref_text=ptx,
                                    instruct="neutral", speed=1.0, seed=42)
        except Exception as e:
            print(f"{name}: synth 异常 {e!r}"[:200])
            continue
        if audio is None:
            print(f"{name}: 生成失败 {meta}")
            continue
        dst = os.path.join(TMP, name.split("(")[0] + ".wav")
        sf.write(dst, np.asarray(audio, dtype="float32"), sr)
        s = cos(e_ref, emb(dst))
        rows.append((name, dur, s, len(audio) / sr))
        print(f"{name:<32} ref={dur:5.2f}s  音色={s:.4f}  出片{len(audio)/sr:.1f}s",
              flush=True)

    print("\n=== 排序 ===")
    for name, dur, s, od in sorted(rows, key=lambda r: -r[2]):
        print(f"{s:.4f}  {name:<32} ref={dur:5.2f}s")


if __name__ == "__main__":
    main()
