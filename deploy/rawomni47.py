#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rawomni47.py —— 定位「卡音/click」到底是哪一步造出来的。

已确认（loc47 / clickctx47 / hf47）：
  · 现役 omnivoice 成品 click 密度 0.48%，EL 0.26%，GS 0.01%
  · 低通到 5kHz 同带宽后仍是 0.34% vs 0.16%  → 不是亮度差异
  · 110 个 click **100% 落在发声中**，0 个在停顿 / 边界
    → 排除 declick、silfloor、拼接（它们只动停顿）
  · 瞬态整形要求 _legacy_post=True，现役快路径**没跑** → 排除

剩下的嫌疑只剩两个：OmniVoice 引擎裸输出 / fix_chain。
本脚本逐个环节加码，看 click 在**哪一步**被引入：

    step0  引擎裸输出
    step1  + fix_chain --no-clean --no-pitch --eq anull （只做速度/响度）
    step2  + 只 EQ
    step3  + 只 --no-pitch（即 EQ + 停顿清洗）
    step4  全开（现役口径）

用法：python3 rawomni47.py [--chars 1500]
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")

OUT = "/mnt/tts/diag/raw47"
SR = 16000


def load(path, extra=None):
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if extra:
        cmd += extra
    cmd += ["-ar", str(SR), "-ac", "1", "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32)


def click_density(x, sr=SR, thr=0.30):
    n = int(sr * 0.02)
    d = np.abs(np.diff(x))
    m = (len(d) // n) * n
    dmax = d[:m].reshape(-1, n).max(axis=1)
    k = int((dmax >= thr).sum())
    return k, len(dmax), k / len(dmax) * 100.0, float(dmax.mean())


def main():
    chars = int(sys.argv[sys.argv.index("--chars") + 1]) \
        if "--chars" in sys.argv else 1500
    os.makedirs(OUT, exist_ok=True)

    # ---- 文本：案例 00b6a45ad6aa 的正文 -------------------------------
    txt_p = "/mnt/tts/data/cases/00b6a45ad6aa/seg0.txt"
    with open(txt_p, encoding="utf-8") as f:
        text = f.read().strip()
    if len(text) > chars:
        text = text[:chars]
    print(f"文本 {len(text)} 字（取自 {txt_p}）")

    import soundfile as sf
    import omni_client

    print("调 OmniVoice 生成裸输出 ...")
    audio, sr, meta = omni_client.synth(text=text, voice="female")
    if audio is None:
        print("生成失败:", meta)
        return 1
    raw = os.path.join(OUT, "step0_raw.wav")
    sf.write(raw, audio, sr)
    print(f"  裸输出 {len(audio)/sr:.2f}s @ {sr}Hz -> {raw}")

    # ---- 各 step ------------------------------------------------------
    steps = [
        ("step0 引擎裸输出", raw, None),
        ("step1 只速度/响度", "step1.wav",
         ["--no-clean", "--no-pitch", "--eq", "anull"]),
        ("step2 只 EQ", "step2.wav",
         ["--no-clean", "--no-pitch"]),
        ("step3 EQ+停顿清洗", "step3.wav", ["--no-pitch"]),
        ("step4 全开(现役口径)", "step4.wav", []),
    ]

    print(f"\n{'环节':<24}{'|Δ|>=0.3':>9}{'占比':>9}{'|Δ|均值':>10}")
    print("-" * 52)
    for tag, out_name, extra in steps:
        if extra is None:
            path = raw
        else:
            path = os.path.join(OUT, out_name)
            cmd = [sys.executable,
                   "/mnt/tts/projects/qwen3-tts/app/fix_chain.py",
                   raw, path, "--gender", "f", "--chars", str(len(text))]
            cmd += extra
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  {tag}: fix_chain 失败\n{r.stderr[-800:]}")
                continue
        x = load(path)
        k, tot, pct, mean = click_density(x)
        print(f"{tag:<24}{k:>9}{pct:>8.2f}%{mean:>10.4f}")

    # 基线
    print("\n参照：")
    for tag, p in [("EL 原声 wav",
                    "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"),
                   ("GS 微调 块1",
                    "/mnt/tts/output/sola47_f_gptsovits-2/"
                    "sola47_f_gptsovits-1.mp3")]:
        x = load(p)
        k, tot, pct, mean = click_density(x)
        print(f"{tag:<24}{k:>9}{pct:>8.2f}%{mean:>10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
