# -*- coding: utf-8 -*-
"""OmniVoice / VoxCPM2 显存与耗时实测（决定能不能常驻、放哪张卡）。

输出：加载耗时、加载后显存、短/中文本生成耗时与实时率 RTF、两者同时驻留的合计显存。
用法: CUDA_VISIBLE_DEVICES=0 <py> bench_engines.py
"""
import gc
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

GSOUT = "/mnt/tts/projects/gpt-sovits/out"
F_REF = (os.path.join(GSOUT, "prompt_sola_00b6a45a.wav"),
         os.path.join(GSOUT, "prompt_sola_00b6a45a.txt"))
M_REF = (os.path.join(GSOUT, "prompt_yohan_1a6c20a6.wav"),
         os.path.join(GSOUT, "prompt_yohan_1a6c20a6.txt"))

# 员工那条文案的一小段（与真实生产同分布）
TEXT_SHORT = (
    "안녕하세요, 필옵틱스 주주 여러분, 반갑습니다. 정민철 프로입니다. "
    "오늘 필옵틱스 주주님들께, 정말 중요한 팩트 하나 짚어드리려고, "
    "이렇게 카메라 켰습니다. 지금 구간에서 절대 섣불리 판단하시면 안 됩니다."
)
TEXT_MID = TEXT_SHORT * 3


def mem(tag):
    a = torch.cuda.memory_allocated() / 1024 ** 3
    r = torch.cuda.memory_reserved() / 1024 ** 3
    print(f"  [{tag}] allocated={a:.2f}GiB reserved={r:.2f}GiB", flush=True)
    return a


def dev_info():
    i = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(i)
    tot = p.total_memory / 1024 ** 3
    print(f"设备 cuda:{i} {p.name} 总显存 {tot:.1f}GiB", flush=True)
    return tot


def main():
    dev_info()
    report = {"device": torch.cuda.get_device_name(0)}
    base = mem("基线")

    # ---------- OmniVoice ----------
    print("\n===== OmniVoice (女声) =====", flush=True)
    import sys
    sys.path.insert(0, "/mnt/tts/projects/omnivoice")
    t = time.time()
    from omnivoice import OmniVoice
    om = OmniVoice.from_pretrained("/mnt/tts/projects/omnivoice/models/OmniVoice",
                                   device_map="cuda", dtype=torch.float16)
    load = time.time() - t
    m_om = mem("加载后")
    print(f"  加载 {load:.1f}s", flush=True)
    for label, txt in (("short", TEXT_SHORT), ("mid", TEXT_MID)):
        t = time.time()
        a = om.generate(text=txt, ref_audio=F_REF[0], ref_text=open(F_REF[1], encoding="utf-8").read().strip())
        el = time.time() - t
        arr = np.asarray(a[0]) if isinstance(a, (list, tuple)) else np.asarray(a)
        dur = len(arr) / 24000.0
        print(f"  {label}: {len(txt)}字 -> {dur:.1f}s 音频, 用时 {el:.1f}s, "
              f"RTF={el/max(dur,1e-6):.3f}, 段数={len(a) if isinstance(a,(list,tuple)) else 1}", flush=True)
        report.setdefault("omnivoice", {}).update(
            {"load_s": round(load, 1), "mem_GiB": round(m_om - base, 2),
             f"{label}_rtf": round(el / max(dur, 1e-6), 3)})
    m_om2 = mem("生成后(峰值)")

    # ---------- VoxCPM2 ----------
    print("\n===== VoxCPM2 (男声) =====", flush=True)
    sys.path.insert(0, "/mnt/tts/projects/voxcpm")
    from voxcpm.core import VoxCPM
    t = time.time()
    vx = VoxCPM(voxcpm_model_path="/mnt/tts/projects/voxcpm/models/VoxCPM2",
                enable_denoiser=False, optimize=True, device="cuda")
    load = time.time() - t
    m_vx = mem("加载后")
    print(f"  加载 {load:.1f}s sr={vx.tts_model.sample_rate}", flush=True)
    ptext = open(M_REF[1], encoding="utf-8").read().strip()
    for label, txt in (("short", TEXT_SHORT), ("mid", TEXT_MID)):
        t = time.time()
        a = vx.generate(text=txt, reference_wav_path=M_REF[0],
                        prompt_wav_path=M_REF[0], prompt_text=ptext,
                        cfg_value=2.0, inference_timesteps=10, normalize=True)
        el = time.time() - t
        arr = np.asarray(a)
        dur = len(arr) / int(vx.tts_model.sample_rate)
        print(f"  {label}: {len(txt)}字 -> {dur:.1f}s 音频, 用时 {el:.1f}s, "
              f"RTF={el/max(dur,1e-6):.3f}", flush=True)
        report.setdefault("voxcpm", {}).update(
            {"load_s": round(load, 1), "mem_GiB": round(m_vx - m_om2, 2),
             f"{label}_rtf": round(el / max(dur, 1e-6), 3),
             "sr": int(vx.tts_model.sample_rate)})
    m_both = mem("两者同时驻留")

    report["both_GiB"] = round(m_both, 2)
    print("\n===== 汇总 =====", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    out = "/mnt/tts/diag/bench_engines.json"
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=2)
    print("-> " + out, flush=True)


if __name__ == "__main__":
    main()
