#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""maleq47.py —— 男声 EQ 的 A/B：带 EQ（现役）vs 不带 EQ。

女声已定案：任何 IIR EQ 都把 click 从 0.27% 推到 0.47~0.55%，
撤掉后 click 归零且评分不降反升（音色 0.941 > 0.938）。

男声 EQ 更狠（EQ_MALE_TMPL 有 4 个窄带：90Hz +7.5dB / 350Hz +3 /
1400Hz +3 / 250Hz +0.45），但它是为修「60~150Hz 偏低」「感冒闷」
等已知问题调的，不能照搬女声结论直接撤 —— 必须跑 A/B 用数据说话。

⚠️ 男声走**变调**（DEFAULT_F0 m=188.0），所以两边都跑完整 fix_chain
   （不能加 --no-pitch），只有 EQ 不同。

用法：python3 maleq47.py [--chars 1500]
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")

OUT = "/mnt/tts/diag/male47"
SR = 16000
MALE_JSON = "/mnt/tts/data/eq_male.json"


def load(p):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ar", str(SR),
                        "-ac", "1", "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def norm(x, peak=0.70):
    m = np.abs(x).max()
    return x * (peak / m) if m > 1e-9 else x


def click(x, thr=0.30):
    n = int(SR * 0.02)
    d = np.abs(np.diff(x))
    m = (len(d) // n) * n
    dmax = d[:m].reshape(-1, n).max(axis=1)
    return int((dmax >= thr).sum()), len(dmax)


def main():
    chars = int(sys.argv[sys.argv.index("--chars") + 1]) \
        if "--chars" in sys.argv else 1500
    os.makedirs(OUT, exist_ok=True)

    with open("/mnt/tts/data/cases/6cdd25bc75f2/seg0.txt",
              encoding="utf-8") as f:
        text = f.read().strip()[:chars]
    print(f"文本 {len(text)} 字（男声 case 6cdd25bc75f2/seg0.txt）")

    import soundfile as sf
    import vox_client

    print("调 VoxCPM2 生成男声裸输出 ...")
    audio, sr, meta = vox_client.synth(text=text, voice="male")
    if audio is None:
        print("生成失败:", meta)
        return 1
    raw = os.path.join(OUT, "raw.wav")
    sf.write(raw, audio, sr)
    print(f"  裸输出 {len(audio)/sr:.2f}s @ {sr}Hz")

    res = {}

    # ---- A：带 EQ（现役口径，不写 eq_male.json） ----------------------
    if os.path.exists(MALE_JSON):
        os.rename(MALE_JSON, MALE_JSON + ".bak-male47")
    a_out = os.path.join(OUT, "A_witheq.wav")
    subprocess.run([sys.executable,
                    "/mnt/tts/projects/qwen3-tts/app/fix_chain.py",
                    raw, a_out, "--gender", "m", "--chars", str(len(text))],
                   check=True)
    res["A 带EQ(现役)"] = a_out

    # ---- B：不带 EQ --------------------------------------------------
    # 🔴 不能用 eq_male.json = {"eq": ""}：eq_male_tmpl() 里有
    #    `or EQ_MALE_TMPL` 兜底，空串会被吃掉、照样返回内置模板
    #    （实测 A/B 完全相同就是这么来的）。必须走命令行 --eq anull：
    #    `_plain = a.eq if a.eq else ...` 且 `_adaptive = (not a.eq) and ...`
    #    → _adaptive 变 False、_eq = "anull"，才真正绕开。
    b_out = os.path.join(OUT, "B_noeq.wav")
    subprocess.run([sys.executable,
                    "/mnt/tts/projects/qwen3-tts/app/fix_chain.py",
                    raw, b_out, "--gender", "m", "--chars", str(len(text)),
                    "--eq", "anull"], check=True)
    res["B 不带EQ"] = b_out

    # ---- 测量 --------------------------------------------------------
    print(f"\n{'样本':<18}{'click窗':>8}{'占比':>9}{'峰值':>9}")
    print("-" * 46)
    for tag in ["raw 裸输出", "A 带EQ(现役)", "B 不带EQ"]:
        p = raw if tag.startswith("raw") else res[tag]
        x = load(p)
        k, tot = click(norm(x))
        print(f"{tag:<18}{k:>8}{k/tot*100:>8.2f}%"
              f"{np.abs(x).max():>9.4f}")

    print("\n✅ 男声 A/B 产出：", json.dumps(res, ensure_ascii=False))
    print("⚠️ eq_male.json 当前是「不带 EQ」状态，定案后要决定是否保留")
    return 0


if __name__ == "__main__":
    sys.exit(main())
