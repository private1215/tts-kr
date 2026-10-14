# -*- coding: utf-8 -*-
"""定位男声 0.953(裸推理) -> 0.844(线上成品) 这 0.109 到底丢在哪一步。

关键事实（2026-10-13 量到）：
    裸推理 cosy_full_m.wav  445.72 s
    线上成品 open_male.wav  490.15 s   -> 慢了 9.97%（= 记忆里的"语速 -9%"）

所以第一嫌疑不是 slowdyn（它只改增益、不改时长），而是**整条链路里那档
atempo 时间拉伸**。本脚本在**裸推理**这条干净基线上逐项叠加，看谁把音色打下去：
    v0 裸推理                     基线
    v1 + atempo 0.9093           时间拉伸（对齐全长）
    v2 + mp3 192k/44.1k 往返      编码
    v3 + slowdyn                  慢尺度能量（纯增益）
    v4 + atempo + slowdyn         两者叠加
    v5 mp3 往返（不 atempo）      单独看编码
对照：ONLINE 线上成品。
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = "/mnt/tts/projects/qwen3-tts"
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.path.insert(0, ROOT)
import spkchk  # noqa: E402

CAMPPLUS = "/mnt/tts/models/Fun-CosyVoice3-0.5B/campplus.onnx"
_SESS = None

REF = "/mnt/tts/data/cases/1a6c20a60481/merged.wav"
BARE = "/mnt/tts/projects/gpt-sovits/out/cosy_full_m.wav"
ONLINE = "/tmp/selftest_open/open_male.wav"

TMP = tempfile.mkdtemp(prefix="gap2_")


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
    return float(a @ b / (np.linalg.norm(a) * (np.linalg.norm(b) + 1e-12)))


def dur(p):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", p],
        capture_output=True, text=True).stdout.strip())


def atempo(src, ratio, tag):
    out = os.path.join(TMP, f"{tag}.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-af", f"atempo={ratio:.6f}", "-ar", "44100", out],
                   check=True)
    return out


def mp3rt(src, tag):
    m = os.path.join(TMP, f"{tag}.mp3")
    out = os.path.join(TMP, f"{tag}_dec.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-c:a", "libmp3lame", "-b:a", "192k", "-ar", "44100", m],
                   check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", m,
                    "-ar", "44100", "-ac", "1", out], check=True)
    return out


def slowdyn(src, tag):
    import soundfile as sf
    import slowdyn
    from prosody_report import sent_windows
    x, sr = sf.read(src)
    if x.ndim > 1:
        x = x.mean(axis=1)
    segs, _, _ = sent_windows(x.astype(np.float32), sr)
    y, ks, kf, diag = slowdyn.apply(x.astype(np.float32), sr, segs=segs,
                                    is_male=True)
    out = os.path.join(TMP, f"{tag}.wav")
    sf.write(out, y.astype(np.float32), sr)
    return out, (ks, kf)


def main():
    d_ref, d_bare, d_on = dur(REF), dur(BARE), dur(ONLINE)
    print(f"时长：参照 {d_ref:.1f}s  裸推理 {d_bare:.1f}s  线上 {d_on:.1f}s")
    ratio = d_bare / d_on
    print(f"裸推理 -> 线上 需要 atempo = {ratio:.6f}（慢 {1/ratio-1:+.2%}）")

    e_ref = emb(REF)
    e_on = emb(ONLINE)
    print(f"\n参照 vs 线上成品 : {cos(e_ref, e_on):.4f}   <- 现状")

    variants = []
    v0 = BARE
    variants.append(("v0 裸推理", v0))
    v1 = atempo(BARE, ratio, "v1_atempo")
    variants.append((f"v1 +atempo {ratio:.4f}", v1))
    v2 = mp3rt(v1, "v2")
    variants.append(("v2 +atempo +mp3", v2))
    v5 = mp3rt(BARE, "v5")
    variants.append(("v5 +mp3(不atempo)", v5))
    try:
        v3, kk = slowdyn(BARE, "v3_slowdyn")
        variants.append((f"v3 +slowdyn {kk[0]:.2f}/{kk[1]:.2f}", v3))
        v4 = atempo(v3, ratio, "v4")
        variants.append(("v4 +slowdyn +atempo", v4))
    except Exception as e:
        print("slowdyn 跳过:", repr(e)[:200])

    print("\n%-26s %-9s %-9s %s" % ("变体", "vs参照", "vs线上", "时长"))
    for name, p in variants:
        e = emb(p)
        print("%-26s %.4f    %.4f    %.1fs"
              % (name, cos(e_ref, e), cos(e_on, e), dur(p)))


if __name__ == "__main__":
    main()
