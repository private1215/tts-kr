#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第30轮：电子音候选修法 —— 补回发声区的噪声细节（调参）

## 依据
员工说「电子音 / 吃字 / 卡字」，而客观指标里**只有平坦度还有巨大偏差**：
    EL 男标杆 0.0492   裸模型 0.0345   现役成品 0.0294（融合后 0.0317）
频段能量分布已经和 EL 对齐（各频段 ±13% 内），所以不是"缺某个频段"，
而是**同频段内过于周期/干净** —— 缺少真实语音该有的送气噪声与摩擦噪声。

送气/摩擦噪声正是韩语辅音（ㅅ/ㅊ/ㅈ/ㅋ）的载体 → 缺它就会「吃字」；
过渡处缺噪声垫底就会「卡字」；整体缺噪声细节就是「电子音」。
**三个症状同一个根因。**

## 做法
按**局部包络**调制地补一层噪声（只在有声处，停顿处不受影响 → 不会有底噪嘶声）：
    y' = y + noise(t) · env(t) · k
- env 用 10ms RMS 平滑，噪声跟着声音走
- 两种噪声谱：pink（全频带）/ hf（高通 800Hz，更像送气与摩擦）
- k = 噪声 RMS / 局部语音 RMS。真实语音送气噪声约 -15~-25dB → k≈0.06~0.18

## 风控
- 停顿处 env≈0 → 不铺底噪，不会变成"沙沙声"
- k 上限 0.20（超过就是明显嘶声）
- 必须 A/B 让员工裁决，不自己拍板上线

## 用法
    ./venv/bin/python tools/detail_tune.py
"""
import json
import os
import subprocess

import numpy as np

SR = 16000
DIAG = "/mnt/tts/diag/muffle"
OUT = os.path.join(DIAG, "detail_tune.json")
SRC = "/mnt/tts/diag/l1_fuse_new.wav"      # 融合后的男声（38.2s）
EL = "/mnt/tts/diag/muffle/el24.wav"

try:
    import librosa
except Exception:
    librosa = None


def load16(p, t=None):
    c = ["ffmpeg", "-v", "error", "-i", p, "-ac", "1", "-ar", str(SR)]
    if t:
        c += ["-t", str(t)]
    c += ["-f", "f32le", "-"]
    raw = subprocess.run(c, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def flat(y):
    return float(librosa.feature.spectral_flatness(
        y=y[:60 * SR], hop_length=256)[0].mean())


def band(y):
    n, hop = 2048, 512
    nfr = (len(y) - n) // hop
    idx = np.arange(nfr)[:, None] * hop + np.arange(n)[None, :]
    fr = y[idx] * np.hanning(n)[None, :]
    sp = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / SR)
    tot = sp.sum()
    return {lo: float(sp[:, (f >= lo) & (f < hi)].sum() / tot * 100)
            for lo, hi in [(150, 300), (500, 1000), (2000, 4000), (4000, 8000)]}


def pause_rms_db(y, ref, thr_db=-45.0):
    """停顿区（相对 ref 低于阈值的帧）里的噪声电平 —— 检查有没有漏进停顿。"""
    n, hop = 2048, 512
    nfr = min((len(y) - n) // hop, (len(ref) - n) // hop)
    idx = np.arange(nfr)[:, None] * hop + np.arange(n)[None, :]
    e = np.sqrt((y[idx] ** 2).mean(axis=1))
    er = np.sqrt((ref[idx] ** 2).mean(axis=1))
    m = er < (er.max() * 10 ** (thr_db / 20.0))
    if m.sum() == 0:
        return None
    return float(20 * np.log10(np.sqrt((e[m] ** 2).mean()) + 1e-12))


def pink(n, rng):
    m = 1 << int(np.ceil(np.log2(max(n, 2))))
    w = np.fft.rfft(rng.standard_normal(m))
    f = np.fft.rfftfreq(m, 1.0)
    amp = np.ones_like(f)
    lo = f < 100.0
    amp[lo] = 0.5 * (1.0 - np.cos(np.pi * f[lo] / 100.0))
    hi = f > 3000.0
    amp[hi] = 0.35 * (1.0 - 0.5 * (1.0 + np.cos(
        np.pi * np.clip((f[hi] - 3000.0) / 3000.0, 0, 1))))
    return np.fft.irfft(w * amp, n=m)[:n]


def hp(x, sr, fc=800.0, order=4):
    """简单零极点做不到的，用 FFT 高通（线性相位、无相位失真）。"""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    g = 1.0 / (1.0 + (fc / np.maximum(f, 1e-6)) ** (2 * order)) ** 0.5
    g[0] = 0.0
    return np.fft.irfft(X * g, n=len(x))


def env_rms(x, win_ms=10.0):
    n = max(1, int(SR * win_ms / 1000.0))
    nfr = len(x) // n
    e = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    xs = (np.arange(nfr) + 0.5) * n
    return np.interp(np.arange(len(x)), xs, e, left=e[0], right=e[-1])


def add_detail(x, k, rng, kind="hf"):
    e = env_rms(x)
    nz = pink(len(x), rng)
    if kind == "hf":
        nz = hp(nz, SR, 800.0)
    nz = nz / (np.sqrt((nz ** 2).mean()) + 1e-20)
    return (x + nz * e * k).astype(np.float32)


def main():
    src = load16(SRC)
    dur = len(src) / SR
    el = load16(EL, t=dur)
    print("素材 %.1fs   源=%s" % (dur, SRC))
    f_el, b_el = flat(el), band(el)
    f0, b0 = flat(src), band(src)
    print("\nEL  平坦度 %.5f  %s" % (f_el, {k: round(v, 3) for k, v in b_el.items()}))
    print("源  平坦度 %.5f  %s" % (f0, {k: round(v, 3) for k, v in b0.items()}))

    rows = []
    print("\n%-6s %-6s %9s %9s %9s %9s %10s"
          % ("噪声", "k", "平坦度", "b150_300", "b2000_4k", "b4k_8k", "停顿区dB"))
    for kind in ("pink", "hf"):
        for k in (0.06, 0.10, 0.15, 0.20):
            rng = np.random.default_rng(7)
            y = add_detail(src, k, rng, kind)
            b = band(y)
            pd = pause_rms_db(y, src)
            rows.append({"kind": kind, "k": k, "flat": flat(y),
                         "bands": b, "pause_db": pd})
            print("%-6s %-6.2f %9.5f %9.3f %9.3f %9.3f %10s"
                  % (kind, k, flat(y), b[150], b[2000], b[4000],
                     "%.1f" % pd if pd else "-"))

    print("\n（EL 目标：平坦度 %.5f，停顿区基准见下）" % f_el)
    print("EL 停顿区 %.1f dB / 源停顿区 %.1f dB"
          % (pause_rms_db(el, el) or -99, pause_rms_db(src, src) or -99))
    best = min(rows, key=lambda r: abs(r["flat"] - f_el))
    print("\n最接近 EL：%s k=%.2f → 平坦度 %.5f（EL %.5f）"
          % (best["kind"], best["k"], best["flat"], f_el))

    json.dump({"el_flat": f_el, "src_flat": f0, "rows": rows},
              open(OUT, "w"), indent=1, ensure_ascii=False)
    print("→ " + OUT)


if __name__ == "__main__":
    main()
