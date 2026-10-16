#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""loc47.py —— 员工点名的**精确秒**定位诊断（整条成品口径）。

员工反馈（听整条成品，不是 20s 片段）：
  · sola47_f_omnivoice（现役）  「11 秒 卡音，25 秒 重复音」
  · sola47_f_gptsovits（GS）    「也有杂音」（没给秒 → 全篇扫）

本脚本回答三件事：
  ① 卡音：波形硬切 / 削波 / 能量突降 / F0 断裂
  ② 重复音：ASR 文本重复 + 短 lag 波形自相似
  ③ 杂音：全篇扫 click（相邻样本跳变）+ 窄带尖峰

用法：
  python3 loc47.py --wav <转好的 wav> [--windows 11,25] [--scan]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

SR = 16000


def load_wav(path, sr=SR):
    """用 ffmpeg 读成 float32 mono（不信任 soundfile 的 mp3 支持）。"""
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ar", str(sr),
         "-ac", "1", "-f", "f32le", "-"],
        capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32)


def frame_rms(x, sr=SR, win_ms=10.0):
    n = max(1, int(sr * win_ms / 1000.0))
    m = (len(x) // n) * n
    return np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))


def db(v):
    return 20.0 * np.log10(max(v, 1e-12))


# ---------------------------------------------------------------- ① 卡音
def click_scan(x, sr=SR, topn=12, exclude_ms=0.0):
    """相邻样本跳变（|Δ|）——硬切/爆音的经典指纹。

    语音本身在 16k 下相邻样本差一般 < 0.05（高频段更陡但也就 0.1 量级）。
    |Δ| > 0.3 基本就是非自然的不连续。
    """
    d = np.abs(np.diff(x))
    n = int(sr * 0.02)          # 20ms 窗，取窗内最大跳变
    m = (len(d) // n) * n
    dmax = d[:m].reshape(-1, n).max(axis=1)
    idx = np.argsort(dmax)[::-1][:topn]
    out = []
    for i in sorted(idx):
        t = i * n / sr
        if exclude_ms and t * 1000 < exclude_ms:
            continue
        out.append((t, float(dmax[i])))
    return out, dmax, n


def clip_ratio(x, thr=0.99):
    return float((np.abs(x) >= thr).mean())


def window_report(x, sr, t0, t1, tag):
    a, b = int(t0 * sr), int(t1 * sr)
    seg = x[a:b]
    if len(seg) < sr * 0.2:
        return None
    r = frame_rms(seg, sr, 10.0)
    cl, _, _ = click_scan(seg, sr, topn=5)
    # 能量突降：相邻 10ms 帧的 dB 落差
    dr = np.diff(20 * np.log10(np.maximum(r, 1e-12)))
    j_drop = int(np.argmin(dr))
    print(f"\n===== {tag}  [{t0:.1f}s ~ {t1:.1f}s] =====")
    print(f"  RMS 中位  {db(np.median(r)):7.1f} dB   "
          f"p5 {db(np.percentile(r,5)):7.1f}   p95 {db(np.percentile(r,95)):7.1f}")
    print(f"  最大 10ms 落差  {dr.min():6.1f} dB @ "
          f"{t0 + j_drop*0.01:.2f}s （往下掉=卡顿/断口）")
    print(f"  削波(>=0.99)  {clip_ratio(seg)*100:.4f} %")
    print("  窗内 click top5:")
    for t, v in cl:
        print(f"     {t0+t:7.3f}s   |Δ|={v:.4f}")
    # 逐 100ms 的能量轮廓，肉眼找断口
    rr = frame_rms(seg, sr, 100.0)
    s = "  ".join(f"{db(v):.0f}" for v in rr)
    print(f"  100ms 轮廓(dB): {s}")
    return seg


# ---------------------------------------------------------------- ② 重复音
def self_sim(seg, sr, lo=0.15, hi=1.20, step=0.01, win=0.40):
    """短 lag 自相似：把 seg 与它自己延迟 lag 后的片段比余弦。
    重复念同一个词 → 在 lag = 那个词的时长处出现尖峰。"""
    n = int(win * sr)
    best = []
    lags = np.arange(lo, hi, step)
    for lag in lags:
        k = int(lag * sr)
        if k + 2 * n > len(seg):
            break
        a = seg[:n]
        b = seg[k:k + n]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        best.append((float(lag), float(a @ b / (na * nb))))
    best.sort(key=lambda z: -z[1])
    return best[:6]


def asr_text(path):
    try:
        import whisper
        m = whisper.load_model("small")
        r = m.transcribe(path, language="ko", fp16=False)
        return (r.get("text") or "").strip()
    except Exception as e:                                  # noqa: BLE001
        return f"[ASR 不可用: {type(e).__name__}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp3", required=True)
    ap.add_argument("--windows", default="11,25")
    ap.add_argument("--scan", action="store_true",
                    help="全篇扫 click")
    ap.add_argument("--half", type=float, default=2.0,
                    help="每个点名秒左右取多少秒")
    ap.add_argument("--outdir", default="/mnt/tts/diag/loc47")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    print(f"读取 {a.mp3} ...")
    x = load_wav(a.mp3)
    print(f"  总长 {len(x)/SR:.2f}s  峰值 {np.abs(x).max():.4f}  "
          f"削波占比 {clip_ratio(x)*100:.4f}%")

    for w in [float(s) for s in a.windows.split(",") if s.strip()]:
        t0 = max(0.0, w - a.half)
        t1 = min(len(x) / SR, w + a.half)
        seg = window_report(x, SR, t0, t1, f"点名 {w:.0f}s")
        if seg is None:
            continue
        # 重复音自相似
        ss = self_sim(seg, SR)
        print("  短 lag 自相似 top6（>0.90 才算重复）:")
        for lag, v in ss:
            print(f"     lag={lag:.2f}s  cos={v:.4f}")
        # 导出片段供人工/ASR
        out = os.path.join(a.outdir, f"w{int(w)}_{t0:.1f}_{t1:.1f}.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.mp3,
                        "-ss", str(t0), "-t", str(t1 - t0),
                        "-ar", "16000", "-ac", "1", out], check=True)
        print(f"  ASR: {asr_text(out)}")
        print(f"  片段: {out}")

    if a.scan:
        print("\n===== 全篇 click 扫描（|Δ| top15） =====")
        cl, dmax, n = click_scan(x, SR, topn=15)
        for t, v in cl:
            print(f"  {t:8.3f}s   |Δ|={v:.4f}")
        thr = 0.30
        bad = np.where(dmax >= thr)[0]
        print(f"  |Δ|>={thr} 的 20ms 窗共 {len(bad)} 个 / {len(dmax)} "
              f"({len(bad)/max(1,len(dmax))*100:.2f}%)")


if __name__ == "__main__":
    sys.exit(main())
