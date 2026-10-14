# -*- coding: utf-8 -*-
"""员工最新反馈（2026-10-13 01:39 UTC）说「卡顿厉害」，但 10-12 自测说
「卡顿已根治」。两边打架，必须回到**音频本身**量。

本脚本量的是 c 轮那四段 + 参照（EL 标杆原声）：
    fb2_{f,m}_old_head.mp3   旧生产（题目选项 A）
    fb2_{f,m}_new_head.mp3   新生产·可听口径修复版（选项 B）
    fb_*_ref_head.mp3        EL 标杆原声

量的东西（"卡顿"拆成三种可测形态）：
  1. **长间隙**：可听口径静音（20ms 帧、dB>-45 为有声）里 >=0.6s 的段数。
     这是 10-12 修的目标，也**只能**抓到"死寂断档"这一种卡顿。
  2. **静音占比**：同上口径的静音时间 / 总时长。
  3. 🔴 **接缝跳变（"咯噔"一下）**：分段生成时每段独立合成再 concat，
     接缝处会有能量/频谱的硬跳变——**它不是静音，长间隙统计完全抓不到**。
     这里量两样：
       · 能量跳变：相邻 50ms 块的 dB 差，超过 12dB 且前后都有声 = 硬切；
       · 静音边界的"陡度"：gap 前后 30ms 的能量落差（人耳对突然归零最敏感）。
  4. **LUFS**：确认两版响度可比（差 >1dB 则盲测不公平）。

输出一张表 + 每个 gap 的明细（起止/时长/前后电平）。
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

DIR = "/mnt/tts/data/tone-ab"
TMP = tempfile.mkdtemp(prefix="pause_")

# (标签, 文件)
ITEMS = [
    ("女·EL标杆(ref)", "fb_f_ref_head.mp3"),
    ("女·旧生产(A)", "fb2_f_old_head.mp3"),
    ("女·新修复(B)", "fb2_f_new_head.mp3"),
    ("男·EL标杆(ref)", "fb_m_ref_head.mp3"),
    ("男·旧生产(A)", "fb2_m_old_head.mp3"),
    ("男·新修复(B)", "fb2_m_new_head.mp3"),
]

# ---- 可听口径参数（与 tools/case_audible_pauses.py 同源）--------------------
SR = 16000
FRAME = 0.020          # 20 ms
HOP = 0.020
DB_ON = -45.0          # 帧电平 > -45 dB 视为有声
GAP_MIN = 0.25         # 内部静音 >= 0.25s 才算一个 gap
LONG_GAP = 0.60        # >= 0.6s 判为"死寂断档"
JUMP_DB = 12.0         # 相邻 50ms 块能量跳变阈值


def load(path):
    """统一解码到 16k 单声道（可听口径的母版）。"""
    out = os.path.join(TMP, os.path.basename(path) + ".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", str(SR), "-ac", "1", out], check=True)
    x, _ = sf.read(out)
    return np.asarray(x, dtype=np.float64)


def frame_db(x):
    n = int(round(FRAME * SR))
    h = int(round(HOP * SR))
    if len(x) < n:
        return None
    idx = np.arange(0, len(x) - n + 1, h)
    e = np.array([np.sqrt(np.mean(x[i:i + n] ** 2) + 1e-12) for i in idx])
    return 20.0 * np.log10(e), np.arange(len(idx)) * h / float(SR)


def gaps_of(db, t):
    """可听口径的静音段：连续低于阈值的区间，>=GAP_MIN 才算。"""
    voiced = db > DB_ON
    out, i = [], 0
    n = len(voiced)
    while i < n:
        if not voiced[i]:
            j = i
            while j < n and not voiced[j]:
                j += 1
            a, b = t[i], t[j - 1] + FRAME
            if b - a >= GAP_MIN:
                out.append((float(a), float(b)))
            i = j
        else:
            i += 1
    return out


def level_at(db, t, a, b):
    """区间 [a,b) 内的平均电平 dB（用于看 gap 前后的落差）。"""
    m = (t >= a) & (t < b)
    if m.sum() == 0:
        return None
    return float(db[m].mean())


def block_db(x, win=0.05):
    n = int(round(win * SR))
    k = len(x) // n
    if k < 4:
        return None
    e = np.sqrt(np.mean(x[:k * n].reshape(k, n) ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(e)


def lufs(path):
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-af", "loudnorm=print_format=json",
         "-f", "null", "-"], capture_output=True, text=True)
    s = r.stderr
    i = s.rfind("{")
    if i < 0:
        return None
    try:
        return json.loads(s[i:s.rfind("}") + 1]).get("input_i")
    except Exception:
        return None


def analyse(tag, path):
    p = os.path.join(DIR, path)
    if not os.path.exists(p):
        print(f"{tag}: 缺文件 {path}")
        return None
    x = load(p)
    dur = len(x) / SR
    got = frame_db(x)
    if got is None:
        return None
    db, t = got
    gaps = gaps_of(db, t)
    longs = [g for g in gaps if g[1] - g[0] >= LONG_GAP]
    sil = sum(g[1] - g[0] for g in gaps)
    # 接缝跳变
    bd = block_db(x)
    jumps = 0
    if bd is not None:
        d = np.abs(np.diff(bd))
        # 前后都有声（避免把静音边界算进来）：两块电平均 > -45
        ok = (bd[:-1] > DB_ON) & (bd[1:] > DB_ON)
        jumps = int(((d > JUMP_DB) & ok).sum())
    # gap 前后落差（"突然归零"的陡度），取最大
    steep = 0.0
    for a, b in gaps:
        pre = level_at(db, t, max(a - 0.10, 0.0), a)
        if pre is not None:
            steep = max(steep, pre - DB_ON)
    return dict(tag=tag, dur=dur, n_gap=len(gaps), n_long=len(longs),
                sil_pct=100.0 * sil / dur, jumps=jumps, steep=steep,
                lufs=lufs(p), gaps=gaps, longs=longs)


def main():
    rows = []
    for tag, f in ITEMS:
        r = analyse(tag, f)
        if r:
            rows.append(r)
    print("=" * 96)
    print("%-18s %7s %6s %7s %10s %8s %9s %7s"
          % ("片段", "时长", "gap数", "≥0.6s", "静音占比%", "接缝跳变", "最陡落差dB", "LUFS"))
    print("-" * 96)
    for r in rows:
        print("%-18s %7.1f %6d %7d %10.2f %8d %9.1f %7s"
              % (r["tag"], r["dur"], r["n_gap"], r["n_long"], r["sil_pct"],
                 r["jumps"], r["steep"],
                 f'{r["lufs"]}' if r["lufs"] is not None else "-"))
    print("=" * 96)
    print("\n各片段 gap 明细（起止 / 时长 / 前电平→后电平）")
    for r in rows:
        if not r["gaps"]:
            continue
        print(f"\n-- {r['tag']} ({len(r['gaps'])} 个)")
        for a, b in r["gaps"]:
            print("   %.2f-%.2fs  %.2fs" % (a, b, b - a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
