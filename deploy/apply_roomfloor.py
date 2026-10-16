#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修「断句有异声」：填平**短**死寂段，保留长换气

## 依据（dead_air.py，与 EL 长篇直接对比，不是跟真人猜）
    死寂段（相对自身底噪 −20dB 以下、≥100ms）
        EL 长篇  763.5s：34 处  1.4%   2.7 处/分钟
        我们长篇 1341.2s：265 处 3.4%   11.9 处/分钟   ← 4.4 倍
    前落差 29 vs 26 dB、中位时长 155 vs 148ms → **接近，不是问题**
→ 差异在**数量**不在深度。真人和 EL 的长死寂是「换气」，间隔规律；
  我们多出来的那批是**句间随机掉到数字零**，就是员工听到的「断句异声」。

## 做法
只填平**短**死寂段（默认 <200ms）→ 让它回到正常停顿底噪；
**长**死寂段（≥200ms）原样保留 → 那是换气，真人和 EL 都有。
填充用粉噪，电平 = 底噪 −10dB，段边界 20ms 淡入淡出（防硬边）。
逐样本 `where(|noise| > |x|, noise, x)`：只顶替更安静的样本，不动任何语音。
🔴 时长严格不变（改时长 = 改语速）。

## 已知坑（别重踩）
- ❌ 全局抬底噪：会让 probe_g 的相对阈值跟着抬，指标失真（2026-10-15 已证伪）。
  本脚本只动 <200ms 且 <底噪−15dB 的段，占比 ~2%，p10 阈值基本不动。
- ❌ 软噪声门压拖尾（apply_tailcut.py 试过）：release 反而 117→152ms，
  且整体电平被压低 2.6dB 会加重吃字。已废弃。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python apply_roomfloor.py [输入] [输出]
      MAXFILL_MS=200 FILL_REL=-10
"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

HOP = 0.005
TMP = "/mnt/tts/diag/_roomfloor"


def load16(path, tag="x"):
    os.makedirs(TMP, exist_ok=True)
    out = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ac", "1", "-ar", "16000", out], check=True)
    return sf.read(out, dtype="float32")


def env(y, sr):
    hop = int(HOP * sr)
    n = len(y) // hop
    e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    return 20 * np.log10(e + 1e-12)


def runs(mask):
    d = np.diff(mask.astype(np.int8))
    st = list(np.where(d == 1)[0] + 1)
    en = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        st.insert(0, 0)
    if mask[-1]:
        en.append(len(mask))
    m = min(len(st), len(en))
    return [(st[i], en[i]) for i in range(m)]


def dead_stats(path, tag, off=20.0):
    y, sr = load16(path, tag)
    dur = len(y) / sr
    db = env(y, sr)
    floor = np.percentile(db, 10)
    seg = [(s, e) for s, e in runs(db < floor - off) if (e - s) * HOP >= 0.10]
    tot = sum((e - s) * HOP for s, e in seg)
    print("  %-10s 死寂 %3d 处 %5.1fs (%.1f%%)  %.2f 处/分钟  底噪 %.1fdB"
          % (tag, len(seg), tot, 100 * tot / dur, len(seg) / (dur / 60.0), floor))
    return len(seg) / (dur / 60.0), 100 * tot / dur


def pink(n, rng):
    """Voss-McCartney 简化：白噪过一阶低通，够用且快"""
    w = rng.standard_normal(n)
    b = np.convolve(w, np.ones(64) / 64.0, mode="same")
    return b / (np.abs(b).max() + 1e-12) * 0.6


def pick_roomtone(y, sr, db, floor):
    """从**自身**挑一段真实停顿当房间底噪（频谱自洽，比粉噪自然得多）。

    条件：电平落在 floor±3dB 的窄带内（真停顿，不是语音也不是死寂），
    取最长的一段，返回其样本 + 实际 RMS。
    """
    band = (db > floor - 3) & (db < floor + 3)
    cands = [(s, e) for s, e in runs(band) if (e - s) * HOP >= 0.30]
    if not cands:
        return None, None
    s, e = max(cands, key=lambda t: t[1] - t[0])
    a, b = int(s * HOP * sr), int(e * HOP * sr)
    seg = y[a:b]
    return seg, float(np.sqrt((seg ** 2).mean()))


def process(src, dst, maxfill_ms=200.0, fill_rel=-10.0):
    y, sr = load16(src, "src")
    db = env(y, sr)
    floor = np.percentile(db, 10)
    thr_db = floor - 15.0
    amp = 10 ** ((floor + fill_rel) / 20.0)
    rng = np.random.default_rng(1234)
    rt, rt_rms = pick_roomtone(y, sr, db, floor)
    mode = "roomtone"
    if rt is None:
        mode = "pink"
        print("  !! 找不到真实停顿段（≥300ms 且电平在底噪±3dB），退回粉噪")

    i0 = int(0.10 / HOP)
    filled = 0
    seg_all = [(s, e) for s, e in runs(db < thr_db) if (e - s) * HOP >= 0.10]
    for s, e in seg_all:
        if (e - s) * HOP * 1000 > maxfill_ms:
            continue
        a, b = int(s * HOP * sr), int(e * HOP * sr)
        n = b - a
        seg_y = y[a:b]
        if mode == "roomtone":
            off = rng.integers(0, max(1, len(rt) - n))
            nz = rt[off:off + n].astype(np.float32)
            if len(nz) < n:
                nz = np.tile(rt, n // len(rt) + 2)[:n]
            nz = nz * (amp / (rt_rms + 1e-12))
        else:
            nz = pink(n, rng) * amp
        # 20ms 淡入淡出，避免填出硬边
        fw = min(int(0.02 * sr), n // 2)
        if fw > 1:
            w = np.ones(n)
            w[:fw] = np.linspace(0, 1, fw)
            w[-fw:] = np.linspace(1, 0, fw)
            nz = nz * w
        y[a:b] = np.where(np.abs(nz) > np.abs(seg_y), nz, seg_y)
        filled += 1

    peak = float(np.abs(y).max())
    if peak > 0.99:
        y = y * (0.99 / peak)
    sf.write(os.path.join(TMP, "out.wav"), y, sr)
    if dst.endswith(".mp3"):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(TMP, "out.wav"),
                        "-b:a", "192k", dst], check=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(TMP, "out.wav"),
                        dst], check=True)
    return filled, len(seg_all)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
    dst = sys.argv[2] if len(sys.argv) > 2 else "/mnt/tts/diag/_roomfloor/out.mp3"
    maxfill = float(os.environ.get("MAXFILL_MS", "200"))
    fill_rel = float(os.environ.get("FILL_REL", "-10"))

    print("=== 填平短死寂（<%gms，填到底噪%gdB）===" % (maxfill, fill_rel))
    print("处理前:")
    dead_stats(src, "原状")
    n_f, n_all = process(src, dst, maxfill, fill_rel)
    print("  填平 %d / 共 %d 处死寂段" % (n_f, n_all))
    print("处理后:")
    dead_stats(dst, "处理后")
    print("  参考目标：EL 长篇 2.7 处/分钟 / 1.4%")
    print("\nROOMFLOOR_DONE")
    print(dst)


if __name__ == "__main__":
    main()
