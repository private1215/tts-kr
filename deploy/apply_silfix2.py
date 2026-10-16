#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""句间停顿「地板抬起」—— 让停顿像 EL 一样平躺在房间底噪上

## 机理（2026-10-16 定案，测出来的，不是猜的）
| | EL 女声原声 52727ec | 我们现役成品 |
|---|---|---|
| 停顿帧中位 | −51.8dB | −51.9dB ← **完全一样** |
| p5 | −60.5dB | **−73.5dB** ← 我们的尾巴深 13dB |
| p10 | −58.7dB | −62.4dB |
| min | −98.5dB | **−240dB**（数字零） |
| <−100dB 帧 | **0** | 32 |
| 深谷(<−60dB,≥40ms) | 8 段（-93~-96dB） | 20 段（中位 −76.7dB） |

更关键的一格：**在我方音频里找不到任何一段 ≥250ms 的连续帧稳定停在 −56~−46dB。**
→ EL 的停顿是"衰减到 −52 就**平躺**在房间底噪上"；
→ 我们的停顿"穿过 −52 继续**往下沉**"，沉到 −73，甚至沉成数字零。
沉下去之后语音再起来，落差从 30dB 变成 50~220dB —— 听感上第一个音节就是
**"突然蹦出来"**。员工点名 3 处（95.5s / 131.9s / 170.5s，就是 z43 题1@5s、
题2@1s、题3@10s）全部落在这种下陷谷上。

## 前九次为什么没修好（逐条对账）
① 前八次改的都是**可听内容**（拖尾 / 起音 / 顶起幅度），方向就不对。
② 第九次 apply_silfix 找到位置了，但自己有两个 bug：
   - `pick_roomtone` 找不到 ≥250ms 的 −56~−46dB 段（因为上面那条：我们就没有）
     → 静默退回 `pink` 分支；
   - `pink()` 是**峰值归一**（峰值 0.6），乘 amp 后实际 RMS 只有 −68dB
     → "填了"但填进去的是 −68dB 的噪声，比原来还深（95.5s −73 → −78）。
   - 淡入淡出只写了 `nz*w`，没把原信号 `sy*(1-w)` 混回来 → 在填补段两端亲手挖坑。

## 本版三条硬规矩（写死，别再犯）
1. **加法式地板**：`out = sy + nz*g`。只会抬高，绝不压低 → 物理上不可能挖出新坑。
   （`keep`/`where(|nz|>|sy|)` 那种"取较大"写法一律禁用 —— 粉噪过零会漏，
   和 app/declick.py 的漏洞同款，已经踩过一次。）
2. **房间底噪自造**：优先用自身落在 −56~−46dB 的帧拼接；不够就放宽到 −58~−44；
   再没有才用粉噪，但**必须按 RMS 归一**到目标电平（不是峰值）。
3. **只认真正的停顿**：段长 ≥60ms，且比前后各 300ms 的上下文低 >12dB。
   否则会往"轻读的音节"上加噪声，变成新的顶起。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python apply_silfix2.py
"""
import os
import subprocess

import numpy as np
import soundfile as sf

SR = 24000
HOP = 0.005
TMP = "/mnt/tts/diag/_silfix2"
os.makedirs(TMP, exist_ok=True)
SRC = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
MARKS = [95.5, 99.0, 112.0, 131.9, 170.5]


def load(src, tag):
    o = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1",
                    "-ar", str(SR), o], check=True)
    return sf.read(o, dtype="float32")[0]


def env_db(y):
    h = int(HOP * SR)
    n = len(y) // h
    e = np.sqrt(np.array([(y[i * h:(i + 1) * h] ** 2).mean() for i in range(n)]))
    return 20 * np.log10(e + 1e-12)


def runs(m):
    d = np.diff(m.astype(np.int8))
    st = list(np.where(d == 1)[0] + 1)
    en = list(np.where(d == -1)[0] + 1)
    if m[0]:
        st.insert(0, 0)
    if m[-1]:
        en.append(len(m))
    k = min(len(st), len(en))
    return [(st[i], en[i]) for i in range(k)]


def build_roomtone(y, db, floor_db):
    """自造房间底噪：优先用自身落在目标带内的帧拼接，最后才用 RMS 归一的粉噪。"""
    for lo, hi in [(floor_db - 4, floor_db + 6), (floor_db - 8, floor_db + 8),
                   (floor_db - 12, floor_db + 10)]:
        idx = np.where((db > lo) & (db < hi))[0]
        if len(idx) * HOP < 0.3:
            continue
        parts = []
        for i in idx[:4000]:
            parts.append(y[i * int(HOP * SR):(i + 1) * int(HOP * SR)])
        seg = np.concatenate(parts).astype(np.float32)
        print("  房间底噪：取自身 %d 帧(%.2fs) 于 %.0f~%.0fdB 带内" %
              (len(idx), len(idx) * HOP, lo, hi))
        return seg
    rng = np.random.default_rng(20261016)
    w = rng.standard_normal(int(SR * 2))
    b = np.convolve(w, np.ones(64) / 64.0, mode="same")       # 低频加重 = 房间感
    print("  房间底噪：无自身素材，用 RMS 归一粉噪")
    return b.astype(np.float32)


def proc(src, dst, floor_db=-52.0, gate_db=-58.0, minlen_ms=60.0,
         ctx_db=12.0, dil_ms=20.0):
    y = load(src, "src")
    db = env_db(y)
    fh = int(HOP * SR)
    amp = 10.0 ** (floor_db / 20.0)
    rt = build_roomtone(y, db, floor_db)
    rt = rt - rt.mean()
    rt = rt * (amp / (np.sqrt((rt ** 2).mean()) + 1e-12))      # 🔴 按 RMS 归一
    print("  目标地板 %.0fdB → 底噪 RMS 实测 %.1fdB"
          % (floor_db, 20 * np.log10(np.sqrt((rt ** 2).mean()))))

    cand = [(s, e) for s, e in runs(db < gate_db) if (e - s) * HOP * 1000 >= minlen_ms]
    # 文件首尾的补零段：不满足 minlen 也别放过（开头 30ms 数字零就是这么漏下来的）
    edge = set((s, e) for s, e in runs(db < -90) if s == 0 or e >= len(db) - 1)
    cand = sorted(set(cand) | edge)
    n_ok = 0
    for s, e in cand:
        if (s, e) in edge:
            s2, e2 = s, e                      # 首尾段：不膨胀、不检查上下文
        else:
            c0, c1 = max(0, s - int(0.30 / HOP)), min(len(db), e + int(0.30 / HOP))
            ctx = np.concatenate([db[c0:s], db[e:c1]])
            if len(ctx) and np.median(ctx) - np.median(db[s:e]) < ctx_db:
                continue                  # 上下文没高出一截 → 不是停顿，跳过
            # 向外膨胀 dil_ms，让淡入淡出落在"本来就有底噪"的地方
            d = int(dil_ms / 1000.0 / HOP)
            s2, e2 = max(0, s - d), min(len(db), e + d)
        a, b = s2 * fh, e2 * fh
        n = b - a
        sy = y[a:b].copy()
        off = int(np.random.default_rng(s * 7919 + e).integers(0, max(1, len(rt) - n)))
        nz = np.tile(rt, n // max(1, len(rt)) + 2)[off:off + n].astype(np.float32)
        g = np.ones(n, dtype=np.float32)
        fw = max(1, int(dil_ms / 1000.0 * SR))
        g[:fw] = np.linspace(0, 1, fw)
        g[-fw:] = np.linspace(1, 0, fw)
        y[a:b] = sy + nz * g              # 🔴 加法：只会抬高
        n_ok += 1
    pk = float(np.abs(y).max())
    if pk > 0.99:
        y = y * (0.99 / pk)
    sf.write(os.path.join(TMP, "o.wav"), y, SR)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(TMP, "o.wav"),
                    "-b:a", "192k", dst], check=True)
    return n_ok


def report(path, tag):
    y = load(path, tag[:6])
    db = env_db(y)
    lo = db[db < -45]
    p = np.percentile(lo, [1, 5, 10, 50]) if len(lo) else [0] * 4
    vs = runs(db < -60)
    vs = [(s, e) for s, e in vs if (e - s) * HOP >= 0.04]
    dep = [db[s:e].min() for s, e in vs]
    fl = runs((db > -56) & (db < -46))
    flat = max([(e - s) * HOP for s, e in fl] or [0])
    print("  %-12s p1 %6.1f p5 %6.1f p10 %6.1f 中位 %6.1f | min %7.1f | <−100dB %3d "
          "| 深谷 %2d 段(中位 %6.1f) | 最长 −52 平台 %.2fs | RMS %6.1fdB"
          % (tag, p[0], p[1], p[2], p[3], db.min(), int((db < -100).sum()),
             len(vs), float(np.median(dep)) if dep else 0, flat,
             20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-12)))
    print("      点名处最低：%s" % "  ".join(
        "%.1fs=%.0f" % (m, db[int((m - 0.7) / HOP):int((m + 0.7) / HOP)].min())
        for m in MARKS))


def main():
    print("=== 修前 ===")
    report(SRC, "现役成品")
    print("参考：EL女声原声 p1 -90.4 p5 -60.5 p10 -58.7 中位 -51.8 | min -98.5 "
          "| <−100dB 0 | 深谷 8 | RMS -22.8dB")
    for floor_db in (-50.0, -52.0, -55.0, -58.0):
        dst = os.path.join(TMP, "f_%g.mp3" % abs(floor_db))
        print("\n=== 地板 %.0fdB ===" % floor_db)
        n = proc(SRC, dst, floor_db=floor_db)
        print("  处理 %d 段" % n)
        report(dst, "f%g" % abs(floor_db))
    print("\nSILFIX2_DONE")


if __name__ == "__main__":
    main()
