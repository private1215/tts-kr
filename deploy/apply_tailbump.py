#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第七个方案：只压「**紧邻停顿前、且明显高于句中**的短段」

## 判据为什么又改了（前六次的教训，全在下面）
z43 三票全 D，디아 的自由文本是钥匙：**「断句有异声。音量大小有变化」**
—— 这其实就是在描述问题本身（音量忽大忽小），不是批评压制。

回头看 95.5s 的结构（tiny_asr 10ms 包络）：
    95.10~95.45  −35dB   句中的延续
    **95.50~95.70  −22dB   句末最后一个音节，比句中响 13dB**   ← 员工听到的「额」
    95.73~95.90  −72dB   紧接停顿（死寂）
    95.90~        −34dB   下一句
真人的句末音节是**降调 + 减弱**，我们反而是**顶起来**的。
→ 判据：**紧邻停顿（≥80ms 静音）之前的短段，若比句中参考电平高 >6dB 就压平**。

## 已否决的六条（别重踩）
| # | 方案 | 怎么翻车的 |
|---|---|---|
| 1 | 软噪声门压句末拖尾 | release 117→152ms、电平掉 2.6dB |
| 2 | 填房间底噪 | 死寂达标但 Σz² 32.1→34.7、平坦度翻向 −4.30σ |
| 3 | 句末淡出 | 命中 0 段 / 要改 397 段 |
| 4 | 元音起音柔化 | z10 三票全「差不多/都不行」 |
| 5 | 分段压顶起（全篇>8dB） | 指标最好（Σz² 22.6、平坦度 +0.20σ）但邻位冒头、留音量台阶 |
| 6 | 迭代压顶起 | 顶起降到 6.9/分但 **Σz²→72.0、发声占比 −5.79σ**（第二轮压的是正常语音） |
| 6b | 连续平滑增益 | 无台阶但太弱（25.1→21.4），压不动 +13dB 的突起 |

## 对照基线
顶起密度：真人 30.5 / **EL 8.7** / 我们 25.1（处每分）。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python apply_tailbump.py
"""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

SR = 24000
HOP = 0.005


def _ab():
    spec = importlib.util.spec_from_file_location("ab", "/mnt/tts/diag/apply_bump.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ab"] = m
    spec.loader.exec_module(m)
    return m


AB = _ab()
TMP = "/mnt/tts/diag/_tailbump"
os.makedirs(TMP, exist_ok=True)
SRC = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
MARKS = [("题1 5s", 95.5), ("题2 1s", 131.9), ("题3 10s", 170.5)]


def load(src, tag="x"):
    out = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ar", str(SR),
                    "-ac", "1", out], check=True)
    return sf.read(out, dtype="float32")


def env_db(y, sr):
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


def find_tailbumps(y, sr, need_db=6.0, minlen=0.08, maxlen=0.40, gap_ms=80):
    db = env_db(y, sr)
    floor = np.percentile(db, 10)
    on = db > floor + 6
    seg = runs(on)
    pauses = [(s, e) for s, e in runs(~on) if (e - s) * HOP * 1000 >= gap_ms]
    out = []
    for ps, pe in pauses:
        # 停顿前的那个发声段
        prev = [i for i, (s, e) in enumerate(seg) if e <= ps]
        if not prev:
            continue
        i = prev[-1]
        s, e = seg[i]
        ln = (e - s) * HOP
        if not (minlen <= ln <= maxlen):
            continue
        a, b = int(s * HOP * sr), int(e * HOP * sr)
        lvl = 20 * np.log10(np.sqrt((y[a:b] ** 2).mean()) + 1e-12)
        # 句中参考：再往前一个发声段（或前两段的中位）
        refs = []
        for j in prev[:-1][-2:]:
            a2, b2 = int(seg[j][0] * HOP * sr), int(seg[j][1] * HOP * sr)
            refs.append(20 * np.log10(np.sqrt((y[a2:b2] ** 2).mean()) + 1e-12))
        if not refs:
            continue
        ref = float(np.median(refs))
        if lvl - ref >= need_db:
            out.append((s * HOP, ln, lvl, ref))
    return out, floor


def apply(src, dst, need_db=6.0, target_db=2.0):
    y, sr = load(src, "src")
    hits, floor = find_tailbumps(y, sr, need_db)
    for t, ln, lvl, ref in hits:
        a, b = int(t * sr), int((t + ln) * sr)
        g = 10 ** ((ref + target_db - lvl) / 20.0)
        if g >= 0.999:
            continue
        seg_y = y[a:b].copy()
        m = len(seg_y)
        fw = min(int(0.025 * sr), m // 2)
        w = np.ones(m)
        w[fw:m - fw] = g
        if fw > 1:
            w[:fw] = np.linspace(1.0, g, fw)
            w[-fw:] = np.linspace(g, 1.0, fw)
        y[a:b] = seg_y * w
    pk = float(np.abs(y).max())
    if pk > 0.99:
        y = y * (0.99 / pk)
    sf.write(os.path.join(TMP, "o.wav"), y, sr)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(TMP, "o.wav"),
                    "-b:a", "192k", dst], check=True)
    return len(hits)


def main():
    y, sr = load(SRC, "probe")
    for need in (4.0, 6.0, 8.0):
        hits, floor = find_tailbumps(y, sr, need)
        print("判据 need>%gdB：命中 %3d 处（底噪 %.1f）" % (need, len(hits), floor))
        if need == 6.0:
            for t, ln, lvl, ref in hits:
                if any(abs(t - m) < 3.0 for _, m in MARKS):
                    print("    %7.2fs %5.0fms %.1fdB (句中 %.1fdB, 高出 %.1fdB)  <<<员工点名"
                          % (t, ln * 1000, lvl, ref, lvl - ref))
    # 对照：真人原声
    for tag, p in [("真人原声", "/mnt/tts/data/cases/52727ec43154/merged.wav"),
                   ("EL长篇", "/mnt/tts/diag/ab/c_el_up.wav")]:
        yy, ss = load(p, tag[:3])
        h, _ = find_tailbumps(yy, ss, 6.0)
        # 归一到每 100 字（两者篇幅不同）
        print("  %-8s 句末突起 %3d 处 / %.0fs = %.1f 处每分"
              % (tag, len(h), len(yy) / ss, len(h) / (len(yy) / ss / 60)))

    print("\n=== 压制对照 ===")
    for tgt in (0.0, 2.0):
        dst = os.path.join(TMP, "tb_t%g.mp3" % tgt)
        n = apply(SRC, dst, 6.0, tgt)
        yy, ss = AB.load(dst, "chk")
        b, _, _, _ = AB.find_bumps(yy, ss, 8.0, maxlen=0.35, ctx_win=0.4)
        dur = len(yy) / ss
        res = []
        for nm, mk in MARKS:
            near = [t for t, l, lv, c in b if abs(t - mk) < 2.0]
            res.append("%s=%s" % (nm, "%.1fs" % near[0] if near else "消除"))
        print("  压到 句中+%gdB：处理 %d 处 -> 顶起密度 %.1f/分  %s"
              % (tgt, n, len(b) / (dur / 60), " ".join(res)))
    print("\nTAILBUMP_DONE")


if __name__ == "__main__":
    main()
