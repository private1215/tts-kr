# -*- coding: utf-8 -*-
"""句级「慢尺度」起伏诊断：查 AI 味的来源。

## 为什么需要它

`tools/drift.py` 用 30 s 等长窗量出来一个五例同向的怪事：

    段落响度 rms_db 的窗间 sd，我们 0.17~0.24 dB，案例 0.53~0.87 dB
    → 我们只有案例的 0.20~0.39 倍

而 voiced（有声占比）两边几乎一样（sd 0.01~0.04）。同样的有声比例、
同样的整体响度，案例的整体 RMS 却在 30 s 尺度上起伏 2~3 dB、我们是直线。

那只能是有声段**本身**的轻重在变 —— 也就是**句与句之间的能量差**。
真人念稿会随内容/情绪自然有轻重，TTS 每句都从同一个参考音频起头、
codec 又归一化，出来就是每句一样响。听感 = 「没感情 / AI 味 / 像读课本」。

30 s 窗会把句间跳变平均掉，所以要**逐句**量：

| 指标 | 抓什么 |
|---|---|
| 句级能量 sd（dB） | **句间轻重差** —— AI 味的头号嫌疑 |
| 句级能量相邻差分 RMS | 是**渐变**（真人）还是**跳变**（逐句重掷） |
| 句级 f0 中位 sd | 句间音高差 —— 「语气就变了」 |
| 句级 f0 相邻差分 RMS | 同上，看是渐变还是跳 |

🔴 同样必须带案例对照组。

用法（服务器，qwen 环境）：
    venv/bin/python tools/sentvar.py --prefix r23
"""
import os
import sys
import glob
import argparse

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "app"))
from app.prosody_report import sent_windows      # noqa: E402
from tools.drift import load, SR, CASES_DIR, OUT_DIR, WORK   # noqa: E402

FIVE = {
    "1a6c20": "1a6c20a60481", "00b6a4": "00b6a45ad6aa",
    "ba24c0": "ba24c0925311", "8197d9": "8197d9ac8a35",
    "6cdd25": "6cdd25bc75f2",
}


def sent_levels(x, sr):
    """逐句的 (平均电平 dB, f0 中位 Hz)。"""
    segs, _cuts, _hop = sent_windows(x, sr)
    db, f0 = [], []
    for a, b in segs:
        s = x[a:b]
        if len(s) < int(0.15 * sr):
            continue
        db.append(20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-12))
        try:
            from app.prosody_report import f0_of
            f = f0_of(s, sr)
            if f is not None and len(f) >= 5:
                f0.append(float(np.median(f)))
        except Exception:
            pass
    return np.asarray(db), np.asarray(f0)


def _ma(v, k):
    """k 句移动平均（valid 模式），长度不足返回 None。"""
    if len(v) < k + 2:
        return None
    c = np.cumsum(np.insert(v, 0, 0.0))
    return (c[k:] - c[:-k]) / float(k)


def slow_share(v, ks=(5, 10, 20)):
    """慢变占比：MA(k) 后的 sd / 原 sd。

    🔴 这是「起伏的时间结构」判据，比 sd 本身重要得多：
      - 逐句白噪声（每句独立随跳）→ MA(10) 后 sd 掉到 1/√10 ≈ 0.32
      - 真人慢变（成段地轻/响、基线缓移）→ MA 几乎不掉，比值接近 1
    同样一句 sd，前者听感是「抖」，后者听感是「有起伏」。
    """
    sd = float(v.std())
    out = {}
    for k in ks:
        m = _ma(v, k)
        if m is not None and sd > 1e-9:
            out[k] = float(m.std() / sd)
    return out


def stats(v, name):
    if len(v) < 4:
        print(f"  {name}: 样本不足({len(v)})")
        return None
    d = np.diff(v)
    # lag-1 自相关：慢变接近 1，逐句跳变接近 0
    a = v - v.mean()
    ac1 = float(np.sum(a[:-1] * a[1:]) / max(np.sum(a * a), 1e-12))
    ss = slow_share(v)
    print(f"  {name:<16} n={len(v):4d} 均值{v.mean():8.2f} "
          f"sd{v.std():6.2f} 极差{v.max()-v.min():7.2f} "
          f"相邻差分RMS{np.sqrt(np.mean(d**2)):6.2f} ac1{ac1:+.2f}")
    print(f"  {'':<16} 慢变占比 MA5 {ss.get(5, float('nan')):.2f} "
          f"MA10 {ss.get(10, float('nan')):.2f} "
          f"MA20 {ss.get(20, float('nan')):.2f}"
          f"   （白噪声≈0.45/0.32/0.22，慢变→1）")
    return dict(sd=float(v.std()), diff=float(np.sqrt(np.mean(d ** 2))),
                mean=float(v.mean()), ac1=ac1, slow=ss)


def one(tag, x, sr):
    db, f0 = sent_levels(x, sr)
    print(f"  [{tag}]")
    a = stats(db, "句级能量 dB")
    b = stats(f0, "句级f0中位 Hz")
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="r23")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    import subprocess
    rows = []
    for short, cid in FIVE.items():
        if a.only and short not in a.only.split(","):
            continue
        prod = sorted(glob.glob(os.path.join(OUT_DIR, f"{a.prefix}-{short}",
                                             "*.mp3")))
        if not prod:
            continue
        pw = os.path.join(WORK, f"{a.prefix}_{short}.wav")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f",
                        "concat", "-safe", "0",
                        "-i", _lst(prod, pw), "-ac", "1", "-ar", str(SR), pw],
                       check=True)
        px = load(pw, SR)
        cx = load(os.path.join(CASES_DIR, cid, "merged.wav"), SR)
        print("\n" + "=" * 70)
        print(f"{short}")
        pdb, pf0 = one("我们", px, SR)
        cdb, cf0 = one("案例", cx, SR)
        if pdb and cdb:
            print(f"  → 句级能量 sd 倍数  我们/案例 = "
                  f"{pdb['sd']/max(cdb['sd'],1e-9):.2f}"
                  f"   (我们 {pdb['sd']:.2f} vs 案例 {cdb['sd']:.2f})")
            print(f"  → 句级能量 相邻差分 倍数 = "
                  f"{pdb['diff']/max(cdb['diff'],1e-9):.2f}"
                  f"   (我们 {pdb['diff']:.2f} vs 案例 {cdb['diff']:.2f})")
        if pdb and cdb:
            for k in (10,):
                p, c = pdb['slow'].get(k), cdb['slow'].get(k)
                if p and c:
                    print(f"  ★ 能量慢变占比 MA{k}  我们 {p:.2f} vs "
                          f"案例 {c:.2f}   倍数 {p/max(c,1e-9):.2f}")
                    break
        if pf0 and cf0:
            print(f"  → 句级f0 sd 倍数  我们/案例 = "
                  f"{pf0['sd']/max(cf0['sd'],1e-9):.2f}"
                  f"   (我们 {pf0['sd']:.2f} vs 案例 {cf0['sd']:.2f})")
            print(f"  → 句级f0 中位差 我们−案例 = "
                  f"{pf0['mean']-cf0['mean']:+.2f} Hz "
                  f"({100*(pf0['mean']-cf0['mean'])/cf0['mean']:+.1f}%)")
            for k in (10,):
                p, c = pf0['slow'].get(k), cf0['slow'].get(k)
                if p and c:
                    print(f"  ★ f0 慢变占比 MA{k}  我们 {p:.2f} vs "
                          f"案例 {c:.2f}   倍数 {p/max(c,1e-9):.2f}")
                    break
        rows.append((short, pdb, cdb, pf0, cf0))
    print("\n" + "=" * 70)
    print(f"{'案例':<8}{'能量sd 我/案':>13}{'能量慢变 我/案':>15}"
          f"{'f0慢变 我/案':>14}{'f0中位差%':>11}")
    for short, pdb, cdb, pf0, cf0 in rows:
        if not (pdb and cdb and pf0 and cf0):
            continue
        pe = pdb['slow'].get(10, float('nan'))
        ce = cdb['slow'].get(10, float('nan'))
        pf = pf0['slow'].get(10, float('nan'))
        cf = cf0['slow'].get(10, float('nan'))
        print(f"{short:<8}{pdb['sd']/max(cdb['sd'],1e-9):>13.2f}"
              f"{pe/max(ce,1e-9):>15.2f}"
              f"{pf/max(cf,1e-9):>14.2f}"
              f"{100*(pf0['mean']-cf0['mean'])/cf0['mean']:>+11.1f}")


def _lst(files, out):
    p = out + ".txt"
    with open(p, "w") as w:
        for f in files:
            w.write("file '%s'\n" % f)
    return p


if __name__ == "__main__":
    main()
