# -*- coding: utf-8 -*-
"""为什么「整篇口径已经和标杆对齐」，员工听 40 秒开头却说卡？

pause_src_locate.py 的结果（整篇）：
    标杆 16.9 个/分钟、静音 12.23%、>=0.6s 有 4 个
    线上 18.2 个/分钟、静音 12.96%、>=0.6s 有 4 个
  → 整篇几乎持平。

但 pause_audible_check.py 量 40 秒开头片段：
    女 标杆 1 个 vs 我们 7 个
    男 标杆 7 个 vs 我们 15 个
  → 开头差 2~7 倍。

两者打架只有两种解释：
  ① **段首效应**：我们的音频开头那一段停顿特别密，后面正常
     （40 秒 ≈ 一个 cosy 生成段，段首没有上文，模型容易碎）；
  ② 采样波动：标杆的开头恰好也是稀疏的，纯属巧合。

本脚本把整条按 40 秒开窗，逐窗算 gap 密度，看：
  - 我们自己：窗间是否前密后疏；
  - 标杆：是否也是前密后疏（若是，则 40 秒对比是公平的，差异真实）；
  - 逐窗差值：差异集中在哪些窗。
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from tools.pause_audible_check import frame_db, gaps_of, SR  # noqa: E402

TMP = tempfile.mkdtemp(prefix="pwin_")
WIN = 40.0

ITEMS = [
    ("标杆(EL原声)", "/mnt/tts/data/cases/1a6c20a60481/merged.wav"),
    ("线上成品", "/tmp/selftest_open/open_male.wav"),
    ("裸推理(cosy)", "/mnt/tts/projects/gpt-sovits/out/cosy_full_m.wav"),
]


def load16(path):
    out = os.path.join(TMP, os.path.basename(path)[:40] + ".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", str(SR), "-ac", "1", out], check=True)
    x, _ = sf.read(out)
    return np.asarray(x, dtype=np.float64)


def win_density(path, win=WIN):
    if not os.path.exists(path):
        return None
    x = load16(path)
    dur = len(x) / SR
    got = frame_db(x)
    if got is None:
        return None
    db, t = got
    gaps = gaps_of(db, t)
    nwin = int(np.ceil(dur / win))
    rows = []
    for k in range(nwin):
        a, b = k * win, min((k + 1) * win, dur)
        g = [q for q in gaps if q[0] >= a and q[1] <= b]
        sil = sum(q[1] - q[0] for q in g)
        rows.append(dict(win=k, a=a, b=b, n=len(g),
                         per_min=len(g) / max(b - a, 1e-9) * 60.0,
                         sil=100.0 * sil / max(b - a, 1e-9)))
    return rows, dur, gaps


def main():
    res = {}
    for tag, p in ITEMS:
        r = win_density(p)
        if r:
            res[tag] = r
    tags = list(res.keys())
    if not tags:
        print("没有可用音频")
        return 1
    nwin = max(len(res[t][0]) for t in tags)
    print(f"窗口 {WIN:.0f}s，逐窗 gap 密度（个/分钟）")
    print("=" * (14 + 18 * len(tags)))
    print("%-14s" % "窗口" + "".join("%18s" % t for t in tags))
    print("-" * (14 + 18 * len(tags)))
    for k in range(nwin):
        line = "%-14s" % f"{k*WIN:.0f}-{min((k+1)*WIN, 999):.0f}s"
        for t in tags:
            rows = res[t][0]
            if k < len(rows):
                line += "%18s" % (f"{rows[k]['n']:3d} 个 {rows[k]['per_min']:5.1f}/min")
            else:
                line += "%18s" % "-"
        print(line)
    print("=" * (14 + 18 * len(tags)))
    print()
    for t in tags:
        rows, dur, gaps = res[t]
        ds = [r["per_min"] for r in rows]
        first = np.mean(ds[:2]) if len(ds) >= 2 else ds[0]
        rest = np.mean(ds[2:]) if len(ds) > 2 else np.nan
        print(f"{t:<16} 总 {len(gaps)} 个 / {dur:.0f}s   前2窗均值 {first:.1f}/min"
              f"   其余窗均值 {rest:.1f}/min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
