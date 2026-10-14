# -*- coding: utf-8 -*-
"""定位「gap 太多」到底是谁造的：引擎合成端，还是线上后处理（PBA）。

上一轮 pause_audible_check.py 的结论：
    女  EL标杆 1 个 gap / 我们修复版 7 个
    男  EL标杆 7 个 gap / 我们修复版 15 个（比旧版 14 个还多）
→ 卡顿的真实形态是**停顿太碎**，不是停顿太长；而我 10-12 只修了长度
  （>=0.6s 长间隙 14->0），数量完全没动，所以"根治"是假的。

本脚本把同一条文案的三段音频放在一起比**停顿密度**：
    A 案例原声（EL 标杆）       ground truth
    B 引擎裸推理（cosy 直出，未走主服务后处理）
    C 线上成品（走了完整后处理）
谁和 A 差得多，谁就是主因。

用法：
    python -u tools/pause_src_locate.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from tools.pause_audible_check import (  # noqa: E402
    analyse, SR, DB_ON, GAP_MIN, LONG_GAP)

TMP = tempfile.mkdtemp(prefix="psrc_")

# (标签, 路径, 是否需要解码)
ITEMS = [
    ("A 案例原声(EL标杆)", "/mnt/tts/data/cases/1a6c20a60481/merged.wav", None),
    ("B 裸推理(cosy直出)", "/mnt/tts/projects/gpt-sovits/out/cosy_full_m.wav", None),
    ("C 线上成品", "/tmp/selftest_open/open_male.wav", None),
]


def load16(path):
    out = os.path.join(TMP, os.path.basename(path)[:40] + ".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", str(SR), "-ac", "1", out], check=True)
    x, _ = sf.read(out)
    return np.asarray(x, dtype=np.float64)


def density(tag, path):
    if not os.path.exists(path):
        print(f"{tag}: 缺文件 {path}")
        return None
    x = load16(path)
    dur = len(x) / SR
    # 直接复用 pause_audible_check 的分析逻辑
    import tools.pause_audible_check as pac
    got = pac.frame_db(x)
    if got is None:
        return None
    db, t = got
    gaps = pac.gaps_of(db, t)
    longs = [g for g in gaps if g[1] - g[0] >= LONG_GAP]
    sil = sum(g[1] - g[0] for g in gaps)
    return dict(tag=tag, dur=dur, n=len(gaps),
                per_min=len(gaps) / dur * 60.0,
                n_long=len(longs),
                sil_pct=100.0 * sil / dur,
                mean_gap=float(np.mean([g[1] - g[0] for g in gaps])) if gaps else 0.0,
                max_gap=float(max([g[1] - g[0] for g in gaps])) if gaps else 0.0)


def main():
    print("口径：%dHz / %.0fms 帧 / dB>%.0f 为有声 / 内部静音>=%.2fs 算一个 gap"
          % (SR, 20, DB_ON, GAP_MIN))
    print("=" * 92)
    print("%-22s %8s %8s %10s %10s %10s %10s"
          % ("片段", "时长s", "gap数", "个/分钟", "≥0.6s", "静音占比%", "平均gap"))
    print("-" * 92)
    rows = []
    for tag, p, _ in ITEMS:
        r = density(tag, p)
        if r:
            rows.append(r)
            print("%-22s %8.1f %8d %10.1f %10d %10.2f %10.2f"
                  % (r["tag"], r["dur"], r["n"], r["per_min"], r["n_long"],
                     r["sil_pct"], r["mean_gap"]))
    print("=" * 92)
    if len(rows) >= 3:
        a, b, c = rows[0], rows[1], rows[2]
        print("\n判读：")
        print("  标杆(A) 密度 %.1f 个/分钟" % a["per_min"])
        print("  裸推理(B) 密度 %.1f 个/分钟  → 相对标杆 %+.0f%%"
              % (b["per_min"], 100 * (b["per_min"] / a["per_min"] - 1)))
        print("  线上(C) 密度 %.1f 个/分钟  → 相对标杆 %+.0f%%"
              % (c["per_min"], 100 * (c["per_min"] / a["per_min"] - 1)))
        db_ = c["per_min"] - b["per_min"]
        print("\n  后处理净贡献 %+.1f 个/分钟（C−B）" % db_)
        if abs(db_) < 0.15 * max(a["per_min"], 1e-9):
            print("  → 后处理基本不背锅，gap 多来自**引擎合成端/文本分段**")
        else:
            print("  → 后处理是主要放大器，重点查 PBA 的停顿密度目标 pause_density()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
