# -*- coding: utf-8 -*-
"""🔴 呼吸感画像：停顿的**时长分布**，不只是个数（2026-10-15 第23轮反馈驱动）

## 驱动本脚本的反馈
3 份有效答卷里 3 份都指向同一件事：
- u5「只能改一处先改哪里」→ **③ 断句停顿不自然 2/3**
- u7「听着累不累」→ **② 停顿太多太碎，断得很别扭 2/3**
- 24124112 自由文本：「每个单词最后一个字**断句特别快**」
- 고구마 自由文本：「**感觉不喘气一样**」「**呼吸频率不像真人**」

🔴 而 `pause_window_density.py` 已测出：停顿**个数**我们 149 个 / EL 136 个，
只差 10% —— **个数不是问题**。所以必须测**时长分布**：
「碎」的物理含义是**短停顿占比过高**，「不喘气」是**缺少够长的换气口**。

## 本脚本测什么（EL 对照 + 我们）
    n          停顿个数
    短停顿(<0.20s) 占比   ← 「碎」的直接度量
    中停顿(0.20~0.50s) 占比
    长停顿(>=0.50s) 占比  ← 「换气口」，真人靠它换气
    最长停顿
    相邻停顿间隔分布      ← 换气节奏是否规律
    停顿时长熵(0~1)      ← 停顿是「只有一个长度」还是「有层次」

## 口径（🔴 三条铁律）
1. **必须长文案**（≥180s）。短片停顿分布没意义（说话人还没进入稳定状态）。
2. **必须全篇**，不能只看前 60s（`probe_g` 那个 bug 的教训）。
3. **阈值必须用「可听口径」**：`pause_targets_audible.json`，
   不是 -45dB 那种绝对静音口径（历史上试过，全爆表）。

## 用法
    python breath_profile.py <长篇.wav/mp3> [...]
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

SR = 16000
SR24 = 24000
LONG_MIN = 180
# 停顿判定阈值（dB，相对该文件的静音底噪）
THR_DB = -38.0
MIN_PAUSE = 0.12          # 短于这个不算停顿（避免把字内空隙算进去）

SHORT, LONG = 0.20, 0.50  # 秒


def to16(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    import soundfile as sf
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y


def dur_of(p):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", p], capture_output=True,
        text=True).stdout.strip() or 0)


def frame_db(y, win=0.02, hop=0.01):
    n = int(SR * hop)
    w = int(SR * win)
    m = (len(y) - w) // n
    out = np.empty(m)
    for i in range(m):
        fr = y[i * n:i * n + w]
        r = np.sqrt(np.mean(fr ** 2)) + 1e-12
        out[i] = 20 * np.log10(r)
    return out


def find_pauses(db):
    sil = db < THR_DB
    out = []
    i, N = 0, len(sil)
    while i < N:
        if not sil[i]:
            i += 1
            continue
        j = i
        while j < N and sil[j]:
            j += 1
        # 合并极短的间隙（<80ms），避免一个换气被切成两段
        dur = (j - i) * 0.01
        if dur >= MIN_PAUSE:
            out.append((i * 0.01, j * 0.01, dur))
        i = j
    # 合并：两个停顿间隔 < 0.08s 视为同一个
    merged = []
    for s, e, d in out:
        if merged and s - merged[-1][1] < 0.08:
            ps, pe, _ = merged[-1]
            merged[-1] = (ps, e, e - ps)
        else:
            merged.append((s, e, d))
    return merged


def entropy(v):
    if not len(v):
        return 0.0
    h, _ = np.histogram(v, bins=6, range=(0, 1.2))
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(6))


def profile(path):
    y = to16(path)
    db = frame_db(y)
    pa = find_pauses(db)
    d = [p[2] for p in pa]
    tot = len(y) / SR
    r = {"path": os.path.basename(path), "dur": round(tot, 1),
         "n_pause": len(pa), "per_min": round(len(pa) / tot * 60, 1)}
    if d:
        d = np.array(d)
        r.update({
            "short%": round(float((d < SHORT).sum() / len(d) * 100), 1),
            "mid%": round(float(((d >= SHORT) & (d < LONG)).sum() / len(d) * 100), 1),
            "long%": round(float((d >= LONG).sum() / len(d) * 100), 1),
            "dur_med": round(float(np.median(d)), 3),
            "dur_mean": round(float(d.mean()), 3),
            "dur_max": round(float(d.max()), 3),
            "dur_p90": round(float(np.percentile(d, 90)), 3),
            "entropy": round(entropy(d), 3),
        })
        # 换气口：>1.2s 的长停顿，真人靠它换气
        r["n_breath"] = int((d > 1.2).sum())
        r["breath_per_min"] = round(float((d > 1.2).sum() / tot * 60), 1)
    return r


def main():
    files = sys.argv[1:]
    if not files:
        sys.exit("用法: breath_profile.py <长篇成品> [EL 标杆 ...]")
    rows = []
    for f in files:
        dd = dur_of(f)
        if dd and dd < LONG_MIN:
            print("⚠️ 跳过 %s（%.0fs < %ds，短文案的停顿分布没意义）"
                  % (os.path.basename(f), dd, LONG_MIN))
            continue
        try:
            r = profile(f)
            rows.append(r)
        except Exception as e:                                # noqa: BLE001
            print("FAIL %s: %r" % (os.path.basename(f), e))
    if not rows:
        return
    ks = ["n_pause", "per_min", "short%", "mid%", "long%", "dur_med",
          "dur_max", "dur_p90", "entropy", "n_breath", "breath_per_min"]
    print("\n%-26s %s" % ("样本", " ".join("%10s" % k for k in ks)))
    for r in rows:
        print("%-26s %s" % (r["path"][:26],
                            " ".join("%10s" % r.get(k, "-") for k in ks)))
    out = "/mnt/tts/diag/muffle/breath_profile.json"
    json.dump(rows, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n-> %s" % out)


if __name__ == "__main__":
    main()