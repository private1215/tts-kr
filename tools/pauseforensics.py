# -*- coding: utf-8 -*-
"""pauseforensics.py —— 停顿结构法证：GS 的「机器感」是不是来自等长死静音？

## 怀疑

`deploy/gs_serve.py:72` 有一个常量：

    SEG_GAP_SEC = 0.30

而 `_run()` 在**每一段之间**插一段 `np.zeros(int(SEG_GAP_SEC * sr))`：

    chunks.append(audio)
    if i < len(segs) - 1:
        chunks.append(gap)

也就是说：GS 按 50 字切段推理，每段之间塞 **0.30 秒的纯数字零**。

这会同时造成两个极不自然的效果：
1. **停顿长度完全一样** —— 真人说话的句间停顿是变化的（0.2~0.9s 不等），
   GS 是「咔、咔、咔」的等长节拍，一听就是机器。
2. **停顿是数字零（-inf dB）** —— 真人录音的静音里有房间底噪、有呼吸，
   绝对不会是数学上的零。

而这两个问题，**常规客观指标一个也测不出来**：
    CER（念对没有）—— 测不出
    click（波形硬切）—— 静音段没有硬切，测不出
    音色 campplus（说话人嵌入）—— 停顿不影响音色，测不出
    杂音（静音段 4-8k 占比）—— 纯零静音反而不吵，测不出

→ 这就解释了 z20 那个「客观全面优于 EL、员工却一致判 0 分」的矛盾。

## 用法

    python3 pauseforensics.py <音频目录> [输出目录]

会打一张停顿长度分布图 + 一张数值表。
"""
import json
import os
import sys

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SR = 16000
FRAME = 0.02          # 20ms 一帧
MIN_PAUSE = 0.12      # 短于这个不算停顿（音节内部间隙）
GAP_SUSPECT = 0.30    # 🔴 GS_SERVE 里写死的段间间隔


def load(p, sr=SR):
    y, _ = librosa.load(p, sr=sr, mono=True)
    return y


def pause_segments(y, sr=SR, frame=FRAME):
    """切出静音段，返回 [(起始s, 长度s, 段内中位dB), ...]。"""
    n = int(frame * sr)
    rms = librosa.feature.rms(y=y, frame_length=n, hop_length=n)[0]
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > np.percentile(db, 35)      # 相对阈值，跟绝对音量无关
    # 找连续静音区
    segs = []
    i = 0
    while i < len(voiced):
        if not voiced[i]:
            j = i
            while j < len(voiced) and not voiced[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    # 丢掉首尾（开头/结尾静音不是句间停顿）
    out = []
    for a, b in segs:
        if a == 0 or b == len(voiced):
            continue
        dur = (b - a) * frame
        if dur < MIN_PAUSE:
            continue
        out.append((a * frame, dur, float(np.median(db[a:b]))))
    return out


def stats(path):
    y = load(path)
    segs = pause_segments(y)
    lens = np.array([s[1] for s in segs]) if segs else np.array([])
    deps = np.array([s[2] for s in segs]) if segs else np.array([])
    dur = len(y) / SR
    r = {"name": os.path.basename(path), "dur_s": float(dur),
         "n_pause": int(len(segs)),
         "pause_per_min": float(len(segs) / dur * 60) if dur else 0}
    if len(lens):
        r.update({
            "len_med": float(np.median(lens)),
            "len_mean": float(np.mean(lens)),
            "len_std": float(np.std(lens)),
            "len_min": float(lens.min()),
            "len_max": float(lens.max()),
            # 🔴 关键：停顿长度的相对变异。真人大，等长插入的机器小
            "len_cv": float(np.std(lens) / np.mean(lens)),
            # 🔴 落在 0.30s ±15% 里的比例：等长插入会形成尖峰
            "near_030_pct": float(
                ((lens > GAP_SUSPECT * 0.85) & (lens < GAP_SUSPECT * 1.15)).mean() * 100),
            "depth_p5_db": float(np.percentile(deps, 5)),
            "depth_med_db": float(np.median(deps)),
            "n_digital_silence": int((deps < -80).sum()),  # 数字零静音
        })
        r["_lens"] = lens.tolist()
        r["_deps"] = deps.tolist()
    return r


def main():
    indir = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/pauseout"
    os.makedirs(outdir, exist_ok=True)
    files = sorted(f for f in os.listdir(indir) if f.endswith((".wav", ".mp3")))
    res = {}
    for f in files:
        print("分析", f, flush=True)
        res[f] = stats(os.path.join(indir, f))
    json.dump(res, open(os.path.join(outdir, "pauses.json"), "w"),
              ensure_ascii=False, indent=1)

    print("\n%-18s %6s %7s %7s %7s %7s %9s %8s" %
          ("文件", "停顿数", "中位s", "标准差", "变异CV", "最短s", "≈0.30s占比", "-80dB以下"))
    for k in sorted(res):
        r = res[k]
        if "len_med" not in r:
            print("%-18s %6d %7s" % (k[:18], r["n_pause"], "无停顿"))
            continue
        print("%-18s %6d %7.2f %7.3f %7.2f %7.2f %9.0f%% %8d" % (
            k[:18], r["n_pause"], r["len_med"], r["len_std"], r["len_cv"],
            r["len_min"], r["near_030_pct"], r["n_digital_silence"]))

    # ---- 停顿长度分布图 ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, k in zip(axes.ravel(), sorted(res)):
        lens = res[k].get("_lens") or []
        ax.set_title(k.replace(".wav", ""), fontsize=10)
        if not lens:
            ax.text(0.5, 0.5, "无停顿", ha="center", transform=ax.transAxes)
            continue
        bins = np.arange(0, max(1.0, max(lens) + 0.1), 0.05)
        ax.hist(lens, bins=bins, color="#378ADD", alpha=.85)
        ax.axvline(GAP_SUSPECT, color="#E24B4A", ls="--", lw=1.4,
                   label="0.30s（GS 段间插入值）")
        ax.set_xlabel("停顿长度 (s)", fontsize=8)
        ax.set_ylabel("次数", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    plt.tight_layout()
    png = os.path.join(outdir, "pause_hist.png")
    plt.savefig(png, dpi=110)
    print("\n图 -> %s" % png)
    print("json -> %s/pauses.json" % outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
