# -*- coding: utf-8 -*-
"""recheck_male_click48.py —— 男声 GS click 复测（yohan 重训后一键对比）。

## 背景
男声 click 是我们离 EL 最远的指标（现役 voxcpm 1.175%，EL 0.259%）。
第48轮定位为「**合成器固有**」（发声中 100%、周期性 19% < 60%，不是块边界），
怀疑是 **yohan 组训练步数不足**：953 段 → 119 步/轮 → e40 只有 4840 步，
而 SOLA 是 6160 步（少 21%）。→ 加了 epoch 自适应（e52 = 6188 步）。

🔴 但自适应**一直没有真正生效**：`gs_auto_train.py` 的「没变化就跳过」漏判
（dataset 比权重新也报无变化）。所以必须手工重训后复测。

## 本脚本做什么
1. 读 `/mnt/tts/diag/recheck_male_click/baseline.json`（训练前的配置 + click 基线）
2. 用**完全相同**的配置重新生成一条 GS 男声（engine=gptsovits / voice=yohan_gs_f8
   / seed=42 / 同一段 1500 字文案）
3. 用 `clickpos48.py` 的同款口径测 click，与基线对比

## 🔴 口径警告
`click %` 在不同脚本里差两个数量级（VERSION 里是 0.517%，clickpos48 是 0.0044%）。
**本脚本只用「次/秒」和「clickpos48 密度%」两个口径，且训练前后必须用同一个。**
基线实测：129 次 / 182.2 s = **0.708 次/秒**。

用法：
  python3 recheck_male_click48.py              # 生成 + 测量 + 对比
  python3 recheck_male_click48.py <已有音频>    # 只测量（训练完手动拿到成品时用）
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")

BASE_JSON = "/mnt/tts/diag/recheck_male_click/baseline.json"
SR = 16000
THR = 0.30      # 与 clickpos48.py 完全一致：峰值归一 0.70 后 |Δ| >= 0.30


def load(p):
    if not p.endswith(".wav"):
        t = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                        "-ar", str(SR), "-ac", "1", t], check=True)
        p = t
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def measure(path):
    x, sr = load(path)
    x = x / (np.abs(x).max() + 1e-12) * 0.70
    d = np.abs(np.diff(x))
    hits = np.where(d >= THR)[0]
    dur = len(x) / sr
    n = int(sr * 0.1)
    ctx = np.array([float(np.sqrt((x[max(0, i - n):i + n] ** 2).mean()))
                    for i in hits[:400]])
    whole = float(np.sqrt((x ** 2).mean()))
    voiced = float((ctx > whole * 0.3).mean() * 100) if len(ctx) else 0.0
    return {"n": int(len(hits)), "dur": round(dur, 1),
            "per_s": round(len(hits) / dur, 3) if dur else 0.0,
            "density_pct": round(len(hits) / len(x) * 100, 4),
            "voiced_pct": round(voiced, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?", default="", help="已有音频；不给则重新生成")
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    base = json.load(open(BASE_JSON, encoding="utf-8"))
    print("=== 训练前基线 ===")
    print("  %s  %s  时长 %.1fs" % (base["title"], base["voice"], base["baseline_dur_s"]))
    print("  click %d 次 = %.3f 次/秒  密度 %.4f%%"
          % (base["baseline_click_n"], base["baseline_click_per_s"],
             129 / (base["baseline_dur_s"] * SR) * 100))

    wav = a.wav
    if not wav:
        from tools.eval5 import login, submit_job, wait_job, wait_dir
        title = a.title or ("gs48m_recheck_%s" % time.strftime("%m%d%H%M"))
        print("\n=== 重新生成：%s ===" % title)
        login()
        r = submit_job(base["text"], base["voice"], title,
                       engine=base["engine"], seed=base.get("seed"))
        jid = r.get("id") or r.get("job_id") or r.get("jid")
        print("  job", jid)
        st = wait_job(jid)
        print("  状态", st.get("status"))
        fs = wait_dir(title)
        if not fs:
            print("❌ 没等到产物")
            return 1
        wav = fs[0]
    print("\n=== 测量：%s ===" % os.path.basename(wav))
    cur = measure(wav)
    print("  时长 %.1fs  click %d 次 = %.3f 次/秒  密度 %.4f%%  发声中 %.0f%%"
          % (cur["dur"], cur["n"], cur["per_s"], cur["density_pct"], cur["voiced_pct"]))

    b = base["baseline_click_per_s"]
    print("\n=== 对比（次/秒）===")
    print("  训练前 %.3f  →  训练后 %.3f   (%+.1f%%)"
          % (b, cur["per_s"], (cur["per_s"] - b) / b * 100))
    if cur["per_s"] < b * 0.85:
        print("  ✅ click 下降 >=15%%，epoch 自适应（e52）生效")
    elif cur["per_s"] > b * 1.15:
        print("  ❌ click 反而上升，方向错了，考虑回退权重")
    else:
        print("  ⚠️ 变化在 ±15%% 内，epoch 自适应**没有明显效果**"
              " → click 不是训练量的问题，别再往训练方向加码")
    out = os.path.join(os.path.dirname(BASE_JSON), "result_%s.json"
                       % time.strftime("%m%d%H%M"))
    json.dump({"baseline": base, "current": cur, "wav": wav},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
