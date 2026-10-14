# -*- coding: utf-8 -*-
"""EL 男声共振峰基线 —— 给 eq_search6 的 FORMANT_TARGET/SD 提供真实分布。

为什么要单独做（2026-10-14）：
    之前 FORMANT_SD=(25, 90) 是我拍的，权重 FORMANT_W 也就没法定：
    SD 一大，共振峰项在 J 里几乎为 0，搜索会完全无视它，白加。
    频段基线的成功经验是「用 EL 案例的真实分布 + σ 作为量纲」——
    那共振峰也应该照做：量纲统一成 σ 之后，共振峰和 7 个频段项就是
    同一把尺子，不需要再人为调 FORMANT_W（保持 1.0 即可）。

口径：Praat Burg，16k，取 [10% 时长, +180s] 段，只统计有 F0 的浊音帧。
      与 eq_search6.formants() 必须同口径，否则不可比。
"""
import json
import os
import subprocess
import tempfile

import numpy as np
import parselmouth
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_JSON = "/mnt/tts/data/cases.json"
CASE_DIR = "/mnt/tts/data/cases"
OUT = os.path.join(HERE, "el_formant_male.json")
BAD_MALE = {"c3117b7b0fb1"}      # 标 male 但 F0=275.7Hz 的脏数据
SEG_LEN = 180.0                  # 最多取 180s，够统计且快


def slice16(path, start, dur):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "%.2f" % start,
                    "-t", "%.2f" % dur, "-i", path,
                    "-ar", "16000", "-ac", "1", t], check=True)
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y, 16000


def formants(path):
    """与 eq_search6.formants 同口径，只是改成相对窗口（案例时长不齐）。"""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True)
    dur = float(probe.stdout.strip() or 0)
    start = max(0.0, 0.10 * dur)
    length = min(SEG_LEN, max(5.0, dur - start))
    y16, sr = slice16(path, start, length)
    s = parselmouth.Sound(y16, sr)
    pit = s.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    fm = s.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                           maximum_formant=5000)
    ts = np.arange(0.05, s.duration - 0.05, 0.01)
    F1, F2 = [], []
    for tt in ts:
        f0 = pit.get_value_at_time(tt)
        if f0 is None or np.isnan(f0) or f0 <= 0:
            continue
        a = fm.get_value_at_time(1, tt)
        b = fm.get_value_at_time(2, tt)
        if a is None or b is None or np.isnan(a) or np.isnan(b) \
                or a <= 0 or b <= 0:
            continue
        F1.append(a)
        F2.append(b)
    if len(F1) < 100:
        return None
    return {"F1": float(np.median(F1)), "F2": float(np.median(F2)),
            "n": len(F1)}


def main():
    raw = json.load(open(CASES_JSON, encoding="utf-8"))
    cases = raw.values() if isinstance(raw, dict) else raw
    rows = []
    for c in cases:
        cid = c.get("id")
        if (c.get("gender") or "") != "male":
            continue
        if cid in BAD_MALE:
            print("跳过脏数据 %s" % cid, flush=True)
            continue
        p = os.path.join(CASE_DIR, cid, "merged.wav")
        if not os.path.exists(p):
            print("缺文件 %s" % p, flush=True)
            continue
        r = formants(p)
        if not r:
            print("测不出 %s" % cid, flush=True)
            continue
        r["id"] = cid
        rows.append(r)
        print("%-14s F1=%7.1f  F2=%7.1f  (n=%d)"
              % (cid, r["F1"], r["F2"], r["n"]), flush=True)

    if len(rows) < 3:
        print("样本不足：%d" % len(rows))
        return 1
    f1 = np.array([r["F1"] for r in rows])
    f2 = np.array([r["F2"] for r in rows])
    out = {"n": len(rows),
           "F1": {"mean": float(f1.mean()), "sd": float(f1.std(ddof=1)),
                  "min": float(f1.min()), "max": float(f1.max())},
           "F2": {"mean": float(f2.mean()), "sd": float(f2.std(ddof=1)),
                  "min": float(f2.min()), "max": float(f2.max())},
           "rows": rows}
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("\n=== EL 男声共振峰基线 n=%d ===" % len(rows))
    print("F1  %7.1f ± %5.1f   (极差 %.1f~%.1f)"
          % (out["F1"]["mean"], out["F1"]["sd"],
             out["F1"]["min"], out["F1"]["max"]))
    print("F2  %7.1f ± %5.1f   (极差 %.1f~%.1f)"
          % (out["F2"]["mean"], out["F2"]["sd"],
             out["F2"]["min"], out["F2"]["max"]))
    print("写入 %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
