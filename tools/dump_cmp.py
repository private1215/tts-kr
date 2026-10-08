# -*- coding: utf-8 -*-
"""逐项**带方向**对账：案例 vs 成品，同时算「铺底噪」对每一项的影响。

为什么需要它
------------
`score()` 里所有偏差都取了绝对值（`_rel` / `_dbdev`），看不出我们是**偏高**
还是**偏低**。而判断"能不能修"必须先知道方向：
  - 系统性偏高/偏低 → 有方向，可以调
  - 五例方向不一致 → 噪声，调了只是换一批噪声

顺带把底噪的逐项影响也量出来（`--rt` 指底噪电平），因为 VAD 用的是
**相对阈值** `帧电平 < p25 + 6 dB`，底噪会把静音帧从 −160 抬到 −50，
p25 跟着动 → 静音占比 / C / F 理论上会漂，必须实测确认不是负收益。

用法（服务器，qwen 环境）：
    venv/bin/python tools/dump_cmp.py --prefix r22
    venv/bin/python tools/dump_cmp.py --prefix r22 --rt -50
"""
import os
import sys
import glob
import json
import argparse

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import roomtone                                    # noqa: E402
from tools.fidelity import measure, score, WEIGHTS          # noqa: E402

SR = 24000
CASES = [
    ("1a6c20a60481", "1a6c20", "male", "标杆男"),
    ("00b6a45ad6aa", "00b6a4", "female", "标杆女"),
    ("ba24c0925311", "ba24c0", "male", "대한광통신"),
    ("8197d9ac8a35", "8197d9", "male", "세미파이브"),
    ("6cdd25bc75f2", "6cdd25", "male", "리노공업y"),
]
CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")
RT = "/mnt/tts/diag/rtwork"
KEYS = ["A", "B", "E", "D", "F", "sil", "C", "dur", "lufs", "peak"]


def levels(x, sr=SR):
    """帧电平分位数：看静音底噪落在哪。"""
    W = int(0.05 * sr)
    n = len(x) // W
    e = np.array([np.sqrt(np.mean(x[i * W:(i + 1) * W] ** 2)) for i in range(n)])
    db = 20 * np.log10(e + 1e-12)
    return [float(np.percentile(db, q)) for q in (5, 25, 75)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="r22")
    ap.add_argument("--rt", default="", help="底噪电平，如 -50；留空则不铺")
    ap.add_argument("--json", default="/mnt/tts/diag/dump_cmp.json")
    a = ap.parse_args()
    # --rt 留空=不铺；auto=按性别取线上实际电平；否则=指定统一电平
    auto = (a.rt == "auto")
    rtdb = None if (not a.rt or auto) else float(a.rt)

    rows = {}
    for cid, short, gender, label in CASES:
        case_w = os.path.join(CASES_DIR, cid, "merged.wav")
        # 🔴 中间 wav 必须带 prefix：不带的话第二次跑不同 prefix 会直接
        #    复用上次的缓存文件，表现为"换了产物分数一模一样"，极难排查。
        prod_w = os.path.join(RT, f"{a.prefix}_{short}.wav")
        if not os.path.exists(prod_w):
            prod = sorted(glob.glob(f"/mnt/tts/output/{a.prefix}-{short}/*.mp3"))
            if not prod:
                continue
            os.makedirs(RT, exist_ok=True)
            import subprocess
            lst = prod_w + ".txt"
            with open(lst, "w") as w:
                for f in prod:
                    w.write("file '%s'\n" % f)
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                            "-f", "concat", "-safe", "0", "-i", lst,
                            "-ac", "1", "-ar", str(SR), prod_w], check=True)
        cm = measure(case_w, "CASE")
        pm = measure(prod_w, "PROD")
        s0, d0 = score(cm, pm)
        cx, _ = sf.read(case_w, dtype="float32")
        px, _ = sf.read(prod_w, dtype="float32")
        rec = {
            "label": label, "gender": gender,
            "case": {k: cm.get(k) for k in KEYS},
            "prod": {k: pm.get(k) for k in KEYS},
            "devs": d0, "score": s0,
            "case_lv": levels(cx), "prod_lv": levels(px),
            "segs": cm.get("segs"), "pauses": cm.get("pauses"),
            "psegs": pm.get("segs"), "ppauses": pm.get("pauses"),
        }
        if auto or rtdb is not None:
            roomtone.ENABLED = True
            # auto = 用线上实际会用的电平（按性别取 level_for）
            db = roomtone.level_for(gender) if auto else rtdb
            y = roomtone.apply(px, SR, db)
            rp = os.path.join(RT, f"{short}_dump.wav")
            sf.write(rp, y, SR)
            rm = measure(rp, "PROD+RT")
            s1, d1 = score(cm, rm)
            rec["rt_db"] = db
            rec["rt"] = {k: rm.get(k) for k in KEYS}
            rec["rt_devs"] = d1
            rec["rt_score"] = s1
            rec["rt_lv"] = levels(y)
        rows[short] = rec
        print(f"[{short}] {label}  base={s0:.2f}"
              + (f"  rt{db:.0f}={s1:.2f} ({s1-s0:+.2f})"
                 if (auto or rtdb is not None) else ""), flush=True)

    # ---- 带符号对账表 ----
    print("\n" + "=" * 92)
    print("逐项「成品 − 案例」带符号相对偏差（%）：正数=我们偏高，负数=我们偏低")
    print("方向五例一致 = 系统性缺陷（可修）；方向乱 = 噪声（别动）")
    print("=" * 92)
    head = f"{'维度':<6}{'权重':>6}" + "".join(f"{s:>11}" for s in rows) + f"{'符号一致?':>10}"
    print(head)
    order = ["A", "B", "E", "D", "F", "sil", "C", "dur", "lufs", "peak", "tone"]
    for k in order:
        if k == "tone":
            continue
        vals, signs = [], []
        for short, r in rows.items():
            c, p = r["case"].get(k), r["prod"].get(k)
            if c is None or p is None or abs(c) < 1e-9:
                vals.append(None)
                continue
            v = (p - c) / abs(c) * 100.0
            vals.append(v)
            signs.append(1 if v > 0 else -1)
        agree = "一致" if signs and (all(s > 0 for s in signs)
                                    or all(s < 0 for s in signs)) else "混"
        line = f"{k:<6}{WEIGHTS[k]:>6.2f}"
        for v in vals:
            line += f"{(v if v is not None else float('nan')):>+11.2f}" if v is not None else f"{'-':>11}"
        line += f"{agree:>10}"
        print(line)
    print(f"\n{'分数':<6}{'':>6}" + "".join(f"{r['score']:>11.2f}" for r in rows.values()))
    mean0 = sum(r["score"] for r in rows.values()) / max(len(rows), 1)
    print(f"均值 {mean0:.2f}")
    if auto or rtdb is not None:
        print("铺底噪后：" + " ".join(
            f"{s}={r['rt_score']:.2f}({r['rt_score']-r['score']:+.2f})"
            for s, r in rows.items()))
        mean1 = sum(r["rt_score"] for r in rows.values()) / max(len(rows), 1)
        print(f"均值 {mean1:.2f}  ({mean1 - mean0:+.2f})")
        print("\n铺底噪前后逐项偏差变化（负数=变好）：")
        for k in order:
            if k == "tone":
                continue
            dd = [r["rt_devs"][k] - r["devs"][k] for r in rows.values()
                  if r["devs"].get(k) is not None and r["rt_devs"].get(k) is not None]
            if dd:
                print(f"  {k:<5} 权重{WEIGHTS[k]:.2f}  "
                      + " ".join(f"{v:+.2f}" for v in dd)
                      + f"   加权合计 {sum(dd)/len(dd)*WEIGHTS[k]:+.3f}")
        print("\n停顿检出：案例 segs/pauses -> 成品 -> 铺噪后")
        for s, r in rows.items():
            print(f"  {s:<8} {r['segs']}/{r['pauses']}  ->  "
                  f"{r['psegs']}/{r['ppauses']}")
    print("\n帧电平 p5/p25/p75（dBFS）：")
    for s, r in rows.items():
        lv = f"案例 {r['case_lv'][0]:6.1f}/{r['case_lv'][1]:6.1f}/{r['case_lv'][2]:6.1f}" \
             f"   成品 {r['prod_lv'][0]:6.1f}/{r['prod_lv'][1]:6.1f}/{r['prod_lv'][2]:6.1f}"
        if r.get("rt_lv"):
            lv += f"   铺噪 {r['rt_lv'][0]:6.1f}/{r['rt_lv'][1]:6.1f}/{r['rt_lv'][2]:6.1f}"
        print(f"  {s:<8}{lv}")

    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        json.dump(rows, open(a.json, "w"), ensure_ascii=False, indent=1)
        print(f"\n已存 {a.json}")


if __name__ == "__main__":
    main()
