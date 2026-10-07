# -*- coding: utf-8 -*-
"""离线试算：停顿条数预算 + 时长对齐，看还原度怎么变。

## 为什么做这个

r22 五案例「成品 − 案例」**带符号**偏差里，只有三项方向五例一致：

    sil  +7.8 ~ +17.1%（均值 +12.5%）  我们静音太多   ← 最大
    C    +2.3 ~ +9.5%  （均值 +6.5%）  能量动态偏大   ← r21 压 dyn 已证伪
    dur  -0.7 ~ -4.1%  （均值 -2.3%）  我们整体偏短

拆开 sil 一看，均值几乎一样、多的是**条数**：

    案例 1a6c20   162 处 / 90.7s（均值 0.560s）
    我们 1a6c20   178 处 / 94.8s（均值 0.533s）

按字数归一后我们比案例多 11.5% 个停顿。α 和 mu 都是「给每条停顿定长度」，
条数多了总量就是多 —— 必须给条数一个预算。

## 试什么

1. `budget`：把多余停顿压到 60ms（低于 min_sil=0.25s，检不出来）的比例。
   0=不改，1.0=全压到案例密度，0.6=保守。
2. `durfix`：把总时长拉到 `字数 / 案例 cps` —— 修 dur 的 −2.3%。

⚠️ 最大的未知是 **F（停顿熵）**：我们 4/5 例 F 已经偏高（+5~+10%），
压掉最短的停顿会让直方图第一桶变轻、熵继续升高。所以必须实测，
不能靠推理拍板。

用法（服务器，qwen 环境）：
    venv/bin/python tools/try_pause.py
"""
import os
import sys
import glob
import json
import argparse

import numpy as np
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# app/ 内部是扁平导入（server.py 就是这么跑的），tools 下必须也把它进 path
sys.path.insert(0, os.path.join(_ROOT, "app"))
from app import pba                                          # noqa: E402
from app import prosody_report                               # noqa: E402
from tools.fidelity import measure, score, WEIGHTS           # noqa: E402

SR = 24000
CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")
RT = "/mnt/tts/diag/rtwork"
CASES = [
    ("1a6c20a60481", "1a6c20", "male", "标杆男"),
    ("00b6a45ad6aa", "00b6a4", "female", "标杆女"),
    ("ba24c0925311", "ba24c0", "male", "대한광통신"),
    ("8197d9ac8a35", "8197d9", "male", "세미파이브"),
    ("6cdd25bc75f2", "6cdd25", "male", "리노공업y"),
]
PBA_CV = 0.28


def case_text(cid):
    import re
    d = os.path.join(CASES_DIR, cid)
    full = os.path.join(d, "full.txt")
    if os.path.exists(full):
        return open(full, encoding="utf-8").read()
    segs = sorted(glob.glob(os.path.join(d, "seg*.txt")),
                  key=lambda p: int(re.findall(r"seg(\d+)", p)[0]))
    return "\n".join(open(p, encoding="utf-8").read().strip() for p in segs)


def pause_runs(x, sr):
    _, cuts, _ = prosody_report.sent_windows(x, sr)
    runs = np.array([(j - i) / float(sr) for i, j in cuts])
    return runs[runs >= 0.28]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="r22")
    ap.add_argument("--grid", default="1.0,0.6")
    ap.add_argument("--durfix", default="1", help="1=开,0=关,1,0=都试")
    ap.add_argument("--scale", default="1", help="1=也试等比缩放")
    a = ap.parse_args()
    global SCALE
    SCALE = a.scale != "0"
    grid = [float(v) for v in a.grid.split(",")]
    durmodes = [bool(int(v)) for v in a.durfix.split(",")]

    # ---- 先扫案例库：算性别的 停顿密度(条/字) 与 cps(字/秒) ----
    info = {}
    for cid, short, gender, label in CASES:
        cw = os.path.join(CASES_DIR, cid, "merged.wav")
        x, sr = sf.read(cw, dtype="float32")
        if getattr(x, "ndim", 1) > 1:
            x = x.mean(axis=1)
        runs = pause_runs(x, sr)
        n_chars = len(case_text(cid))
        info[short] = dict(gender=gender, label=label, cid=cid,
                           pauses=len(runs), chars=n_chars,
                           dur=len(x) / float(sr), runs=runs)
    dens, cps = {}, {}
    for g in ("male", "female"):
        ps = sum(v["pauses"] for v in info.values() if v["gender"] == g)
        ch = sum(v["chars"] for v in info.values() if v["gender"] == g)
        du = sum(v["dur"] for v in info.values() if v["gender"] == g)
        dens[g] = ps / max(ch, 1)
        cps[g] = ch / max(du, 1e-9)
    print(f"案例库密度(条/字)  男 {dens['male']:.5f}  女 {dens['female']:.5f}")
    print(f"案例库语速(字/秒)  男 {cps['male']:.4f}  女 {cps['female']:.4f}")

    # 案例停顿样本：按性别拼（与线上 pause_targets 同口径）
    samples = {}
    for g in ("male", "female"):
        samples[g] = np.concatenate(
            [v["runs"] for v in info.values() if v["gender"] == g])

    import librosa
    out = {}
    for short, ci in info.items():
        pw = os.path.join(RT, f"{short}.wav")
        if not os.path.exists(pw):
            prod = sorted(glob.glob(f"/mnt/tts/output/{a.prefix}-{short}/*.mp3"))
            if not prod:
                continue
            import subprocess
            lst = pw + ".txt"
            with open(lst, "w") as w:
                for f in prod:
                    w.write("file '%s'\n" % f)
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                            "-f", "concat", "-safe", "0", "-i", lst,
                            "-ac", "1", "-ar", str(SR), pw], check=True)
        cm = measure(os.path.join(CASES_DIR, ci["cid"], "merged.wav"), "CASE")
        x0, _ = sf.read(pw, dtype="float32")
        base_m = measure(pw, "PROD")
        s0, d0 = score(cm, base_m)
        n_ours = len(pause_runs(x0, SR))
        n_tgt = dens[ci["gender"]] * ci["chars"]
        dur_tgt = ci["chars"] / cps[ci["gender"]]
        rec = dict(base=s0, n_ours=n_ours, n_tgt=n_tgt, n_case=ci["pauses"],
                   dur_tgt=dur_tgt, devs=d0, variants={})
        print(f"\n[{short}] {ci['label']}  base={s0:.2f}  "
              f"停顿 案例{ci['pauses']} / 我们{n_ours} / 预算{n_tgt:.0f}  "
              f"时长 案例{ci['dur']:.1f} / 目标{dur_tgt:.1f} / 我们{len(x0)/SR:.1f}")
        variants = []
        for f in grid:
            budget = None
            if f > 0 and n_ours > n_tgt:
                budget = n_ours - f * (n_ours - n_tgt)
            variants.append((f"drop{f:.1f}", dict(pause_budget=budget)))
        if SCALE and n_tgt > 0:
            variants.append((f"scale{n_tgt/n_ours:.2f}",
                             dict(pause_scale=n_tgt / max(n_ours, 1))))
        for vname, kw in variants:
            y, _ = pba.reshuffle(x0, SR, PBA_CV,
                                 target_samples=samples[ci["gender"]], **kw)
            for df in durmodes:
                if df:
                    r = (len(y) / float(SR)) / dur_tgt
                    if abs(r - 1.0) > 1e-4:
                        y2 = librosa.effects.time_stretch(
                            y.astype(np.float32), rate=float(r))
                    else:
                        y2 = y
                else:
                    y2 = y
                p = os.path.join(RT, f"{short}_{vname}_d{int(df)}.wav")
                sf.write(p, y2, SR)
                m = measure(p, "T")
                s, dv = score(cm, m)
                rec["variants"][f"{vname}|d{int(df)}"] = dict(score=s, devs=dv)
                print(f"   {vname:<10} durfix={int(df)}  {s:6.2f} "
                      f"({s - s0:+.2f})   sil {dv['sil']:5.2f}({d0['sil']:5.2f})"
                      f"  F {dv['F']:5.2f}({d0['F']:5.2f})"
                      f"  dur {dv['dur']:5.2f}({d0['dur']:5.2f})"
                      f"  D {dv['D']:5.2f}({d0['D']:5.2f})", flush=True)
        out[short] = rec

    # ---- 汇总 ----
    print("\n" + "=" * 78)
    print(f"{'方案':<18}{'均值':>8}" + "".join(f"{s:>10}" for s in out))
    keys = sorted({k for r in out.values() for k in r["variants"]})
    print(f"{'base':<18}{sum(r['base'] for r in out.values())/len(out):>8.2f}"
          + "".join(f"{r['base']:>10.2f}" for r in out.values()))
    best = None
    for k in keys:
        vs = [r["variants"][k]["score"] for r in out.values()]
        mu = sum(vs) / len(vs)
        print(f"{k:<18}{mu:>8.2f}" + "".join(f"{v:>10.2f}" for v in vs))
        if best is None or mu > best[1]:
            best = (k, mu)
    print(f"\n最优：{best[0]}  均值 {best[1]:.2f}")
    # 逐项失分变化
    bk = best[0]
    print(f"\n{bk} 相对 base 的逐项失分变化（负=变好）：")
    for d in ["A", "B", "E", "D", "F", "sil", "C", "dur", "lufs", "peak",
              "tone"]:
        dd = []
        for r in out.values():
            x = r["variants"][bk]["devs"].get(d)
            y = r["devs"].get(d)
            if x is not None and y is not None:
                dd.append(x - y)
        if dd:
            print(f"  {d:<5} w={WEIGHTS[d]:.2f}  "
                  + " ".join(f"{v:+6.2f}" for v in dd)
                  + f"   加权 {sum(dd)/len(dd)*WEIGHTS[d]:+.3f}")
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "devs"}
               for k, v in out.items()},
              open("/mnt/tts/diag/try_pause.json", "w"), ensure_ascii=False,
              indent=1)


if __name__ == "__main__":
    main()
