# -*- coding: utf-8 -*-
"""离线试算：软削波把峰值/CF 拉回 r22 的水平，看还原度怎么变。

## 为什么需要它

r23 上线「停顿预算」后 CF（真峰值 − LUFS）系统性涨了 1.6 ~ 3.4 dB：

    r22-ba24c0  LUFS -24.80  真峰值 -2.70  CF 22.10
    r23-ba24c0  LUFS -24.90  真峰值 -0.90  CF 24.00
    r22-00b6a4  LUFS -22.70  真峰值 -4.00  CF 18.70
    r23-00b6a4  LUFS -22.70  真峰值 -0.60  CF 22.10

峰值偏差随之暴涨（ba24c0 6.7% -> 36.7%），按权重 0.06 折算就是 -1.8 个点，
这是 r23 没能冲上 94% 的最大单项原因。

## 为什么"调限幅器"没用

实测 ffmpeg 的 `alimiter=limit=X` 在这个版本里**是纯增益 output = input / X**
（输入 0.771 / 0.386 / 0.154，输出一律 +2.05 dB，与"限幅"无关）。所以改
limit 只是整体改电平，响度闭环会 1:1 补回来，**CF 一点不变**。
CF 是信号固有的，只能靠真的动态处理去压。

## 试什么

软削波（soft clip）：阈值 t 以下完全线性不动，以上 tanh 渐进压到 1.0。
只碰最响的那一小撮样本，LUFS 几乎不掉，峰值直接下来 —— 与"压慢包络"
（r21 已证伪：会把 C 一起压掉、音色恶化）不是一回事。

用法（服务器，qwen 环境）：
    venv/bin/python tools/try_clip.py --prefix r23 --grid 0.7,0.6,0.5   # 旧：扫固定拐点
    venv/bin/python tools/try_clip.py --prefix r25a --mode q             # 新：扫分位数拐点

## r25b 为什么要换算法

旧 `softclip` 的渐近上限是 1.0，要把峰值从 ~0.93 压到 ~0.76，**反解出的拐点必然
掉到 KNEE_MIN=0.35（−9 dBFS）** → −9 dBFS 以上全被压（真实语音里占比可达 10%）
→ 能量分布上端改动 → 相对阈值 VAD 判句漂移，E/F 崩（r24：94.34→92.77，净值 −1.57）。

新算法反过来：**拐点由样本分位数直接指定**（默认 p99.9 → 只碰最响的 0.1%），
拿**渐近上限**当自由度去凑目标峰值（见 `app/peakfit.py` 的 `solve_asym`）。
实测 RMS 只变 −0.0045 dB → LUFS 不动 → 响度闭环不会补增益。
"""
import os
import sys
import glob
import argparse

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
from tools.fidelity import measure, score, WEIGHTS           # noqa: E402
import peakfit                                               # noqa: E402

SR = 44100
CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")
RT = "/mnt/tts/diag/rtwork"
CASES = [
    ("1a6c20a60481", "1a6c20"),
    ("00b6a45ad6aa", "00b6a4"),
    ("ba24c0925311", "ba24c0"),
    ("8197d9ac8a35", "8197d9"),
    ("6cdd25bc75f2", "6cdd25"),
]


def softclip(x, t):
    """阈值 t 以下线性，以上 tanh 渐进压到 1.0。连续、无硬拐点。"""
    a = np.abs(x)
    s = np.sign(x)
    over = a > t
    if not over.any():
        return x
    y = np.where(over, t + (1.0 - t) * np.tanh((a - t) / (1.0 - t)), a)
    return (s * y).astype(np.float32)


def to_mp3(x, sr, out):
    import subprocess
    w = out.replace(".mp3", ".wav")
    sf.write(w, x, sr, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", w,
                    "-codec:a", "libmp3lame", "-b:a", "192k",
                    "-ar", str(sr), out], check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="r23")
    ap.add_argument("--grid", default="0.7,0.6,0.5")
    ap.add_argument("--only", default="")
    # r25b 新增：q = 用 peakfit 的分位数拐点算法，grid 里给的是 PEAK_Q 分位数
    ap.add_argument("--mode", default="t", choices=["t", "q"])
    ap.add_argument("--target", default="", help="覆盖目标峰值 dBFS，逗号分隔（按案例顺序）")
    a = ap.parse_args()
    grid = [float(v) for v in a.grid.split(",")]
    tgts = ([float(v) for v in a.target.split(",")]
            if a.target else None)

    res = {}
    tags = []
    for i, (cid, short) in enumerate(CASES):
        if a.only and short not in a.only.split(","):
            continue
        fs = sorted(glob.glob(f"/mnt/tts/output/{a.prefix}-{short}/*.mp3"))
        if not fs:
            continue
        src = os.path.join(RT, f"clip_{a.prefix}_{short}.wav")
        import subprocess
        lst = src + ".txt"
        with open(lst, "w") as w:
            for f in fs:
                w.write("file '%s'\n" % f)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat",
                        "-safe", "0", "-i", lst, "-ac", "1", "-ar", str(SR),
                        src], check=True)
        cm = measure(os.path.join(CASES_DIR, cid, "merged.wav"), "CASE")
        x, _ = sf.read(src, dtype="float32")
        # 基准：原样再编一遍 mp3（隔离"多一次 mp3 编码"本身的影响）
        base_mp3 = to_mp3(x, SR, os.path.join(RT, f"clip_{short}_base.mp3"))
        bm = measure(base_mp3, "BASE")
        s0, d0 = score(cm, bm)
        rec = {"base": s0, "vars": {}}
        print(f"\n[{short}] base={s0:.2f}  LUFS {bm['lufs']:.2f} "
              f"峰值 {bm['peak']:.2f}  CF {bm['peak']-bm['lufs']:.2f}",
              flush=True)
        for gv in grid:
            if a.mode == "q":
                # 目标 = 案例自己的峰值（验收口径就是跟案例比）
                tgt = tgts[i] if tgts else cm["peak"]
                peakfit.PEAK_Q = gv
                y = peakfit.fit(x, tgt)
                tag = f"q{gv}"
            else:
                y = softclip(x, gv)
                tag = f"t{gv}"
                tgt = None
            p = to_mp3(y, SR, os.path.join(RT, f"clip_{short}_{tag}.mp3"))
            m = measure(p, "T")
            s, dv = score(cm, m)
            rec["vars"][tag] = dict(score=s, devs=dv,
                                    lufs=m["lufs"], peak=m["peak"])
            # 🔴 E/F 是 r24 崩掉的两项，必须盯住（VAD 相对阈值漂移）
            print(f"   {tag:<8} {s:6.2f} ({s - s0:+.2f})  "
                  f"LUFS {m['lufs']:.2f}({bm['lufs']:.2f})  "
                  f"峰值 {m['peak']:.2f}({bm['peak']:.2f})  "
                  f"peak偏差 {dv['peak']:.1f}({d0['peak']:.1f})  "
                  f"E {dv['E']:.1f}({d0['E']:.1f})  "
                  f"F {dv['F']:.1f}({d0['F']:.1f})  "
                  f"C {dv['C']:.1f}({d0['C']:.1f})  "
                  f"tone {dv['tone']:.1f}({d0['tone']:.1f})", flush=True)
        res[short] = rec

    print("\n" + "=" * 74)
    print(f"{'方案':<10}{'均值':>8}" + "".join(f"{s:>10}" for s in res))
    print(f"{'base':<10}{sum(r['base'] for r in res.values())/len(res):>8.2f}"
          + "".join(f"{r['base']:>10.2f}" for r in res.values()))
    for r in res.values():
        tags = list(r["vars"])
        break
    for tag in tags:
        vs = [r["vars"][tag]["score"] for r in res.values()]
        print(f"{tag:<10}{sum(vs)/len(vs):>8.2f}"
              + "".join(f"{v:>10.2f}" for v in vs))


if __name__ == "__main__":
    main()
