# -*- coding: utf-8 -*-
"""多案例 EQ 联合拟合：把 tone（音色 LTAS 偏差）降到最小（r25e）。

## 为什么不能用「逐段独立调」——实测踩到的坑

LTAS 是 `10*log10(p / sum(p))`，**按总能量归一**，所以它是零和的。实测
（在 24k 上给 superequalizer 单段 +6 dB）：

    提升 740Hz   -> 它所在的段只涨 4.28 dB，其余 14 段**全部跌 1.05 dB**
    提升 65Hz    -> 几乎不动（0.00）：那段本来就没能量
    提升 8372Hz  -> 影响落在第 13 段(5920-8372)，不是第 14 段

而且 tone_dev 的段划分是 `[65,92) [92,131) ...`（把中心频率当**下边界**用），
与 superequalizer 以这些频率为**中心**的滤波器天然错位。

后果：**逐段按差值补偿会正反馈发散** —— 第一轮 1.073 dB，第三轮跑到 5.0 dB。
（同上 jede迭代都想"抬自己"，但抬任何一段都会压所有其他段。）

## 正确做法：测响应矩阵 + 最小二乘联立求解

    R[i][j] = 给第 j 段 +6 dB 时，第 i 段的 dB 变化 / 6     （含串扰与归一化）
    解  R @ x = 残差(num_i = 目标_i - 当前_i)，  x = 每段需要的 dB

这样一次就把"抬这段会压那段"算进去了。

## 口径铁律

1. 🔴 **与验收同源**：直接用 `tools.fidelity` 的 `load / ltas_db / EQ_BANDS`。
2. 🔴 **叠加位置必须等于线上**（陷阱 #17）：线上 EQ 作用在 **24k** 上，之后才
   `aresample=44100` → 空气感 → volume → 限幅 → mp3。所以这里把成品 mp3
   先**下采样到 24k** 再施 EQ，然后重采样回 44.1k，而不是图省事在 44.1k 成品上叠。
3. 🔴 **只叠「补偿倍数」**（陷阱 #15）：成品已打过一次 EQ，再打 = 双倍。
   最终线上值 = 旧 EQ × 这里求出的补偿。
4. 🔴 **185Hz（4b）永不改**：基频区，改了丢说话人身份（r18 结论）。

## 用法（服务器 qwen 环境）

    venv/bin/python -u tools/eqfit.py --prefix r25d --iters 3
    venv/bin/python -u tools/eqfit.py --prefix r25d --only 00b6a4
"""
import os
import sys
import glob
import argparse
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fidelity import load, ltas_db, EQ_BANDS, LTAS_FMAX   # noqa: E402

RT = "/mnt/tts/diag/eqwork"
CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")

PAIRS = [
    ("1a6c20a60481", "1a6c20", "male"),
    ("ba24c0925311", "ba24c0", "male"),
    ("8197d9ac8a35", "8197d9", "male"),
    ("6cdd25bc75f2", "6cdd25", "male"),
    ("00b6a45ad6aa", "00b6a4", "female"),
]

N_BAND = len(EQ_BANDS)          # 15（superequalizer 的 1b..15b）
SKIP_BAND = 3                   # 185 Hz，永不改
SR_EQ = 24000                   # 🔴 线上 EQ 就作用在 24k
SR_OUT = 44100
PROBE_DB = 6.0                  # 测响应矩阵时的扰动幅度
MAX_GAIN_DB = float(os.environ.get("TTS_EQFIT_MAXDB", "6.0"))
# 🔴 低频前 3 段(65/92/131Hz)单独限幅：抬它会把**停顿里的底噪**一起抬上去
#    → 验收 VAD 用相对阈值 p25+6dB → 判句/停顿检出漂移 → D 和 sil 恶化
#    （r25e 实测：tone 赚 0.78，但 D 亏 0.43、sil 亏 0.16）。
#    实测低频不动时 tone 反而更好（6.1 vs 8.3）—— 那几列响应本来就不可靠
#    （65Hz 完全无响应、92Hz 很弱），硬补只会引入误差。
LOW_BANDS = [0, 1, 2]
LOW_MAX_DB = float(os.environ.get("TTS_EQFIT_LOWDB", "6.0"))               # 每段最终增益上限（防止病态解）
MIN_COL_RESP = 0.05             # 某段最大响应低于此值 = 该段无效，不解


def _run(cmd):
    subprocess.run(cmd, check=True, timeout=1800,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def concat_mp3(files, out_wav, sr):
    lst = out_wav + ".txt"
    with open(lst, "w") as w:
        for f in files:
            w.write("file '%s'\n" % f)
    _run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
          "-i", lst, "-ac", "1", "-ar", str(sr), out_wav])


def apply_gain(src24, gains18, dst_wav, to44=True):
    parts = ":".join(f"{i + 1}b={v:.5f}" for i, v in enumerate(gains18))
    af = f"superequalizer={parts}"
    if to44:
        af += f",aresample={SR_OUT}"
    _run(["ffmpeg", "-v", "error", "-y", "-i", src24, "-af", af,
          "-ac", "1", "-ar", str(SR_OUT if to44 else SR_EQ), dst_wav])


def to_mp3(wav, mp3):
    _run(["ffmpeg", "-v", "error", "-y", "-i", wav,
          "-codec:a", "libmp3lame", "-b:a", "192k", "-ar", str(SR_OUT), mp3])


def grid_of(path):
    """LTAS 在 tone_dev 的 15 段划分上的值（dB）。"""
    x, sr = load(path, sr0=SR_EQ)
    db, mf = ltas_db(x, sr)
    out = []
    for lo_, hi_ in zip(EQ_BANDS, EQ_BANDS[1:] + [LTAS_FMAX]):
        s = (mf >= lo_) & (mf < hi_)
        out.append(float(np.mean(db[s])) if int(s.sum()) else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="r25d")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(RT, exist_ok=True)
    result = {}

    for gender in ("male", "female"):
        pairs = [p for p in PAIRS if p[2] == gender]
        if a.only:
            pairs = [p for p in pairs if p[1] in a.only.split(",")]
        if not pairs:
            continue
        print(f"\n{'=' * 78}\n{gender}：{len(pairs)} 例 "
              f"{[p[1] for p in pairs]}\n{'=' * 78}", flush=True)

        # ---- 准备：案例目标 + 产物源（24k）------------------------------
        case_g, srcs = [], []
        for cid, short, _g in pairs:
            cw = os.path.join(CASES_DIR, cid, "merged.wav")
            fs = sorted(glob.glob(f"/mnt/tts/output/{a.prefix}-{short}/*.mp3"))
            if not os.path.exists(cw) or not fs:
                print(f"  !! 跳过 {short}（案例/产物缺失）", flush=True)
                continue
            src = os.path.join(RT, f"src24_{short}.wav")
            concat_mp3(fs, src, SR_EQ)
            case_g.append(grid_of(cw))
            srcs.append((short, src))
        if not srcs:
            continue
        target = np.nanmean(np.stack(case_g), axis=0)

        # ---- 基准（不施加补偿）------------------------------------------
        def evaluate(gains, rate=True):
            gs = []
            for short, src in srcs:
                tag = f"{short}_{abs(hash(tuple(np.round(gains, 4)))) % 10**8}"
                wav = os.path.join(RT, f"eq_{tag}.wav")
                apply_gain(src, gains, wav)
                if rate:
                    mp3 = os.path.join(RT, f"eq_{tag}.mp3")
                    to_mp3(wav, mp3)
                    gs.append(grid_of(mp3))
                else:
                    gs.append(grid_of(wav))
            return np.nanmean(np.stack(gs), axis=0), gs

        gains = np.ones(18)
        cur, base_per = evaluate(gains)
        base_dev = float(np.mean(np.abs(target - cur)))
        base_each = tuple((s_, float(np.mean(np.abs(target - g))))
                          for s_, g in zip([x for x, _ in srcs], base_per))
        print(f"  基准 tone 偏差 {base_dev:.3f} dB  逐例 "
              + " ".join(f"{n}={v:.3f}" for n, v in base_each), flush=True)

        # ---- 测响应矩阵 R[i][j]（用第一个源，wav 口径够快）---------------
        # 注意要在**已叠加当前补偿**的基础上扰动才有意义，但 R 在小范围内近似
        # 不变，所以只在基准处测一次。
        base_grid_src = None
        for short, src in srcs:
            if short == srcs[0][0]:
                base_grid_src = grid_of(src)
                break
        R = np.zeros((N_BAND, N_BAND))
        for j in range(N_BAND):
            if j == SKIP_BAND:
                continue
            g = np.ones(18)
            g[j] = 10.0 ** (PROBE_DB / 20.0)
            wav = os.path.join(RT, f"probe_{j}.wav")
            apply_gain(srcs[0][1], g, wav)
            R[:, j] = (grid_of(wav) - base_grid_src) / PROBE_DB
        print("  响应矩阵对角/列最大：", flush=True)
        print("    diag " + " ".join(f"{R[i, i]:>6.2f}" for i in range(N_BAND)))
        print("    colmx " + " ".join(
            f"{np.nanmax(np.abs(R[:, j])):>6.2f}" for j in range(N_BAND)))
        alive = [j for j in range(N_BAND)
                 if not np.allclose(R[:, j], 0)
                 and np.nanmax(np.abs(R[:, j])) > MIN_COL_RESP]
        print(f"    有效段 {len(alive)}/15（其余四舍五入后无响应，跳过）")

        # ---- 迭代：最小二乘求每段 dB -------------------------------------
        gains_db = np.zeros(18)
        history = [(gains.copy(), base_dev, base_each)]
        for it in range(1, a.iters + 1):
            res = target - cur
            A = R[:, alive]
            x, *_ = np.linalg.lstsq(A, res, rcond=None)
            step = np.zeros(N_BAND)
            for k, j in enumerate(alive):
                lim = LOW_MAX_DB if j in LOW_BANDS else MAX_GAIN_DB
                step[j] = np.clip(float(x[k]), -lim, lim)
            step = np.clip(step, -MAX_GAIN_DB, MAX_GAIN_DB)
            step[SKIP_BAND] = 0.0
            gains_db[:N_BAND] += step
            gains[:N_BAND] = 10.0 ** (np.clip(gains_db[:N_BAND],
                                              -MAX_GAIN_DB, MAX_GAIN_DB) / 20.0)
            cur, per = evaluate(gains)
            dev = float(np.mean(np.abs(target - cur)))
            tag_used = min(history, key=lambda h: h[1])[1]
            print(f"  轮{it}  step "
                  + " ".join(f"{v:+.1f}" for v in step)
                  + f"   偏差 {dev:.3f} dB"
                  + ("  <-- 最优" if dev <= tag_used else ""), flush=True)
            history.append((gains.copy(), dev,
                            tuple((s, float(np.mean(np.abs(target - g))))
                                  for s, g in zip([x for x, _ in srcs], per))))
            if dev > base_dev * 2.5:      # 明显失控，早停
                print("  !! 偏差暴涨，早停", flush=True)
                break

        bg, bd, bper = min(history, key=lambda h: h[1])
        print(f"\n  全程最优 {bd:.3f} dB（起点 {base_dev:.3f}）", flush=True)
        print("  每段累计 dB: " + " ".join(
            f"{v:+.2f}" for v in 20 * np.log10(np.maximum(bg[:N_BAND], 1e-6))))
        print("  补偿倍数 1b..15b:")
        print("    " + "/".join(f"{v:.4f}" for v in bg[:N_BAND]))
        w = 0.07
        print(f"  ≈ 还原度收益 {(base_dev - bd) / 6 * 100 * w:+.2f} 分")
        result[gender] = dict(mult=[round(v, 4) for v in bg[:N_BAND]],
                              base=round(base_dev, 3), best=round(bd, 3))

    print("\n" + "=" * 78)
    print("把下面的**倍数逐段乘到** server.py 里对应的 TONE_EQ_FILTER_* 上")
    print("（⚠️ 是乘，不是替换 —— 成品已打过一次 EQ，叠完整新 EQ = 双倍，陷阱 #15）")
    for g, r in result.items():
        print(f"\n{g.upper()}: " + "/".join(f"{v:.3f}" for v in r["mult"]))


if __name__ == "__main__":
    main()
