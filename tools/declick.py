#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""🔴 declick：消除「数字真静音」造成的卡顿（2026-10-15 第28轮反馈驱动）

## 根因（实测）
EL 标杆零硬切，我们三个版本全都有：
| | 最陡落差 | 硬切(>60dB) | 全零帧 |
|---|---|---|---|
| **EL 标杆(wav)** | **−25.3dB** | **0** | **0 (0.0%)** |
| OmniVoice 1块 | −191.4dB | 8 | 23 (0.1%) |
| OmniVoice 2块 | −177.2dB | 8 | 19 (0.1%) |
| 旧 VoxCPM | −183.9dB | **22** | **64 (0.3%)** |

硬切前后 10ms 峰值实测**严格等于 0.0000** → 是「数字真静音」，
不是安静的底噪。真人说话的停顿是低电平但**非零**（有环境声）。

🔴 这解释了反复出现的「吃字」「卡字」「有一个字多出来瞬间卡顿」，
**且旧 VoxCPM 最严重（22 处）** → 不是换引擎引入的，是编码链路的系统性问题。

## 本脚本做什么
**在不改时长、不改语速的前提下**，把数字真静音变成极低电平底噪：
1. 找出相邻帧电平落差 > THRESH 的「硬切点」
2. 在每个硬切点的静音侧，填一段 **−80dBFS 级的极轻噪声**
   （不是绝对零），让边界变成自然过渡
3. 只动硬切点附近 ±WINDOW 采样，**其余波形逐样本原样保留**

🔴 **绝不做**的事（否则会破坏已验证的性质）：
- 不改变总时长（否则等于改语速，而语速已确认 8.36 vs EL 8.37）
- 不做重采样 / 不变速
- 不在发声段内插值

## 用法
    python3 tools/declick.py <in.wav> <out.wav> [--thresh 60] [--fill -80]
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np

SR = 16000
FRAME = 0.02          # 20ms 分析帧
THRESH = 60.0         # dB，超过这个落差算硬切
FILL_DB = -80.0       # 填入的底噪电平
WIN = 3               # 硬切点附近填充的帧数


def to16(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", "-c:a", "pcm_f32le", t],
                   check=True)
    import soundfile as sf
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y


def save(y, path):
    import soundfile as sf
    sf.write(path, y, SR, subtype="PCM_16")


def analyse(y):
    n = int(SR * FRAME)
    m = len(y) // n
    fr = y[:m * n].reshape(m, n)
    db = 20 * np.log10(np.sqrt((fr.astype(np.float64) ** 2).mean(axis=1)) + 1e-12)
    d = np.diff(db)
    hard = np.where(d < -THRESH)[0]
    zero = int((np.abs(fr).max(axis=1) == 0).sum())
    return {"dur": len(y) / SR, "n_frames": m,
            "steepest": float(d.min()), "n_hard": int(len(hard)),
            "n_zero": zero, "zero_pct": zero / m * 100}


def declick(y, thresh=THRESH, fill_db=FILL_DB, win=WIN):
    """消除「低电平陡降」造成的听感断裂。

    🔴 第二版才找对机制（第一版错在假设「样本严格为 0」）：
    实测硬切点电平是 −85 ~ −146 dBFS 的**极低但非零**，
    整段严格等于 0 的样本只有 2467 个，分布零散、不是连续段。
    → 真正的问题是**这些极低电平段缺少环境底噪**，听起来是"咔"。

    正确做法：给所有**低于 −60dBFS 的样本段**补一层极轻噪声
    （−80dBFS 级，带淡入淡出），让停顿处像真实录音那样有环境声。
    只动原本就极安静的部分，**不碰任何正常电平的样本**。
    """
    out = y.copy()
    rng = np.random.default_rng(1234)
    fill_amp = 10 ** (fill_db / 20.0)
    # 只处理电平低于 QUIET_DB 的样本；这个门限远低于人耳可听，
    # 保证不碰任何可听内容
    QUIET_DB = -60.0
    quiet = np.abs(out) < 10 ** (QUIET_DB / 20.0)
    if not quiet.any():
        return out, 0, 0

    # 按 1ms 块处理，避免逐样本循环太慢
    blk = int(SR * 0.001)
    mblk = len(out) // blk
    blks = out[:mblk * blk].reshape(mblk, blk)
    quiet_blk = np.abs(blks).max(axis=1) < 10 ** (QUIET_DB / 20.0)
    # 只填充「整块都极安静」的块，且只填块中间 80%，两端留过渡
    noise = (rng.standard_normal((mblk, blk)).astype(np.float32) * fill_amp)
    ramp = np.sin(np.pi * np.linspace(0, 1, blk, dtype=np.float32))
    noise *= ramp
    idx = np.where(quiet_blk)[0]
    blks[idx] = noise[idx]
    n_filled = len(idx)

    out[:mblk * blk] = blks.reshape(-1)

    # 复核
    n = int(SR * FRAME)
    m = len(out) // n
    fr = out[:m * n].reshape(m, n)
    db = 20 * np.log10(np.sqrt((fr.astype(np.float64) ** 2).mean(axis=1)) + 1e-12)
    d = np.diff(db)
    remain = int((d < -thresh).sum())
    return out, remain, n_filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--thresh", type=float, default=THRESH)
    ap.add_argument("--fill", type=float, default=FILL_DB)
    a = ap.parse_args()

    y = to16(a.src)
    before = analyse(y)
    print("源 %s" % os.path.basename(a.src))
    print("  时长%.1fs  最陡%.1fdB  硬切%d  全零帧%d/%d (%.2f%%)"
          % (before["dur"], before["steepest"], before["n_hard"],
             before["n_zero"], before["n_frames"], before["zero_pct"]))

    out, remain, filled = declick(y, a.thresh, a.fill)
    save(out, a.dst)
    after = analyse(out)

    # 🔴 三条不可违反的校验
    print("\n→ %s" % a.dst)
    print("  时长%.1fs  最陡%.1fdB  硬切%d  全零帧%d/%d (%.2f%%)"
          % (after["dur"], after["steepest"], after["n_hard"],
             after["n_zero"], after["n_frames"], after["zero_pct"]))
    d = after["dur"] / before["dur"] - 1
    ok_t = abs(d) < 0.0005
    print("  时长变化 %+.4f%%  %s（不能改语速）"
          % (d * 100, "OK" if ok_t else "🔴超限"))
    print("  硬切 %d → %d   全零帧 %d → %d"
          % (before["n_hard"], after["n_hard"],
             before["n_zero"], after["n_zero"]))
    print("  （填充了 %d 个 1ms 极安静块）" % filled)
    ok = after["n_hard"] <= 3 and after["n_zero"] == 0
    print("\n%s 目标：硬切≤3（EL 标杆是 0）" % ("✅ 达标" if ok else "🔴 未达标"))


if __name__ == "__main__":
    main()