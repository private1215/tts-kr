# -*- coding: utf-8 -*-
"""d 轮（20261013a）反馈片段：让员工用**他认得的那条声音**做判断。

## 为什么重出这轮
b 轮（20261012b）员工 6/7 票选的是 `fb_{f,m}_cosy_head.mp3`，而那个文件的源是
**裸推理** `cosy_full_{f,m}.wav`（campplus vs 标杆 **0.9532**）。
线上实际交付的成品走完整后处理，只有 **0.8444**。
→ **员工认可的那条声音，线上根本没在交付**（差 0.109，已定案是主服务
  250s 分块链路造成的，不是引擎、不是参考音频、不是后处理单项）。

c 轮（20261012c）没用员工选中的那条，换成来源不明的 `fb2_*`，员工当场听出
"不是昨天选的那些" → 那轮信号作废。

## 本轮怎么出
- **参照（目标嗓音）** = EL 标杆原声（我们要对齐的目标，不是"真人原声"）
- **A（第 1 段）** = 线上现在交付的版本
- **B（第 2 段）** = 裸推理版本（就是员工昨天选中的那条声音）
- 员工若选 B，就等于批准"把线上链路改成一次成型"，我们据此动刀。

## 两个坑（都踩过）
1. **文案对齐**：两版语速不同（裸推理 445.7s / 线上 490.1s），按同一时间点切
   会切到不同的句子。这里按**时长比例**换算起点，保证两段念的是同一句。
2. **不能做 loudnorm**：单趟动态 loudnorm 会把 −50dB 的停顿泵到 −45dB 以上，
   "卡顿"特征被抹平（c 轮教训）。这里只做**静态增益**对齐整条 RMS——
   静态增益不改变动态，卡顿特征完整保留。

用法：
    python -u tools/mk_feedback_clips_d.py            # 切
    python -u tools/mk_feedback_clips_d.py --dry-run  # 只看计划
"""
import argparse
import os
import subprocess

import numpy as np
import soundfile as sf

OUT_DIR = os.environ.get("FB_CLIP_DIR", "/mnt/tts/data/tone-ab")
GSOUT = "/mnt/tts/projects/gpt-sovits/out"
CASES = "/mnt/tts/data/cases"
TMP = "/tmp/fbclipd"
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

CLIP_SS = 10.0        # 起点秒（以"基准"那版为准）
CLIP_DUR = 40.0

# (性别, 案例 merged.wav【目标嗓音】, 线上成品【A】, 裸推理【B】)
GROUPS = [
    ("f", "%s/00b6a45ad6aa/merged.wav" % CASES,
     "/tmp/selftest_open/open_female.wav",
     "%s/cosy_full_f.wav" % GSOUT),
    ("m", "%s/1a6c20a60481/merged.wav" % CASES,
     "/tmp/selftest_open/open_male.wav",
     "%s/cosy_full_m.wav" % GSOUT),
]


def dur_of(p):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", p],
        capture_output=True, text=True).stdout.strip())


def rms_dbfs(p):
    """整条 RMS（dBFS）。用于静态增益对齐，不做任何动态处理。"""
    w = os.path.join(TMP, "rms.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", "44100", "-ac", "1", w], check=True)
    x, _ = sf.read(w)
    x = np.asarray(x, dtype=np.float64)
    return 20.0 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def make(src, dst, ss, dur, gain_db=0.0):
    """裸切 + 可选静态增益。**绝不** loudnorm / compand / dynaudnorm。"""
    af = []
    if abs(gain_db) > 0.05:
        af.append("volume=%.3fdB" % gain_db)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y",
           "-ss", str(ss), "-t", str(dur), "-i", src]
    if af:
        cmd += ["-af", ",".join(af)]
    cmd += ["-ar", "44100", "-ac", "1",
            "-codec:a", "libmp3lame", "-b:a", "192k", dst]
    subprocess.run(cmd, check=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    report = []
    for g, ref, online, bare in GROUPS:
        for p in (ref, online, bare):
            if not os.path.exists(p):
                print("⚠️ 缺文件", p)
        d_on, d_bare = dur_of(online), dur_of(bare)
        # 🔴 文案对齐：两版语速不同，按总时长比例换算起点
        ratio = d_on / d_bare
        ss_bare, ss_on = CLIP_SS, CLIP_SS * ratio
        r_on, r_bare = rms_dbfs(online), rms_dbfs(bare)
        # 把 B 静态增益对齐到 A（只改整体电平，不动动态）
        gain = r_on - r_bare
        print("\n== %s ==" % g)
        print("   时长 线上 %.1fs / 裸推理 %.1fs  比例 %.4f" % (d_on, d_bare, ratio))
        print("   切片起点 线上 %.1fs / 裸推理 %.1fs" % (ss_on, ss_bare))
        print("   RMS 线上 %.2f / 裸推理 %.2f  → 静态增益 %+.2f dB" %
              (r_on, r_bare, gain))
        report.append((g, r_on, r_bare, gain))
        if a.dry_run:
            continue
        make(ref, os.path.join(OUT_DIR, "fd_%s_target_head.mp3" % g),
             CLIP_SS, CLIP_DUR)
        make(online, os.path.join(OUT_DIR, "fd_%s_A_online_head.mp3" % g),
             ss_on, CLIP_DUR)
        make(bare, os.path.join(OUT_DIR, "fd_%s_B_direct_head.mp3" % g),
             ss_bare, CLIP_DUR, gain)
        print("   -> fd_%s_{target,A_online,B_direct}_head.mp3" % g)

    print("\n=== 响度核对（必须 <1dB，否则盲测不公平）===")
    for g, r_on, r_bare, gain in report:
        print("   %s: 线上 %.2f  dBFS / 候选 %.2f dBFS  差 %.2f dB"
              % (g, r_on, r_bare, abs(r_on - r_bare)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
