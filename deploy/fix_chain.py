#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用修复链（把 raw 合成结果加工到可交付）——新文案版。

与 female_fix_v3 / male_fix_v3 的差别：那两个脚本硬要求「同一篇文案的 EL 参考」，
拿来做逐窗速率对齐。但**新文案没有 EL 参考**（我们无法生成新的 ElevenLabs 音频），
所以这里支持两种定速率的方式：
  --el <wav>          有同文本 EL：逐窗对齐（原口径，最准）
  --target-rate <r>   无 EL：对齐到 EL 的**典型速率**（默认 女8.5 / 男8.9）
                      这两个数来自 52727ec43154(女) 与 8197d9ac8a35(男) 的实测均值。

流程（与获胜链路一致，顺序不能换）：
  ① 逐 60s 窗音节核速率对齐 → 在静音处切窗做 atempo（只放慢，不加快，钳 [1.0,1.30]）
  ② 停顿清洗：≥0.08s 停顿内容整体替换为合成 clean room tone（长度不变、5ms 淡化）
  ③ 男声压亮 EQ（highshelf 4000 -4dB, lowpass 14000）；**女声不加 EQ**
     —— v2 实测女声频谱已与 EL 对齐，加 EQ 反而更差
  ④ 响度对齐到目标 LUFS（女 -22.6 / 男 -25.0，取自 EL 实测）

用法:
  fix_chain.py <raw.wav> <out.wav> [--gender f|m] [--el EL.wav]
               [--target-rate 8.5] [--lufs -22.6] [--no-clean]
"""
import argparse
import os
import re
import subprocess
import sys
import shutil
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "..", "diag", "cross-20261013"),
          "/mnt/tts/diag/cross-20261013"):
    if p not in sys.path:
        sys.path.insert(0, p)

from pause_clean_v3 import clean_gaps, detect_gaps_wav, verify_pauses  # noqa: E402

WIN = 60.0
# EL 典型速率（音节核/秒，实测均值）：女 8.51 / 男 8.93
DEFAULT_RATE = {"f": 8.5, "m": 8.9}
# 🔴 EL 实测「字/秒」（chars per second）：女 7.45 / 男 8.37
#   这才是**跨文案可比**的锚。2026-10-14 踩坑：用音节核速率当固定目标时，
#   男声成品 9.17 字/秒 vs EL 8.37，快了 9.6%，员工直接判「不像真人」。
#   老脚本没出事，是因为它们拿**同一篇文案的 EL** 逐窗对齐，量纲差互相抵消；
#   一旦换成固定常数，音节核/秒 ≠ 字/秒 的差异就暴露了。
# 🔴 女声 2026-10-14 复核：4 条 EL 案例实测 7.45 / 7.622 / 7.507 / 7.612 → 均值 7.573。
#   之前锚在单条案例的 7.45，等于我们比 EL 快 1.6%，员工连续两轮点名「女声太快」。
DEFAULT_CPS = {"f": 7.57, "m": 8.37}
# EL 实测响度：女 -22.6 / 男 -25.0
DEFAULT_LUFS = {"f": -22.6, "m": -25.0}
DEFAULT_F0 = {"f": 276.0, "m": 197.7}
# 🔴 变调会把整个频谱搬上去：男声升 1.21 倍后 >4kHz 能量占比 0.0185，
#    而 EL 只有 0.0104（高 78%，听感发尖）。10*log10(0.0104/0.0185) = -2.5dB，
#    所以变调后高频要压到 -6.5dB。未变调时保持默认 -4dB。
#    ❌ rubberband 的 formant=preserved（只升基频不动声道）**不可用**：
#       频谱平坦度 0.050 → 0.015（EL 0.042），声音发闷发假。
# 🔴 男声 EQ 三段式（2026-10-14 用 eq_search.py 网格搜出来的，不是拍脑袋）：
#    原单段 highshelf(4000,-4) 只会把高频整体压掉，结果能量全堆在 100~1kHz：
#      我们 100-1k 0.763 / 1k-4k 0.225，而 EL 是 0.712 / 0.275 —— 听感"厚而闷、不清晰"。
#    三段：lowshelf 800 -1.0（减低频堆积）+ highshelf 1500 +1.5（补中频清晰度）
#          + highshelf 4000 %g（压变调搬上来的高频）
#    网格搜索结果（24k 口径 60s 频段能量 L1 偏差）：
#      基线 0.1026 → 单提中频 0.033 → **加低频衰减+高频回压 0.0060**（flat 也 0.0378→0.0389）
#    变调会把频谱整体搬高（hf 0.0064→0.0185，EL 只有 0.0104），
#    所以变调时高频那一档要从 -4dB 加深到 -8dB。
EQ_MALE_TMPL = ("lowshelf=f=800:g=-1.0,highshelf=f=1500:g=1.5,"
                "highshelf=f=4000:g=%g,lowpass=f=14000")
# EL 实测 F0 中位（librosa.pyin / 16k 口径，与 tools 里的诊断同口径，不可换算法）：
#   男（13 条 EL 案例）190.5~205.3 → 均值 197.7
#   女（5 条 SOLA 案例）271.0~282.1 → 均值 276.0
DEFAULT_F0 = {"f": 276.0, "m": 197.7}
EQ_MALE = "highshelf=f=4000:g=-4,lowpass=f=14000"


def f0_median(y, sr, win=60.0):
    """F0 中位。必须与 EL 基准值同口径：pyin / 16k / fmin60 fmax400 / **前 60s**。
    🔴 2026-10-14 踩坑：这里取整篇、而 DEFAULT_F0 是前 60s 口径，
       全篇 F0 比开头高约 13Hz（185.1 vs 171.7），导致变调只做了 1.14 半音
       （应该 2.5 半音），音高根本没补够。跨口径数字害死人。"""
    n = int(win * sr)
    if len(y) > n:
        y = y[:n]
    try:
        import librosa
        f0, vo, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr,
                                 frame_length=1024, hop_length=256)
        m = (vo if vo is not None else np.zeros(len(f0), dtype=bool))
        m = m & np.isfinite(f0)
        v = f0[m]
        return float(np.median(v)) if len(v) > 50 else None
    except Exception as e:
        print("F0 测量失败（缺 librosa？）: %r" % (e,), flush=True)
        return None


def load16(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", t], check=True)
    y, sr = sf.read(t, dtype="float32")
    os.unlink(t)
    return y, sr


def syl_rates(y, sr):
    hop = int(0.01 * sr)
    n = len(y) // hop
    env = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean()
                            for i in range(n)]))
    env = np.convolve(env, np.ones(5) / 5, mode="same")
    thr = np.percentile(env, 30) * 2.2
    w = int(WIN * 100)
    out, peaks, voiced = [], 0, 0
    for i in range(n):
        if env[i] > thr:
            voiced += 1
        if (env[i] > thr and 0 < i < n - 1 and env[i] >= env[i - 1]
                and env[i] > env[i + 1]
                and env[i] - max(env[max(0, i - 8):i].min(),
                                 env[i + 1:i + 9].min() if i + 9 < n else 0)
                > thr * 0.35):
            peaks += 1
        if (i + 1) % w == 0:
            if voiced > 0:
                out.append(peaks / (voiced / 100.0))
            peaks, voiced = 0, 0
    return out


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    ms = re.findall(r"I:\s+(-?[\d.]+)\s+LUFS", r.stderr)
    return float(ms[-1]) if ms else -24.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--gender", default="f", choices=["f", "m"])
    ap.add_argument("--el", default="")
    ap.add_argument("--target-rate", type=float, default=None)
    ap.add_argument("--target-cps", type=float, default=None,
                    help="按「字/秒」锚定（跨文案可比，推荐）。需配合 --chars")
    ap.add_argument("--chars", type=int, default=0,
                    help="raw 对应的正文字数，--target-cps 模式下必填")
    ap.add_argument("--lufs", type=float, default=None)
    ap.add_argument("--pitch", type=float, default=0.0,
                    help="手动频率比（不是半音）：>1 升调。留空则按 --target-f0 自动算")
    ap.add_argument("--target-f0", type=float, default=None,
                    help="目标 F0 中位(Hz)。默认男 197.7 / 女 276.0（EL 案例实测均值）")
    ap.add_argument("--no-pitch", action="store_true",
                    help="跳过 F0 对齐（对照组用）")
    ap.add_argument("--eq-db", type=float, default=None,
                    help="覆盖男声 highshelf 增益(dB)。默认：变调时 -6.5，否则 -4")
    ap.add_argument("--no-clean", action="store_true")
    a = ap.parse_args()

    g = a.gender
    target = a.target_rate if a.target_rate else DEFAULT_RATE[g]
    lufs_t = a.lufs if a.lufs is not None else DEFAULT_LUFS[g]

    y, sr = load16(a.raw)
    dur = len(y) / sr
    rc = syl_rates(y, sr)

    if a.el and os.path.exists(a.el):
        ely, _ = load16(a.el)
        dur_el = len(ely) / sr
        ec = syl_rates(ely, sr)
        centers = [(j + 0.5) * WIN / dur_el for j in range(len(ec))]
        fs = []
        for i, r in enumerate(rc):
            frac = (i + 0.5) * WIN / dur
            er = float(np.interp(frac, centers, ec)) if ec else r
            fs.append(min(max(r / er if er > 0 else 1.0, 1.0), 1.30))
        mode = "EL逐窗对齐"
    elif a.target_cps and a.chars > 0:
        # 字/秒锚：先算全局该慢多少，再按各窗相对快慢分配，保留自然起伏
        our_cps = a.chars / dur
        f = our_cps / a.target_cps
        mean_r = sum(rc) / len(rc) if rc else 1.0
        fs = [min(max(f * (r / mean_r) if r > 0 else f, 1.0), 1.35) for r in rc]
        mode = (f"字/秒锚 {a.target_cps}（当前 {our_cps:.2f} 字/秒，"
                f"全局需慢 {f:.3f}×）")
    else:
        fs = [min(max(r / target if target > 0 else 1.0, 1.0), 1.30) for r in rc]
        mode = f"目标速率 {target}"

    fs = [(fs[i - 1] + 2 * fs[i] + fs[i + 1]) / 4 if 0 < i < len(fs) - 1 else fs[i]
          for i in range(len(fs))]
    print(f"模式: {mode}")
    print("raw速率:", [round(x, 2) for x in rc])
    print("窗口系数:", [round(x, 3) for x in fs], flush=True)

    graw, _ = detect_gaps_wav(a.raw)
    bounds = [0.0]
    for i in range(1, len(fs)):
        want = i * WIN
        cand = [gg for gg in graw if abs(gg[0] - want) <= 10.0 and gg[1] >= 0.08]
        b = min(cand, key=lambda gg: abs(gg[0] - want))[0] if cand else want
        bounds.append(max(b, bounds[-1] + 1.0))
    bounds.append(dur)

    tmpd = tempfile.mkdtemp()
    segs = []
    for i in range(len(fs)):
        s0, s1 = bounds[i], bounds[i + 1]
        t1 = os.path.join(tmpd, "in.wav")
        t2 = os.path.join(tmpd, "out.wav")
        sf.write(t1, y[int(s0 * sr):int(s1 * sr)], sr)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", t1, "-af",
                        "atempo=%.4f" % (1.0 / fs[i]), t2], check=True)
        s2, _ = sf.read(t2, dtype="float32")
        segs.append(s2)
    buf = np.concatenate(segs)
    print("分段拉伸后 %.1fs" % (len(buf) / sr), flush=True)
    t3 = os.path.join(tmpd, "st.wav")
    sf.write(t3, buf, sr)

    # ---- ② 变调：就地按实测 F0 算比例（只升音高、不改时长）----
    #    🔴 2026-10-14 实测：VoxCPM2 / Qwen 男声 F0 系统性偏低
    #      —— EL 男声 13 条案例 F0 中位 190.5~205.3（均值 ~198），
    #         我们只有 168~172，低约 15%。女声（OmniVoice 277 vs EL 271~282）没问题。
    #      🔴 必须在 atempo **之后**就地测：atempo 改了时间轴，
    #        「前 60s」覆盖的文本变了，拿 raw 的 F0 算会少补 3.4%。
    #      🔴 ffmpeg rubberband 的 pitch 是**频率比**，不是半音！
    #        （踩坑：填 2.85 以为是半音，实际升了 17 个半音，声音变花栗鼠）
    tf = a.target_f0 if a.target_f0 else DEFAULT_F0[g]
    if a.no_pitch:
        a.pitch = 1.0
    elif not a.pitch:
        yv, _ = load16(t3)
        f0r = f0_median(yv, sr)
        if f0r and f0r > 0:
            a.pitch = min(max(tf / f0r, 0.90), 1.30)
            print("F0 实测(拉伸后前60s) %.1fHz → 目标 %.1fHz，变调 ×%.4f（%.2f 半音）"
                  % (f0r, tf, a.pitch, 12 * np.log2(a.pitch)), flush=True)
        else:
            a.pitch = 1.0
    if a.pitch and abs(a.pitch - 1.0) > 1e-6:
        tp = os.path.join(tmpd, "pitch.wav")
        # 🔴 变调用 asetrate+aresample，**不用 rubberband**：
        #    rubberband 会把频谱平坦度压掉（0.050 → 0.035，而 EL 是 0.042），
        #    声音变"过净"、发假 —— 2026-10-14 员工在 F0 已对齐(198.3=EL)的情况下
        #    仍判「EL 更像真人」，怀疑就是这个副作用。
        #    asetrate 只是重采样（等价磁带变速 + 恢复时长），波形结构保得住：
        #      flat 0.049（EL 0.042）、发声段数 110（EL 110）、段长 0.291（EL 0.277）
        #    🔴 顺序必须是 asetrate → aresample → atempo(1/R)，**三步缺一不可**：
        #      asetrate=sr*R 改标记（音高×R，时长/R）→ aresample=sr 重采样（点数 N→N/R，
        #      时长仍 /R）→ atempo=1/R 放慢才把时长补回来。
        #      2026-10-14 踩坑：漏了 atempo，成品 440.9s 而应为 534.8s（短了 17.6%），
        #      字/秒直接飙到 10.2。
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", t3, "-af",
                        "asetrate=%d,aresample=%d,atempo=%.6f"
                        % (int(round(sr * a.pitch)), sr, 1.0 / a.pitch),
                        tp], check=True)
        t3 = tp
        # 🔴 必须把变调结果读回 buf：后面停顿清洗用的是 buf，不是 t3。
        #    2026-10-14 踩坑：漏了这一行，变调白做（日志显示变调成功，
        #    但成品 F0 仍是 171.7 没变），因为清洗直接拿未变调的 buf 覆盖了。
        buf, _ = sf.read(t3, dtype="float32")
        # 半音 = 12 * log2(频率比)，别写成 17.31
        print("变调 ×%.4f（%.2f 半音）" % (a.pitch, 12 * np.log2(a.pitch)),
              flush=True)

    if a.no_clean:
        cur = t3
    else:
        gs, _ = detect_gaps_wav(t3)
        buf, n_clean, floors = clean_gaps(buf, sr, gs, min_dur=0.08, w_ms=5)
        fl = 20 * np.log10(np.median(floors) + 1e-9) if floors else float("nan")
        print("停顿清洗 %d 处（合成tone，底噪中位 %.1f dBFS，长度不变）"
              % (n_clean, fl), flush=True)
        t4 = os.path.join(tmpd, "clean.wav")
        sf.write(t4, buf, sr)
        verify_pauses(t4, "清洗后")
        cur = t4

#    🔴 变调后高频需要额外回压（默认 -2.5dB，见 EQ_MALE_DB 注释）。
    eq_db = a.eq_db if a.eq_db is not None else (
        -4.0 + (-4.0 if a.pitch and a.pitch > 1.02 else 0.0))
    eq_m = EQ_MALE_TMPL % eq_db
    gain = round(lufs_t - lufs(cur), 1)
    # 🔴 二次校正：EQ 会改变响度，按 EQ 前的响度算出来的增益是错的
    #   （实测男声目标 -25.0 结果落到 -25.4）。所以先粗调、再实测补一次。
    af = (eq_m + "," if g == "m" else "") + "volume=%gdB" % gain
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", cur, "-af", af, a.out],
                   check=True)
    delta = round(lufs_t - lufs(a.out), 2)
    if abs(delta) > 0.05:
        t5 = os.path.join(tmpd, "loud2.wav")
        # os.replace 跨设备会炸（/tmp → /mnt），必须 shutil.move
        shutil.move(a.out, t5)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", t5, "-af",
                        "volume=%gdB" % delta, a.out], check=True)
        print("响度二次校正 %+.2fdB（EQ 改变了响度）" % delta, flush=True)
    yv, srv = load16(a.out)
    fc = syl_rates(yv, srv)
    gs2, _ = detect_gaps_wav(a.out)
    mid = [d for _, d in gs2 if 0.3 <= d < 0.8]
    print("修复后速率:", [round(x, 2) for x in fc], flush=True)
    print("DONE %s %.1fs 停顿%d处(0.3-0.8s:%d处/%.2fs) LUFS=%.1f(目标 %.1f)"
          % (a.out, len(yv) / srv, len(gs2), len(mid), sum(mid), lufs(a.out), lufs_t),
          flush=True)


if __name__ == "__main__":
    main()
