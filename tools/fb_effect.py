# -*- coding: utf-8 -*-
"""反馈题自检：员工的选项**客观上**到底差多少？

员工反馈「听起来都一样」，有两种可能，必须分开：
  A. 真的没差别（出题失败）——四个选项的客观指标差异低于人耳察觉阈
  B. 有差别但没听出来（材料/引导失败）——差异够大，但员工不知道该听什么

这份脚本先把 A 排除掉：对 items.json 里每一题的每个选项测一组客观指标，
算出**跨选项极差**，再和已知的听觉阈值比。低于阈值的题，员工说"一样"是诚实的，
要换更强的刺激；达到阈值的题，则是引导问题（要在题目里点明听哪里）。

顺带查自由文本里提的两条硬伤：
  「有电子背景音」-> 静音段底噪 / 高频嘶声，和案例原声直接对比
  「播放会卡顿」  -> 逐样本不连续（爆音/断点）计数

听觉阈值（本项目已在 tools/prosody_target.py 注释里定过口径）：
  句尾升降   1.0 半音    —— 低于此=放大噪声
  句中位音高 0.5 半音    —— 语音 F0 的 JND 量级
  停顿时长   max(0.03s, 8%)
  停顿密度   1.0 次/分
  呼吸频率   0.5 次/分
"""
import os
import sys
import json
import subprocess

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")

import numpy as np  # noqa: E402
import librosa  # noqa: E402
import soundfile as sf  # noqa: E402
from prosody_target import measure_wav, sil_gaps  # noqa: E402

FB = os.environ.get("TTS_FB_DIR", "/mnt/tts/data/tone-ab")
SR = 24000

# 各维度关心的指标 + 阈值
DIM_KEYS = {
    "语调": [("tone_amp_p50", 1.0, "句尾降幅(半音)"), ("pitch_med", 5.0, "句中位音高(Hz)")],
    "断句": [("pause_dur_p50", None, "停顿时长中位数(s)"), ("pause_n_per_min", 1.0, "停顿密度(次/分)")],
    "停顿": [("pause_dur_p50", None, "停顿时长中位数(s)"), ("pause_n_per_min", 1.0, "停顿密度(次/分)")],
    "音色": [],
    "长文本": [],
}


def load24(path):
    x, sr = librosa.load(path, sr=SR, mono=True)
    return x, sr


def semitone(a, b):
    """两个频率差多少半音（f0 用半音比线性 Hz 更符合听觉）。"""
    if not a or not b or a <= 0 or b <= 0:
        return 0.0
    return abs(12 * np.log2(a / b))


def noise_floor(path):
    """静音段底噪。员工说的「电子背景音」就在这些段里。

    返回：静音段 RMS(dBFS)、静音段高频占比、有声段高频占比
    （高频占比用 10k 以上能量 / 总能量，嘶声会让它明显偏高）
    """
    x, sr = load24(path)
    gaps, db, hop = sil_gaps(x, sr)
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=512)) + 1e-12
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hi = freqs >= 10000
    e_all = S ** 2
    hi_ratio_all = float(e_all[hi].sum() / e_all.sum())

    if not gaps:
        return -120.0, 0.0, hi_ratio_all
    vals, hrs = [], []
    for a, b in gaps:
        seg = x[int(a):int(b)]
        if len(seg) < 512:
            continue
        vals.append(float(np.sqrt(np.mean(seg ** 2) + 1e-12)))
        Sg = np.abs(librosa.stft(seg, n_fft=2048, hop_length=512)) + 1e-12
        eg = Sg ** 2
        hrs.append(float(eg[hi].sum() / eg.sum()))
    if not vals:
        return -120.0, 0.0, hi_ratio_all
    floor = 20 * np.log10(float(np.mean(vals)) + 1e-12)
    return floor, float(np.mean(hrs)), hi_ratio_all


def click_rate(path, k=4.0, win_ms=20.0):
    """逐样本断点（爆音/拼接咔哒）计数，个/秒。

    🔴 第一版写错了：用 cumsum 做滑动 RMS 时窗口和对齐没对上，同一个人同一条源音频
       派生出的文件竟然报 0.6/s 和 22/s 两种结果 —— 那是**测量 bug 不是音频问题**
       （用绝对阈值复核后各文件完全一致）。现在改成**以断点为中心**的窗口：
         c[i] = sum(x[:i]^2)
         rms[t] = sqrt((c[hi_t] - c[lo_t]) / len)
       再拿 |x[t+1]-x[t]| 和 rms[t] 比，索引严格对齐。

    k=4 而不是 8：真实爆音的样本间跳变约 0.1~0.5，而语音局部 RMS 常在 0.07 上下，
    k=8 会把真实爆音一并放过。另设 0.05 的绝对下限，避免极安静处的数值噪声被计入。
    """
    x, sr = load24(path)
    if len(x) < 4:
        return 0.0
    d = np.abs(np.diff(x))
    half = max(int(sr * win_ms / 2000.0), 1)
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    t = np.arange(len(x))
    lo = np.clip(t - half, 0, len(x))
    hi = np.clip(t + half + 1, 0, len(x))
    rms = np.sqrt(np.maximum(c[hi] - c[lo], 0.0) / np.maximum(hi - lo, 1))
    thr = np.maximum(k * rms[:-1], 0.05)
    return float(np.sum(d > thr) / (len(x) / sr))


def bitrate(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=bit_rate,sample_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def main():
    items = json.load(open(os.path.join(FB, "items.json"), encoding="utf-8"))["items"]
    print("=" * 92)
    print("反馈题自检：员工的选项客观上差多少（低于听觉阈值 = 员工说「都一样」是诚实的）")
    print("题库目录:", FB)
    print("=" * 92)

    for it in items:
        opts = it.get("options", [])
        files = [(o.get("key"), o.get("file"), o.get("label", "")) for o in opts if o.get("file")]
        if it.get("ref", {}).get("file"):
            r = it["ref"]
            files = [("参照", r["file"], r.get("label", ""))] + files
        print()
        print("── %s  ［%s］" % (it.get("title", it["id"]), it.get("dim", "-")))
        print("   id=%s   选项数=%d（其中无音频 %d 个）"
              % (it["id"], len(opts), len(opts) - len([o for o in opts if o.get("file")])))

        res = {}
        for key, f, label in files:
            p = os.path.join(FB, f)
            if not os.path.exists(p):
                print("   ⚠️ 缺文件", f)
                continue
            try:
                res[key] = measure_wav(p)
            except Exception as e:
                print("   ⚠️ 测量失败 %s: %s" % (f, str(e)[:80]))

        keys = list(res.keys())
        if len(keys) < 2:
            continue

        # 判断这一题到底该盯哪个指标
        dim = it.get("dim", "")
        checks = []
        for tag, lst in DIM_KEYS.items():
            if tag in dim:
                checks = lst
                break
        if not checks:
            checks = [("tone_amp_p50", 1.0, "句尾降幅(半音)"), ("pause_dur_p50", None, "停顿时长(s)"),
                      ("pause_n_per_min", 1.0, "停顿密度"), ("pitch_med", 5.0, "句中位Hz")]

        print("   %-16s %s" % ("选项", " ".join("%-14s" % k for k in keys)))
        verdict = []
        for key, thr, name in checks:
            vals = [res[k].get(key) for k in keys]
            if any(v is None for v in vals):
                continue
            vals = [float(v) for v in vals]
            rng = max(vals) - min(vals)
            if key == "pitch_med":
                rng = semitone(max(vals), min(vals))
                thr = 0.5
            if key == "pause_dur_p50":
                thr = max(0.03, 0.08 * float(np.mean(vals)))
            row = "   %-16s %s   极差=%.3f  阈值≈%.3f  %s" % (
                name, " ".join("%-14s" % ("%.2f" % v) for v in vals),
                rng, thr, "✅可辨" if rng >= thr else "❌低于察觉阈")
            print(row)
            verdict.append(rng >= thr)

        if not verdict:
            print("   （维度「%s」没有预设指标，按音色/整体处理，见下方底噪与爆音检查）" % dim)
        elif not any(verdict):
            print("   🔴 结论：全部指标的极差都低于听觉阈值 —— 这题员工说「一样」是正常的，")
            print("      必须换更强的刺激（拉开参数间距）或改问法。")
        else:
            print("   ✅ 结论：至少有一项超过察觉阈。员工听不出的话是「不知道听哪里」，")
            print("      应在题目里点明具体位置和听感线索，而不是继续拉大差距。")

        # 附带：每个选项的底噪与断点
        print("   %-16s %s" % ("底噪/细节", " ".join("%-14s" % k for k in keys)))
        nf, hrs, cr, br = [], [], [], []
        for k, f, _ in files:
            if k not in res:
                continue
            p = os.path.join(FB, f)
            try:
                floor, hr, _ = noise_floor(p)
                rate = click_rate(p)
            except Exception as e:
                print("   ⚠️ 附加检查失败 %s: %s" % (f, str(e)[:60]))
                continue
            nf.append(floor)
            hrs.append(hr)
            cr.append(rate)
            br.append(res[k].get("breath_n_per_min"))
        if nf:
            print("   %-16s %s   ⚠️差 %.1f dB"
                  % ("静音底噪dBFS", " ".join("%-14s" % ("%.1f" % v) for v in nf),
                     max(nf) - min(nf)))
            print("   %-16s %s"
                  % ("静音高频占比", " ".join("%-14s" % ("%.4f" % v) for v in hrs)))
            print("   %-16s %s"
                  % ("呼吸(次/分)", " ".join("%-14s" % ("%.1f" % v) for v in br)))
            print("   %-16s %s"
                  % ("爆音(个/秒)", " ".join("%-14s" % ("%.2f" % v) for v in cr)))


if __name__ == "__main__":
    main()
