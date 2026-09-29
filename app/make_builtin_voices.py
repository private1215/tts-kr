# -*- coding: utf-8 -*-
"""制作内置音色模板：韩语克隆男声 / 韩语克隆女声。

用法（在服务器上，项目根目录）：
    ./venv/bin/python app/make_builtin_voices.py --src /tmp/builtin

素材目录约定：
    /tmp/builtin/male/1.mp3 2.mp3 nan.txt
    /tmp/builtin/female/아모텍1.mp3 아모텍2.mp3 아모텍tts.txt

为什么不能直接复用 voice_trim.pick_window
------------------------------------------
员工上传录音走的是 `voice_trim.pick_window`，打分 = 平均能量 × 稳定度，
**偏好最响亮的片段**。而官方 QwenLM/Qwen3-TTS#16 明确说：笑声来自参考音频
自带的情绪风格（Tokenizer 完全保留副语言信息）。主播最亢奋的喊单段被选中，
等于教模型「念着念着就笑」。现网「요한 쿡」正是这种片段——员工听到的笑声
与此吻合。

所以内置音色用**本脚本独立的选择器**：在候选窗口里挑「能量最接近全篇中位
+ 内部最稳定」的片段，即最典型、最不极端的一段。员工上传流程不受影响。

另外两条官方依据（详见 docs/内置音色与自动分块方案.md 2.5）：
  - 片段长度 15 秒：QwenLM/Qwen3-TTS#211 —— 参考音频 >20~30 秒时
    不吐 EOS 卡死明显更高发，官方缓解建议 ≤15 秒
  - ref_text 必须完整覆盖片段：mlx-audio#921 —— 用官方 PyTorch 权重做 A/B，
    截断文本 3/3 崩坏，完整转写 5/5 正常
"""
import os
import sys
import json
import argparse
import subprocess
import difflib

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr_check  # noqa: E402

import voice_trim as vt  # noqa: E402

BASE = os.environ.get("TTS_BASE", "").strip() or "/mnt/tts"
VOICE_DIR = os.path.join(BASE, "data", "voices")
VOICES_JSON = os.path.join(VOICE_DIR, "voices.json")

TARGET_SEC = 15.0       # 官方 #211 建议 ≤15 秒
MIN_CLIP_SEC = 8.0
MAX_CLIP_SEC = 20.0
# 韩语正常语速。用来校验「文字量」与「音频时长」是否匹配
KO_CPS = 6.15
# 预计时长 / 实际时长 的允许区间（mlx-audio#921 建议加的合理性校验）
RATE_LO, RATE_HI = 0.7, 1.4

SPECS = [
    {"id": "builtin_male", "dir": "male",
     "audio": ["1.mp3", "2.mp3"], "text": "nan.txt"},
    {"id": "builtin_female", "dir": "female",
     "audio": ["아모텍1.mp3", "아모텍2.mp3"], "text": "아모텍tts.txt"},
]


def to_wav24k(src: str, dst: str):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", dst],
        check=True)
    return dst


def load_concat(paths, workdir):
    """把多个音频转码后拼成一条，返回 (x, sr)。"""
    xs = []
    for i, p in enumerate(paths):
        w = os.path.join(workdir, f"_part{i}.wav")
        to_wav24k(p, w)
        a, sr = sf.read(w, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        xs.append(a)
        print(f"    {os.path.basename(p)}: {len(a) / sr:.2f}s")
    return np.concatenate(xs), sr


def pick_typical_window(x, sr, target=TARGET_SEC):
    """挑「最典型」的一段：能量接近全篇中位 + 内部稳定 + 无长停顿。

    与 voice_trim.pick_window 的区别：那边用 `平均能量 × 稳定度`，会挑中
    最响亮的喊单段；这里把「能量项」换成「与中位能量的接近程度」，
    挑的是主播最常态的那 15 秒。
    """
    dur = len(x) / float(sr)
    # 注意：short_time_rms 返回的 hop 是「采样点数」不是秒，时间换算一律用
    # vt.HOP_SEC。voice_trim 内部也是这么做的（它从不使用返回的 hop 做时间换算）。
    rms, _hop_samples = vt.short_time_rms(x, sr)
    hop = vt.HOP_SEC
    n = len(rms)
    target_f = int(target / hop)
    med = float(np.median(rms[rms > 0])) if np.any(rms > 0) else 0.0

    cum = np.concatenate([[0.0], np.cumsum(rms)])
    cum2 = np.concatenate([[0.0], np.cumsum(rms ** 2)])
    mask = vt._speech_mask(rms)

    def score(sf_, ef):
        m = (cum[ef] - cum[sf_]) / (ef - sf_)
        v = max(0.0, (cum2[ef] - cum2[sf_]) / (ef - sf_) - m * m)
        cv = np.sqrt(v) / m if m > 0 else 9.9
        stab = 1.0 / (1.0 + min(cv, 3.0))
        # 与全篇中位能量的接近程度（越接近 1 越好）
        typical = 1.0 / (1.0 + abs(m - med) / med) if med > 0 else 0.5
        # 内部最长静音
        worst, run = 0, 0
        for b in mask[sf_:ef]:
            if not b:
                run += 1
                worst = max(worst, run)
            else:
                run = 0
        pen = 1.0 / (1.0 + max(0.0, worst * hop - vt.LONG_PAUSE_SEC) * 2.0)
        return typical * stab * pen, m, cv

    best = None
    step = max(1, int(0.25 / hop))
    for s0 in range(0, max(1, n - target_f), step):
        e0 = min(n, s0 + target_f)
        if (e0 - s0) * hop < MIN_CLIP_SEC:
            continue
        sc, m, cv = score(s0, e0)
        if best is None or sc > best[0]:
            best = (sc, s0, e0, m, cv)
    if best is None:
        raise RuntimeError(
            f"挑不出可用窗口：n={n} target_f={target_f} dur={dur:.1f}s "
            f"(素材太短或几乎无声)")
    _, s0, e0, m, cv = best
    half = int(vt.SNAP_SEC / hop)
    s1 = vt._snap(s0, rms, half)
    e1 = vt._snap(e0, rms, half)
    if (e1 - s1) * hop < MIN_CLIP_SEC:
        s1, e1 = s0, e0
    # 上限保护：吸附后可能变长，超过 MAX_CLIP_SEC 就截回（官方 #211：>20~30s
    # 不吐 EOS 明显更高发）
    a = s1 * hop
    b = min(dur, e1 * hop)
    if b - a > MAX_CLIP_SEC:
        b = a + MAX_CLIP_SEC
    return a, b, {
        "mean_rms": round(float(m), 4), "cv": round(float(cv), 3),
        "median_rms": round(med, 4), "method": "typical"}


def align_to_script(asr_text: str, script: str) -> str:
    """把 ASR 粗略文本吸附到原始文字稿，取出原文中完全一致的片段。

    Whisper 转写会有错字，但**首尾几个字通常是对的**。用首 30 字定位起点、
    尾 30 字定位终点，中间整段取原文——这样 ref_text 与音频逐字一致。

    找不到就返回空串，调用方会拒绝写入（绝不能拿错字稿做克隆）。
    """
    a = asr_check.normalize(asr_text)
    s = asr_check.normalize(script)
    if len(a) < 20 or not s:
        return ""
    head, tail = a[:30], a[-30:]
    i0 = _locate(s, head)
    i1 = _locate(s, tail, from_end=True)
    if i0 < 0 or i1 < 0 or i1 <= i0:
        return ""
    # 归一化串的下标换算回原文下标：用一个等长的映射表
    idxs = [i for i, c in enumerate(script) if asr_check._KEEP.match(c)]
    if len(idxs) != len(s):
        return ""
    j0, j1 = idxs[i0], idxs[min(i1 + 29, len(idxs) - 1)]
    # 刻意**不**向句边界外扩：音频起点往往落在词中间（如 "...목, 하나씩"），
    # 一旦外扩成完整句子，ref_text 就会包含音频里根本没说的字，
    # 这正是 mlx-audio#921 记录的「文本不覆盖音频 → 3/3 崩坏」。
    # 宁可首尾是残句，也要保证文字与音频逐字一致。
    return script[j0:j1 + 1].strip()


def _locate(hay: str, needle: str, from_end: bool = False) -> int:
    """模糊定位：在 hay 里找与 needle 最像的位置。返回下标，找不到 -1。"""
    if not needle:
        return -1
    L = len(needle)
    best_i, best_r = -1, 0.0
    rng = range(len(hay) - L + 1)
    for i in rng:
        r = difflib.SequenceMatcher(
            None, hay[i:i + L], needle, autojunk=False).ratio()
        if r > best_r:
            best_r, best_i = r, i
        if best_r > 0.95:
            break
    return best_i if best_r >= 0.6 else -1


def build_one(spec, src_root, workdir, force=False):
    vid = spec["id"]
    d = os.path.join(src_root, spec["dir"])
    print(f"\n=== {vid} ===")
    paths = [os.path.join(d, a) for a in spec["audio"]]
    for p in paths:
        if not os.path.exists(p):
            print(f"  跳过：缺少 {p}")
            return None
    txt_path = os.path.join(d, spec["text"])
    script = open(txt_path, encoding="utf-8").read().strip()
    print(f"  文字稿 {len(script)} 字")

    x, sr = load_concat(paths, workdir)
    print(f"  拼接后总时长 {len(x) / sr:.2f}s")

    a, b, info = pick_typical_window(x, sr)
    seg = x[int(a * sr):int(b * sr)].copy()
    clip_dur = len(seg) / sr
    print(f"  选中片段 {a:.2f}s ~ {b:.2f}s ({clip_dur:.2f}s) {info}")

    tmp_wav = os.path.join(workdir, f"{vid}_clip.wav")
    sf.write(tmp_wav, seg, sr)

    # ref_text：ASR 转写 + 原文逐字对齐
    hyp = asr_check.transcribe(tmp_wav, language="ko")
    print(f"  ASR 转写 {len(hyp)} 字")
    ref_text = align_to_script(hyp, script)
    if not ref_text:
        print("  ✗ 无法把转写结果对齐到原文，拒绝写入（宁可不做，也不做错）")
        return None
    print(f"  ref_text {len(ref_text)} 字")
    print(f"    开头: {ref_text[:46]}")
    print(f"    结尾: {ref_text[-30:]}")

    # 合理性校验（mlx-audio#921 建议）
    # 基准语速必须用**这份素材自己实测的语速**，不能用通用 6.15：
    # 男声素材实测 4045 字 / 483.76s = 8.36 字每秒，用 6.15 会算出 1.33
    # 误判成「文字比音频多 33%」，其实覆盖是准的。
    cps = len(script) / (len(x) / sr) if len(x) else KO_CPS
    rate = (len(ref_text) / cps) / clip_dur if clip_dur else 99
    print(f"  时长校验: 素材实测语速 {cps:.2f} 字/秒 → "
          f"预计 {len(ref_text) / cps:.2f}s / 实际 {clip_dur:.2f}s "
          f"= {rate:.3f} (允许 {RATE_LO}~{RATE_HI})")
    if not (RATE_LO <= rate <= RATE_HI):
        print("  ✗ 文字量与音频时长对不上，拒绝写入")
        return None

    # 音频先落在工作目录：VOICE_DIR 属 root（zheng 无写权限），
    # 统一由最后的 sudo 步骤搬过去，避免脚本半途因权限失败留下残件。
    out_wav = os.path.join(VOICE_DIR, f"{vid}.wav")
    local_wav = os.path.join(workdir, f"{vid}.wav")
    sf.write(local_wav, seg, sr)
    return {
        "_local_wav": local_wav,
        "path": out_wav,
        "ref_text": ref_text,
        "sr": sr,
        "dur": round(clip_dur, 2),
        "builtin": True,
        "src_dur": round(len(x) / sr, 2),
        "src_text_len": len(script),
        "src_ref_text": script,
        "clip": {"start": round(a, 2), "end": round(b, 2)},
        "clip_info": info,
        "asr_hyp": hyp,
        "rate": round(rate, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/builtin")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    workdir = os.path.join(args.src, "_work")
    os.makedirs(workdir, exist_ok=True)

    data = {}
    if os.path.exists(VOICES_JSON):
        data = json.load(open(VOICES_JSON, encoding="utf-8"))
    if isinstance(data, list):
        data = {}

    made = {}
    for spec in SPECS:
        if spec["id"] in data and not args.force:
            print(f"\n{spec['id']} 已存在，跳过（--force 可覆盖）")
            continue
        got = build_one(spec, args.src, workdir, args.force)
        if got:
            made[spec["id"]] = got

    if not made:
        print("\n没有产出任何内置音色")
        return
    data.update(made)
    # VOICE_DIR 属 root：这里只写「待安装清单」，真正的搬运与写 voices.json
    # 由 install_builtin_voices.py（sudo 运行）完成。
    plan_path = os.path.join(args.src, "install_plan.json")
    json.dump(data, open(plan_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n✓ 产出待安装清单 {plan_path}")
    for k, v in made.items():
        print(f"   {k}: {v['dur']}s, ref_text {len(v['ref_text'])} 字, "
              f"rate {v['rate']}")
        print(f"      音频: {v['_local_wav']}")


if __name__ == "__main__":
    main()
