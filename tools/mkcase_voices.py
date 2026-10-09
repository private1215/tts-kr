# -*- coding: utf-8 -*-
"""从案例库批量生成克隆音色，直接放进「预设音色」，员工不必自己录音。

为什么要有这个
--------------
案例库上传者 prompted：既然案例里已经有真人原声，为什么还要员工自己上传录音？
答案是**不需要**。案例库每条都是几分钟的高质量原声 + 逐字稿，正好是现成的
克隆素材；缺的只是「挑最好听的 15 秒 + 把文字稿对齐截出来」这一步 —— 而
`app/voice_trim.py` 本来就是干这个的（员工自己录音色时也走它）。

🔴 关键前提：案例库 35 条听着多，实际只有 **4 个人**在发声（2026-10-09 盘点）：
    案例-男声 1 条 / 案例-女声 1 条 / yohan 12 条 / SOLA 21 条（20 女 + 1 男）
    所以「一条案例一个音色」会得到 35 个几乎一样的选项，员工照样不会选。
    本脚本按 **(说话人, 性别)** 聚类，一组只出一个音色 —— 这就是全部的差异。

产出
----
每个音色写进 voices.json（含 `gender` 与 `bench_case`），音频落 VOICE_DIR：
- `bench_case` 让生产目标值直接对标**这个人自己的那条案例**。
  实测（r26g/r26h）对标 vs 不对标，五例均值 **96.07 vs 93.59（+2.48）**，
  不给员工增加任何操作。
- `gender` 之前 voices.json 里是缺失的，自动按性别匹配案例必须要有它。

用法
----
    venv/bin/python tools/mkcase_voices.py              # 先 dry-run 看计划
    venv/bin/python tools/mkcase_voices.py --apply      # 真正写入
    venv/bin/python tools/mkcase_voices.py --apply --only yohan,SOLA

⚠️ 与 `install_builtin_voices.py` 的分工：那一个是把 make_builtin_voices.py
产出的音频装进 voices.json（要 root）；本脚本直接从 cases 读，自己跑裁剪，
写完就能用。
"""
import os
import sys
import re
import json
import argparse
import logging

import numpy as np
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 🔴 app 目录也要进 path：`voice_trim` 跟 server.py 住在一起，不在项目根。
#    server.py 是因为 systemd 的 WorkingDirectory=app 才能直接 import 的。
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "app"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("mkcase")

BASE = os.environ.get("TTS_BASE", "").strip() or "/mnt/tts"
CASES_DIR = os.path.join(BASE, "data", "cases")
CASES_JSON = os.path.join(BASE, "data", "cases.json")
VOICE_DIR = os.path.join(BASE, "data", "voices")
VOICES_JSON = os.path.join(VOICE_DIR, "voices.json")

TARGET_SEC = 15.0        # 15 秒是克隆参考的甜点（太长显存暴涨，太短音色不像）
TRIM_ABOVE = 30.0        # 低于这个长度不裁剪
TARGET_SR = 24000

# 已有音色 → 说话人名。这些人**不重建**，脚本只给他们补 gender / bench_case。
KNOWN = {
    "EL_male":   "1a6c20a60481",
    "EL_female": "00b6a45ad6aa",
}
KNOWN_SPEAKER = {"案例-男声": "EL_male", "案例-女声": "EL_female"}

_G = {"male": {"zh": "男声", "en": "male", "ko": "남성", "ja": "男性"},
      "female": {"zh": "女声", "en": "female", "ko": "여성", "ja": "女性"}}


def make_labels(name, gender):
    """给音色一个「人话」的显示名。voices.json 的 key 用 ASCII（避免接口/路径
    编码问题），展示名另存一份多语 label，前端按界面语言挑。"""
    g = _G.get(gender) or _G["male"]
    return {"zh": f"{name} {g['zh']}", "en": f"{name} ({g['en']})",
            "ko": f"{name} {g['ko']}", "ja": f"{name} {g['ja']}"}


def load_json(p, default):
    if not os.path.exists(p):
        return default
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception as e:
        log.warning("读 %s 失败：%s", p, e)
        return default


def case_text(cid):
    """取案例全文：优先 full.txt，否则按 segN.txt 顺序拼。"""
    d = os.path.join(CASES_DIR, cid)
    full = os.path.join(d, "full.txt")
    if os.path.exists(full):
        return open(full, encoding="utf-8").read()
    segs = sorted([p for p in os.listdir(d) if re.match(r"seg\d+\.txt$", p)],
                  key=lambda p: int(re.findall(r"seg(\d+)", p)[0]))
    return "\n".join(open(os.path.join(d, p), encoding="utf-8").read().strip()
                     for p in segs)


def case_audio(cid):
    """案例合并后的整段音频。没有 merged.wav 才退回拼 segN.wav。"""
    d = os.path.join(CASES_DIR, cid)
    m = os.path.join(d, "merged.wav")
    if os.path.exists(m):
        return m
    segs = sorted([p for p in os.listdir(d) if re.match(r"seg\d+\.wav$", p)],
                  key=lambda p: int(re.findall(r"seg(\d+)", p)[0]))
    return [os.path.join(d, p) for p in segs] or None


def read_mono(paths):
    """读一段或多段音频，重采样到 TARGET_SR 后拼接。"""
    import librosa
    if isinstance(paths, str):
        paths = [paths]
    out = []
    for p in paths:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != TARGET_SR:
            x = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR)
        out.append(x)
    return np.concatenate(out) if len(out) > 1 else out[0]


_CAMPPLUS = os.environ.get(
    "TTS_CAMPPLUS", "/mnt/tts/models/Fun-CosyVoice3-0.5B/campplus.onnx")
_sess = None


def _emb_session():
    global _sess
    if _sess is None:
        import onnxruntime
        _sess = onnxruntime.InferenceSession(
            _CAMPPLUS, providers=["CPUExecutionProvider"])
    return _sess


def _embed(x16k):
    """campplus 说话人嵌入。⚠️ 吃的是 kaldi fbank(80mel) @16k，不是波形。"""
    import torch
    import torchaudio.compliance.kaldi as kaldi
    sess = _emb_session()
    f = kaldi.fbank(torch.tensor(x16k).unsqueeze(0), num_mel_bins=80,
                    dither=0, sample_frequency=16000)
    f = f - f.mean(dim=0, keepdim=True)
    e = sess.run(None, {sess.get_inputs()[0].name: f.unsqueeze(0).numpy()})[0]
    e = np.asarray(e).flatten()
    n = np.linalg.norm(e)
    return e / n if n else e


def pick_source_centroid(cids, target=TARGET_SEC):
    """🔴 **这才是「从案例里提取这个人的音色」的正确做法**。

    背景：案例库里一个人往往有十几条、合计几千秒的素材，而克隆参考只要
    15 秒。朴素做法是「挑一段没长停顿的连续 15 秒」，但那只是**最容易切**
    的一段，跟**像不像这个人**毫无关系 —— 可能正好挑到他状态最差、离麦
    最近、或者一句含混的过渡句。

    正确做法：把同人**所有**素材切成大量重叠窗口，各提一个说话人嵌入，
    取均值得到这个人的「音色质心」，再挑**余弦最接近质心**的窗口。
    挑出来的就是最能代表他、最没有个人状态漂移的那一段。

    ⚠️ campplus 官方窗口上限 10 秒，所以用 10 秒窗提嵌入、5 秒步长滑，
    命中后再以该窗为中心扩到 target 秒（克隆参考要 15 秒）。
    """
    import librosa
    wins = []
    for cid in cids:
        try:
            x = read_mono(case_audio(cid))
            x16 = librosa.resample(x, orig_sr=TARGET_SR, target_sr=16000)
        except Exception as e:
            log.warning("    %s 读取失败：%s", cid[:8], repr(e)[:60])
            continue
        W = min(10 * 16000, len(x16))
        step = 5 * 16000
        if len(x16) < 16000:
            continue
        for s in range(0, max(1, len(x16) - W), step):
            wins.append((cid, s / 16000.0, _embed(x16[s:s + W])))
        del x, x16
    if not wins:
        return None, []
    M = np.mean([w[2] for w in wins], axis=0)
    M = M / np.linalg.norm(M)
    scored = sorted(((float(np.dot(w[2], M)), w[0], w[1]) for w in wins),
                    reverse=True)
    top = scored[:5]
    best = scored[0]
    # 以命中的 10 秒窗为中心，扩成 target 秒的克隆参考
    center = best[2] + 5.0
    return (best[1], max(0.0, center - target / 2.0)), top


def _cut_at(x, txt, start, target):
    """从质心命中的那个位置裁一段，并把文字稿按**有声时长**对齐截出来。

    与 trim_reference 的区别只有一个：起点不是「最好切的窗口」，而是
    「最像这个人」的位置（由 pick_source_centroid 算出）。
    """
    from voice_trim import (short_time_rms, _speech_mask, slice_transcript,
                            HOP_SEC, FADE_SEC)
    sr = TARGET_SR
    a = max(0.0, float(start))
    b = min(len(x) / float(sr), a + target)
    i0, i1 = int(a * sr), int(b * sr)
    seg = x[i0:i1].copy()
    fade = max(1, int(sr * FADE_SEC))
    if seg.size > 2 * fade:
        seg[:fade] *= np.linspace(0.0, 1.0, fade).astype(np.float32)
        seg[-fade:] *= np.linspace(1.0, 0.0, fade).astype(np.float32)
    rms, hop = short_time_rms(x, sr)
    mask = _speech_mask(rms)
    sub, tinfo = slice_transcript(txt, mask, HOP_SEC, a, b)
    info = {"method": "centroid", "clip_start": round(a, 2),
            "clip_end": round(b, 2), "clip_dur": round(b - a, 2)}
    info.update(tinfo)
    return seg, sr, sub, info


def groups(cases):
    """按「说话人 + 性别」聚类。同一个人往往上传了十几条案例。"""
    g = {}
    for cid, c in cases.items():
        key = ((c.get("name") or "").strip(), (c.get("gender") or "").strip())
        if not key[0]:
            continue
        g.setdefault(key, []).append(cid)
    return g


def pick_source(cases, cids):
    """从一组同人案例里挑**最佳**克隆源。

    🔴 不是挑「音频最长的那条」：长的那条未必能截出好的 15 秒。
    实测 yohan 组里最长的一条（555s）只能截到 **9.00s**，因为该音频里
    长停顿密，pick_window 找不到更长的无长停顿连续段。
    15 秒是克隆参考的甜点，短参考会让音色相似度下降（yo han 那条 8.66s
    就是例子）。所以这里对**每一条**都试跑一次窗口挑选，取能截出最长
    片段的那条 —— 一组里往往差 3~6 秒。

    返回 (cid, 试跑明细列表)。
    """
    from voice_trim import pick_window
    tried, best = [], None
    for cid in cids:
        try:
            paths = case_audio(cid)
            x = read_mono(paths)
            a, b, winfo = pick_window(x, sr=TARGET_SR, target=TARGET_SEC)
            dur = float(b - a)
        except Exception as e:
            tried.append((cid, 0.0, repr(e)[:60]))
            continue
        tried.append((cid, round(dur, 2), ""))
        if best is None or dur > best[1]:
            best = (cid, dur)
        del x
    return (best[0] if best else None), tried


def safe_name(s):
    """说话人名 → 音色 id。韩文/英文直接保留，其它符号清掉。"""
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", (s or "").strip())
    return s.strip("_") or "voice"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正写文件（默认 dry-run）")
    ap.add_argument("--only", default="", help="只处理这些说话人，逗号分隔")
    ap.add_argument("--target", type=float, default=TARGET_SEC)
    ap.add_argument("--min-cases", type=int, default=3,
                    help="同人样本少于这么多就不做成预设音色")
    ap.add_argument("--pick", default="centroid", choices=("centroid", "window"),
                    help="挑源方式：centroid=按说话人嵌入质心挑「最像这个人」的一段"
                         "（默认，会用上同人全部素材）；window=只挑最好切的连续段")
    args = ap.parse_args()

    from voice_trim import trim_reference        # 延迟导入：librosa 较慢
    cases = load_json(CASES_JSON, {})
    voices = load_json(VOICES_JSON, {})
    if not cases:
        log.error("读不到案例：%s", CASES_JSON)
        return 1

    grp = groups(cases)
    log.info("案例 %d 条 -> 按「说话人+性别」聚成 %d 组：", len(cases), len(grp))
    for (nm, gender), cids in sorted(grp.items(), key=lambda kv: -len(kv[1])):
        log.info("  %-14s %-7s %2d 条", nm, gender or "-", len(cids))

    plan = []
    skipped = []
    for (nm, gender), cids in grp.items():
        if args.only and nm not in args.only.split(","):
            continue
        # 已经有现成音色的不重建（重建只是白花时间，音频来源完全相同）
        if nm in KNOWN_SPEAKER:
            skipped.append((nm, gender, f"已有音色 {KNOWN_SPEAKER[nm]}"))
            continue
        # 🔴 一条案例只是一个「样本」，不是一个「声源」。
        #   要求至少几条同人样本，才说明这个声音稳定复现、值得做成预设音色。
        #   顺带滤掉 SOLA_male 这种只有 1 条、性别疑似标错的组。
        if len(cids) < args.min_cases:
            skipped.append((nm, gender, f"仅 {len(cids)} 条样本，不足 {args.min_cases}"))
            continue
        start = None
        tried = []
        if args.pick == "centroid":
            r, top = pick_source_centroid(cids, args.target)
            if r:
                src, start = r
                log.info("  %-16s 质心 Top5（余弦越接近 1 越像这个人）：",
                         re.sub(r"[^\w]+", "", nm))
                for cos, c, s in top:
                    log.info("      %.4f  %s  @%.1fs", cos, c[:8], s)
            else:
                log.warning("  %s 质心提取失败，回退 window 模式", nm)
                src, _ = pick_source(cases, cids)
        else:
            src, tried = pick_source(cases, cids)
        if not src:
            continue
        best_dur = max((t[1] for t in tried), default=0.0)
        gender = gender or cases[src].get("gender") or "male"
        key = re.sub(r"[^\w]+", "", nm) + "_" + gender
        log.info("  %-16s 组内 %d 条逐条试过，最佳片段 %.2fs（来自 %s）",
                 key, len(cids), best_dur, src[:8])
        plan.append({"name": key, "label": make_labels(nm, gender),
                     "speaker": nm, "gender": gender, "cid": src,
                     "start": start, "src_dur": round(best_dur, 1),
                     "n_case": len(cids)})

    if skipped:
        log.info("\n跳过：")
        for nm, gender, why in skipped:
            log.info("  %-14s %-7s %s", nm, gender or "-", why)

    log.info("\n计划：" if not args.apply else "\n执行：")
    for p in plan:
        log.info("  %-16s <- %s (%s, %d 条, 源音频 %.0fs)",
                 p["name"], p["cid"], p["gender"], p["n_case"], p["src_dur"])
    if not args.apply:
        log.info("\n（dry-run）确认无误后加 --apply 真正写入")
        return 0

    for p in plan:
        cid = p["cid"]
        paths = case_audio(cid)
        if not paths:
            log.warning("  %s 没有可用音频，跳过", p["name"])
            continue
        try:
            x = read_mono(paths)
            txt = case_text(cid)
            if not (txt or "").strip():
                log.warning("  %s 缺文字稿，跳过（克隆必须有 ref_text）", p["name"])
                continue
            if args.pick == "centroid" and p.get("start") is not None:
                seg, sr, ref_text, info = _cut_at(x, txt, p["start"],
                                                  args.target)
            else:
                seg, sr, ref_text, info = trim_reference(
                    x, TARGET_SR, txt, target=args.target,
                    trim_above=TRIM_ABOVE)
        except Exception as e:
            log.warning("  %s 裁剪失败：%s", p["name"], repr(e)[:160])
            continue

        out_path = os.path.join(VOICE_DIR, p["name"] + ".wav")
        sf.write(out_path, seg, sr)
        voices[p["name"]] = {
            "path": out_path,
            "label": p["label"],
            "ref_text": ref_text,
            "sr": int(sr),
            "dur": round(len(seg) / float(sr), 2),
            "engine": "qwen",          # 🔴 别留成 cosyvoice3：那是 10-08 的事故
            "gender": p["gender"],
            # 对标这个人自己的案例 —— 员工什么都不用选就能吃到 +2.48
            "bench_case": cid,
            "origin": "cases",
            "src_case": cid,
            "src_speaker": p["speaker"],
            "src_dur": p["src_dur"],
            "clip": [info.get("clip_start"), info.get("clip_end")],
        }
        log.info("  ✅ %-16s %5.2fs  文字稿 %d 字  clip=%s",
                 p["name"], len(seg) / float(sr), len(ref_text), info.get("clip_dur"))

    # 已有音色补 gender / bench_case（这两个字段决定了「自动最佳配置」能不能生效）
    for vname, cid in KNOWN.items():
        v = voices.get(vname)
        if not v:
            continue
        changed = False
        if not v.get("gender"):
            v["gender"] = cases.get(cid, {}).get("gender", "")
            changed = True
        if not v.get("bench_case") and cid in cases:
            v["bench_case"] = cid
            changed = True
        if changed:
            log.info("  ✅ %s 补上 gender=%s bench_case=%s",
                     vname, v.get("gender"), v.get("bench_case"))

    try:
        json.dump(voices, open(VOICES_JSON, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    except PermissionError:
        log.error("写 %s 没权限（目录属 root）。用：sudo ./venv/bin/python %s --apply",
                  VOICES_JSON, __file__)
        return 1
    log.info("\n已写入 %s，共 %d 个音色：%s",
             VOICES_JSON, len(voices), ", ".join(voices))
    return 0


if __name__ == "__main__":
    sys.exit(main())
