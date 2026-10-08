# -*- coding: utf-8 -*-
"""从案例库构建 CosyVoice3 SFT 数据集（kaldi 四件套）。

为什么放进仓库
--------------
上一版 `/tmp/sft_prep2.py` 在 2026-10-07 重启时被清空，316 条训练数据全没了。
**数据集构建脚本属于项目资产**，必须可复现，所以固化在这里。

做什么
------
案例库里每个案例是「几分钟音频 + 逐字文案」，分段存在：
    /mnt/tts/data/cases/<id>/seg0.wav  seg0.txt  seg1.wav  seg1.txt ... （或 full.txt）

SFT 要的是 **10~15 秒** 的短音频 + 精确文本。所以核心问题是**对齐**：
把 ASR 的词级时间戳对回原文，再按句边界切。

🔴 必须踩过的三个坑（都在这份代码里修掉了）
1. `faster-whisper` 的 `word_timestamps` 返回的是**秒**，老版本当成样本索引用，
   算出来 dur 全是 1e-4，被 `dur < 1.0` 全部丢掉 → 产出 **0 条**。
2. kaldi 四件套（`wav.scp` / `text` / `utt2spk` / `spk2utt`）必须**先攒后写**，
   按案例循环用 "w" 模式会互相覆盖，只剩最后一个案例。
3. 单条字数上限别按「字」估：Qwen/Cosy 的 tokenizer 对韩语只有 **0.67 token/字**
   （实测 40 字 → 24 token、150 字 → 100 token），所以 140 字是安全的，
   70 字会误杀掉 43 组里的 38 组。

用法（服务器，qwen 环境）：
    venv/bin/python tools/sft/prep.py --out /mnt/tts/data/sft/male --gender male
    venv/bin/python tools/sft/prep.py --out /mnt/tts/data/sft/male --verify
"""
import os
import re
import glob
import json
import random
import argparse
import difflib

import numpy as np
import soundfile as sf

CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")
# CosyVoice3 官方脚本把音频重采样到 16k
SR = 16000
# 🔴 Qwen3-TTS 的 codec pipeline **断言 24 kHz**，别的采样率会在训练深处崩，
#    且没有任何提前报错。所以导出 Qwen 数据集时必须另写一份 24k。
QWEN_SR = 24000
# 切分目标/上限（秒）。15 秒是「参考音频甜点」，也是显存与上下文的折中
TARGET_S = 10.0
MAX_S = 15.0
MIN_S = 1.0
PAD = 0.08
# 语速合理区间：太慢/太快都是对齐失败或 ASR 抽风
CPS_LO, CPS_HI = 4.0, 11.0
# 单条字数上限（见文件头第 3 点）
MAX_CHARS = 140
# CosyVoice3 LM 需要 instruct 字段，缺了会 KeyError: 'instruct_token'
INSTRUCT = "You are a helpful assistant.<|endofprompt|>"


def _keep(t):
    return re.findall(r"[가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]+", t or "")


def read_case(cid):
    """取 ([(wav路径, 文本)], 性别)。"""
    d = os.path.join(CASES_DIR, cid)
    pairs = []
    full = os.path.join(d, "full.txt")
    if os.path.exists(full):
        # full.txt 覆盖 merged.wav（整篇没按段存文本）
        m = os.path.join(d, "merged.wav")
        if os.path.exists(m):
            return [(m, open(full, encoding="utf-8").read())]
    for w in sorted(glob.glob(os.path.join(d, "seg*.wav"))):
        t = w[:-4] + ".txt"
        if os.path.exists(t):
            pairs.append((w, open(t, encoding="utf-8").read().strip()))
    return pairs


def sent_split(text):
    """按句末标点切，保留标点。"""
    parts = re.split(r"(?<=[.!?。！？\n])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def asr_words(wav, lang="ko"):
    """返回 [(词, start, end)]，秒。失败返回 []。"""
    from faster_whisper import WhisperModel
    global _M
    try:
        _M
    except NameError:
        _M = WhisperModel(os.environ.get("TTS_ASR_MODEL",
                                         "Systran/faster-whisper-small"),
                          device="cpu", compute_type="int8", cpu_threads=4,
                          local_files_only=True)
    x, sr = sf.read(wav, dtype="float32")
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    if int(sr) != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    segs, _ = _M.transcribe(x, language=lang, word_timestamps=True, beam_size=1)
    out = []
    for s in segs:
        for w in (getattr(s, "words", None) or []):
            out.append((w.word.strip(), float(w.start), float(w.end)))
    return out


def align(ref_text, words):
    """把 ASR 词时间戳对齐到原文的**句**，返回 [(句, start, end)]。

    用 token 级 difflib，抗 ~9% 的 ASR 错误率。
    ⚠️ 不能按「累计字符数比例」切 —— 那样算出来的语速 CV 必然是 0.000，
    而且错一句之后会一路漂到隔壁句子去。
    """
    sents = sent_split(ref_text)
    if not words:
        return []
    ref_tok, sent_of = [], []
    for i, s in enumerate(sents):
        for w in _keep(s):
            ref_tok.append(w)
            sent_of.append(i)
    hyp_tok = [w for w, _, _ in words]
    sm = difflib.SequenceMatcher(None, ref_tok, hyp_tok, autojunk=False)
    matched = {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            matched[a + k] = words[b + k]
    # 未匹配的取最近已匹配邻居的时间
    last = None
    times = []
    for i in range(len(ref_tok)):
        if i in matched:
            last = matched[i]
        times.append(last)
    if not any(t is not None for t in times):
        return []
    # 每句取首词 start / 末词 end；缺失则沿用前一个可用值
    out, prev_end = [], 0.0
    for i in range(len(sents)):
        idx = [k for k, s in enumerate(sent_of) if s == i]
        if not idx:
            continue
        st = en = None
        for k in idx:
            if times[k] is not None:
                st = times[k][1] if st is None else st
                en = times[k][2]
        if st is None:
            continue
        st = max(st, prev_end)
        if en is None or en <= st:
            en = st + 1.0
        out.append((sents[i], float(st), float(en)))
        prev_end = en
    return out


def pack(spans, total_dur):
    """把句子攒成 10~15 秒的块：装到 TARGET_S 就断，超过 MAX_S 先断。"""
    chunks, buf, bs, be = [], [], None, None
    for s, a, b in spans:
        if bs is None:
            buf, bs, be = [s], a, b
            continue
        if (b - bs) > MAX_S:
            chunks.append((buf, bs, be))
            buf, bs, be = [s], a, b
        else:
            buf.append(s)
            be = b
            if (be - bs) >= TARGET_S:
                chunks.append((buf, bs, be))
                buf, bs, be = [], None, None
    if buf:
        chunks.append((buf, bs, be))
    return [(t, max(0.0, a - PAD), min(total_dur, b + PAD)) for t, a, b in chunks]


def build(cases, out_dir, gender, holdout, qwen=None):
    """qwen=None/False：只出 CosyVoice3 kaldi 四件套。
    qwen={"ref": "self"|"voice"|"<路径>"}：额外出 Qwen3-TTS 的 train_raw.jsonl。
    """
    os.makedirs(out_dir, exist_ok=True)
    recs = []
    for cid in cases:
        for wav, txt in read_case(cid):
            if not txt.strip():
                continue
            try:
                x, sr = sf.read(wav, dtype="float32")
            except Exception as e:
                print(f"  !! 读不了 {wav}: {e}")
                continue
            if getattr(x, "ndim", 1) > 1:
                x = x.mean(axis=1)
            dur = len(x) / sr
            print(f"  {cid[:6]} {os.path.basename(wav)} {dur:.1f}s "
                  f"{len(txt)}字 -> ASR 对齐中 ...", flush=True)
            words = asr_words(wav)
            spans = align(txt, words)
            if not spans:
                print("     !! 对齐失败，跳过")
                continue
            for sents, a, b in pack(spans, dur):
                text = " ".join(sents).strip()
                n = len(_keep(text))
                d = b - a
                if d < MIN_S or d > MAX_S + 2 * PAD:
                    continue
                if n < 4 or n > MAX_CHARS:
                    continue
                if not (CPS_LO <= n / d <= CPS_HI):
                    continue
                recs.append(dict(cid=cid, wav=wav, text=text,
                                 a=a, b=b, dur=d, chars=n))
    if not recs:
        print("🔴 一条都没产出来")
        return None
    # 留一验证：拿一个完整案例做 dev
    cids = sorted(set(r["cid"] for r in recs))
    dev_cid = holdout or (cids[-1] if len(cids) > 1 else None)
    train = [r for r in recs if r["cid"] != dev_cid]
    dev = [r for r in recs if r["cid"] == dev_cid]
    if not train or not dev:
        print("⚠️ 案例数不足，全部进 train")
        train, dev = recs, recs[:1]

    random.seed(42)
    random.shuffle(train)
    spk = "SPK_" + (gender or "x").upper()
    for name, rs in (("train", train), ("dev", dev)):
        d = os.path.join(out_dir, name)
        os.makedirs(d, exist_ok=True)
        wd = os.path.join(d, "wav")
        os.makedirs(wd, exist_ok=True)
        wd24 = os.path.join(d, "wav24k")
        if qwen:
            os.makedirs(wd24, exist_ok=True)
        wav_scp, text_f, utt2spk = [], [], []
        for i, r in enumerate(rs):
            uid = f"{name}_{r['cid'][:6]}_{i:05d}"
            x, sr = sf.read(r["wav"], dtype="float32")
            if getattr(x, "ndim", 1) > 1:
                x = x.mean(axis=1)
            a, b = int(r["a"] * sr), int(r["b"] * sr)
            yraw = x[max(0, a - int(PAD * sr)):min(len(x), b + int(PAD * sr))]
            if sr != SR:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(int(sr), SR)
                y = resample_poly(yraw, SR // g, int(sr) // g).astype(np.float32)
            else:
                y = yraw
            p = os.path.join(wd, uid + ".wav")
            sf.write(p, y, SR)
            if qwen:
                # 从**原始采样率**直接重采样到 24k，别从 16k 再升上去
                # （等于先丢一次高频再插值回来）。
                if sr != QWEN_SR:
                    from math import gcd
                    from scipy.signal import resample_poly
                    g2 = gcd(int(sr), QWEN_SR)
                    y24 = resample_poly(yraw, QWEN_SR // g2,
                                        int(sr) // g2).astype(np.float32)
                else:
                    y24 = yraw
                sf.write(os.path.join(wd24, uid + ".wav"), y24, QWEN_SR)
            wav_scp.append(f"{uid}\t{p}")
            text_f.append(f"{uid}\t{r['text']}")
            utt2spk.append(f"{uid}\t{spk}")
        # 🔴 先攒后写，别在循环里开 "w"
        open(os.path.join(d, "wav.scp"), "w", encoding="utf-8").write(
            "\n".join(wav_scp) + "\n")
        open(os.path.join(d, "text"), "w", encoding="utf-8").write(
            "\n".join(text_f) + "\n")
        open(os.path.join(d, "utt2spk"), "w", encoding="utf-8").write(
            "\n".join(utt2spk) + "\n")
        open(os.path.join(d, "spk2utt"), "w", encoding="utf-8").write(
            f"{spk}\t" + " ".join(u.split("\t")[0] for u in utt2spk) + "\n")
        # CosyVoice3 官方 make_parquet 要这两个文件
        open(os.path.join(d, "text.normalized"), "w", encoding="utf-8").write(
            "\n".join(text_f) + "\n")
        # 🔴 缺 instruct 会在 CosyVoice3LM.forward 里 KeyError: 'instruct_token'
        open(os.path.join(d, "instruct"), "w", encoding="utf-8").write(INSTRUCT)
        sec = sum(r["dur"] for r in rs)
        print(f"  {name}: {len(rs)} 条 / {sec/60:.1f} 分钟 -> {d}")
    json.dump({"gender": gender, "dev_case": dev_cid,
               "n_train": len(train), "n_dev": len(dev)},
              open(os.path.join(out_dir, "meta.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n✅ 数据集就绪：{out_dir}（dev 留一 = {dev_cid}）")
    if qwen:
        export_qwen(out_dir, qwen.get("ref", "self"), qwen.get("arg", ""))
    return out_dir


def _case_dir_by_prefix(pref):
    """uid 里只带 cid 前 6 位，靠前缀找回案例完整目录。"""
    hits = [d for d in os.listdir(CASES_DIR)
            if d.startswith(pref) and os.path.isdir(os.path.join(CASES_DIR, d))]
    return os.path.join(CASES_DIR, hits[0]) if hits else None


def _ref_wav_for(cid_pref, ref_dir):
    """给一个案例截 13 秒参考音频（24k），音色与该案例自洽。"""
    d = _case_dir_by_prefix(cid_pref)
    if not d:
        return None
    src = None
    for cand in ("merged.wav", "seg0.wav"):
        p = os.path.join(d, cand)
        if os.path.exists(p):
            src = p
            break
    if src is None:
        w = sorted(glob.glob(os.path.join(d, "*.wav")))
        src = w[0] if w else None
    if src is None:
        return None
    dst = os.path.join(ref_dir, cid_pref + ".wav")
    if os.path.exists(dst):
        return dst
    x, sr = sf.read(src, dtype="float32")
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    x = x[: int(13.0 * sr)]
    if len(x) < sr:                      # 太短就不用它
        return None
    if sr != QWEN_SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), QWEN_SR)
        x = resample_poly(x, QWEN_SR // g, int(sr) // g).astype(np.float32)
    sf.write(dst, x, QWEN_SR)
    return dst


def export_qwen(out_dir, ref_mode="self", ref_arg=""):
    """把已生成的 24k wav 写成 Qwen3-TTS 的 train_raw.jsonl。

    每行必须有 `audio` / `text` / `ref_audio` 三个字段（官方 README）。

    ref_mode
    --------
    self   每条用它**所属案例**自己截的 13 秒参考段 → 音色自洽，
           模型学到的是"这段话的韵律风格"，而不是被迫 fitting 一个外来音色。
    voice  全部共用同一个 ref（ref_arg 给路径）→ 官方推荐，说话人一致性最好，
           但 ref 与 audio 音色不一致时模型会被拉扯。
    <路径> 同上，直接给路径。

    🔴 官方 README 强烈建议所有样本共用同一个 ref_audio。**但那是为了训出
       一个固定音色**；我们要的是把案例的**韵律风格**迁移到自己的克隆源上，
       所以 self 与 voice 都得试，别默认盲信官方。
    """
    made = {}
    ref_dir = os.path.join(out_dir, "ref24k")
    os.makedirs(ref_dir, exist_ok=True)
    for name in ("train", "dev"):
        d = os.path.join(out_dir, name)
        scp = os.path.join(d, "wav.scp")
        txtf = os.path.join(d, "text")
        if not os.path.exists(scp):
            continue
        texts = {}
        for line in open(txtf, encoding="utf-8"):
            if "\t" in line:
                u, t = line.rstrip("\n").split("\t", 1)
                texts[u] = t
        lines, skipped = [], 0
        for line in open(scp, encoding="utf-8"):
            if "\t" not in line:
                continue
            uid, p16 = line.rstrip("\n").split("\t", 1)
            # wav.scp 存的是 16k 路径，24k 版在同级 wav24k/ 下同名
            p24 = os.path.join(os.path.dirname(p16), "..", "wav24k",
                               os.path.basename(p16))
            p24 = os.path.normpath(p24)
            if not os.path.exists(p24):
                skipped += 1
                continue
            cid6 = uid.split("_")[1] if uid.count("_") >= 2 else ""
            if ref_mode == "self":
                ref = made.get(cid6)
                if ref is None:
                    ref = _ref_wav_for(cid6, ref_dir)
                    made[cid6] = ref
            else:
                ref = ref_arg or (ref_mode if os.path.exists(ref_mode) else "")
            if not ref:
                skipped += 1
                continue
            lines.append(json.dumps(
                {"audio": p24, "text": texts.get(uid, ""), "ref_audio": ref},
                ensure_ascii=False))
        out = os.path.join(d, "train_raw.jsonl")
        open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"  Qwen jsonl: {len(lines)} 条 -> {out}"
              + (f"（跳过 {skipped} 条）" if skipped else ""))
    json.dump({"ref_mode": ref_mode, "ref_arg": ref_arg},
              open(os.path.join(out_dir, "qwen_meta.json"), "w"),
              ensure_ascii=False, indent=1)


def verify(out_dir, n=12):
    """随机抽 n 条做 ASR 回检：CER 中位数应 < 噪声底 0.2227。"""
    from app.asr_check import transcribe, cer as _cer
    d = os.path.join(out_dir, "train")
    txt = {}
    for line in open(os.path.join(d, "text"), encoding="utf-8"):
        u, t = line.rstrip("\n").split("\t", 1)
        txt[u] = t
    scp = [l.rstrip("\n").split("\t") for l in
           open(os.path.join(d, "wav.scp"), encoding="utf-8")]
    random.seed(0)
    pick = random.sample(scp, min(n, len(scp)))
    out = []
    for u, p in pick:
        hyp = transcribe(p, "ko")
        out.append(_cer(txt[u], hyp))
    out.sort()
    med = out[len(out) // 2]
    print(f"CER 中位数 {med:.4f}（噪声底 0.2227）"
          f"  {'✅' if med < 0.2227 else '🔴 偏高'}")
    print("  逐条:", out)
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出目录，放 /mnt/tts 下！")
    ap.add_argument("--gender", default="male")
    ap.add_argument("--cases", default="", help="案例 id 逗号分隔；默认取全部")
    ap.add_argument("--holdout", default="", help="留一验证用的案例 id")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--qwen", default="",
                    help="额外导出 Qwen3-TTS 的 train_raw.jsonl（24k）。"
                         "接 self=每条用本案例自截参考段（默认）/ voice:<wav路径>"
                         "=全部共用该参考音频")
    a = ap.parse_args()
    qwen = None
    if a.qwen:
        if a.qwen.startswith("voice:"):
            qwen = {"ref": "voice", "arg": a.qwen.split(":", 1)[1]}
        elif a.qwen.startswith("/"):
            qwen = {"ref": "voice", "arg": a.qwen}
        else:
            qwen = {"ref": "self", "arg": ""}
    if a.verify:
        verify(a.out)
        return
    cases = a.cases.split(",") if a.cases else sorted(
        d for d in os.listdir(CASES_DIR)
        if os.path.isdir(os.path.join(CASES_DIR, d)))
    print(f"{len(cases)} 个案例 -> {a.out}")
    build(cases, a.out, a.gender, a.holdout or None, qwen)


if __name__ == "__main__":
    main()
