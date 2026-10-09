#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 ASR 词级时间戳做**精确对齐**，构建 LoRA 训练切片。

为什么必须有这个脚本
--------------------
`tools/lora_data.py` 是按「字符数占比」把音频时长分摊到句子再切片的：
    bounds = cumsum(字符权重) * 总时长
这**不是**文本—语音对齐。语速起伏、句间停顿、原文与音频的细微差异都会让
切片里的声音和文字错位，模型就会学到漏读、重复和错误的断句位置。

本脚本复用 `tools/sft/prep.py` 已验证过的对齐思路：
    faster-whisper 词级时间戳 → difflib 把识别词对回原文 token → 句级时间范围
并做了两处关键加固（2026-10-09 试跑发现）：
  1. **ASR 词先过 _keep 再比对**：识别词常带标点（"공업,"），原文 token 已去标点，
     不过滤会把覆盖率强行压到 0.7~0.8，看起来像"对齐全坏了"。
  2. **逐句覆盖率闸门，而不是整段闸门**：长音频（4 分钟）整体覆盖率必然被
     少数识别差的句子拖低；改为逐句判断，丢弃对不上的句子、保留可信句子，
     切片只由可信句子拼成。

输出与 `lora_train.py` 完全兼容的 manifest.json：
    {ref:{wav,ref_text,bench_case}, sr:24000, train:[{wav,text,case,dur}], val:[...]}
训练时 `TTS_LORA_ROOT=<本脚本输出根> --speaker <speaker 名>` 即可，不用改训练脚本。

用法（服务器 qwen 环境）：
    TTS_LORA_ROOT=/mnt/tts/diag/lora_aligned \
      venv/bin/python tools/lora_data_align.py --speaker yohan_aligned --limit 4 --verify 6
    # 全量：
    TTS_LORA_ROOT=/mnt/tts/diag/lora_aligned \
      venv/bin/python tools/lora_data_align.py --speaker yohan_aligned
    # 换更大的 ASR 模型提覆盖率（慢 3~4 倍，但时间戳更准）：
    TTS_ASR_MODEL=Systran/faster-whisper-medium venv/bin/python tools/lora_data_align.py ...
"""
import os
import re
import sys
import json
import glob
import random
import argparse
import difflib

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)                     # 让 `from app.asr_check import ...` 能找到
sys.path.insert(0, os.path.join(HERE, "sft"))
from prep import asr_words, sent_split, _keep  # noqa: E402

ROOT = "/mnt/tts"
CASES_JSON = os.path.join(ROOT, "data", "cases.json")
CASES_DIR = os.path.join(ROOT, "data", "cases")
VOICES_JSON = os.path.join(ROOT, "data", "voices", "voices.json")
DEF_LORA_ROOT = os.path.join(ROOT, "data", "lora")
# 服务机 /mnt/tts/data/lora 是 root 目录，普通账号写不进去 →
# 用 TTS_LORA_ROOT 指到可写的持久盘（如 /mnt/tts/diag/lora_aligned）。
LORA_ROOT = os.environ.get("TTS_LORA_ROOT", DEF_LORA_ROOT)

SR_OUT = 24000          # 🔴 Qwen3-TTS codec 断言 24 kHz
# 切片长度：比 lora_data.py 的 10~28s 保守一点，降低显存与长程漂移风险
MIN_S, TARGET_S, MAX_S = 3.0, 12.0, 24.0
PAD = 0.08
# 语速合理区间：太慢/太快都是对齐失败或 ASR 抽风
CPS_LO, CPS_HI = 4.0, 11.0
MAX_CHARS = 220
# 案例级对齐质量闸（沿用 lora_data.py）
MAX_CASE_CER = 0.30
# 逐句覆盖率下限：低于此值的**句子**直接丢弃（不进任何切片）
SENT_COVER_MIN = 0.75
# 案例原声自身的 ASR 噪声底（faster-whisper-small）
NOISE_FLOOR = 0.2227


def align_spans(text, words, sent_min=SENT_COVER_MIN):
    """把 ASR 词时间戳对齐到原文的**句**。

    返回 (spans, seg_cover)：
      spans     = [(句子, start, end, 该句覆盖率)]，只含通过逐句闸门的可信句子
      seg_cover = 整段的 token 匹配比例（仅用于观察，不再作为整段丢弃依据）
    """
    sents = sent_split(text)
    if not words or not sents:
        return [], 0.0
    ref_tok, sent_of = [], []
    for i, s in enumerate(sents):
        for w in _keep(s):
            ref_tok.append(w)
            sent_of.append(i)
    if not ref_tok:
        return [], 0.0
    # 🔴 ASR 词也要过 _keep：识别结果常带标点/空格，不过滤会系统性压低覆盖率
    hyp_tok, hyp_t = [], []
    for w, st, en in words:
        toks = _keep(w)
        for t in toks:
            hyp_tok.append(t)
            hyp_t.append((float(st), float(en)))
    if not hyp_tok:
        return [], 0.0
    sm = difflib.SequenceMatcher(None, ref_tok, hyp_tok, autojunk=False)
    matched = {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            matched[a + k] = hyp_t[b + k]
    seg_cover = len(matched) / float(len(ref_tok))
    # 未匹配 token 沿用最近已匹配邻居的时间
    last, times = None, []
    for i in range(len(ref_tok)):
        if i in matched:
            last = matched[i]
        times.append(last)
    if not any(t is not None for t in times):
        return [], 0.0
    spans, prev_end = [], 0.0
    for i in range(len(sents)):
        idx = [k for k, s in enumerate(sent_of) if s == i]
        if not idx:
            continue
        # 逐句覆盖率：只统计**直接匹配**上的 token（不含沿用邻居的）
        nm = sum(1 for k in idx if k in matched)
        cov = nm / float(len(idx))
        if cov < sent_min:
            continue                     # 这句对不上 → 丢弃，不进任何切片
        st = en = None
        for k in idx:
            if times[k] is not None:
                st = times[k][0] if st is None else st
                en = times[k][1]
        if st is None:
            continue
        st = max(st, prev_end)
        if en is None or en <= st + 0.15:
            continue                     # 时间区间不可信
        spans.append((sents[i], float(st), float(en), round(cov, 3)))
        prev_end = en
    return spans, seg_cover


def pack(spans, total_dur, target=TARGET_S, maxs=MAX_S, pad=PAD):
    """把可信句子攒成 target~maxs 秒的块：装到 target 就断，超过 maxs 先断。"""
    chunks, buf, bs, be, bc = [], [], None, None, 1.0
    for s, a, b, c in spans:
        if bs is None:
            buf, bs, be, bc = [s], a, b, c
            continue
        if (b - bs) > maxs:
            chunks.append((buf, bs, be, bc))
            buf, bs, be, bc = [s], a, b, c
        else:
            buf.append(s)
            be, bc = b, min(bc, c)
            if (be - bs) >= target:
                chunks.append((buf, bs, be, bc))
                buf, bs, be, bc = [], None, None, 1.0
    if buf:
        chunks.append((buf, bs, be, bc))
    return [(t, max(0.0, a - pad), min(total_dur, b + pad), c)
            for t, a, b, c in chunks]


def load_ref(src_speaker):
    """取该说话人的克隆源：优先复用已有 manifest（与线上部署同一份），
    否则回落到 voices.json 查找。"""
    for base in (LORA_ROOT, DEF_LORA_ROOT):
        mp = os.path.join(base, src_speaker, "manifest.json")
        if os.path.exists(mp):
            m = json.load(open(mp, encoding="utf-8"))
            if m.get("ref", {}).get("wav"):
                return dict(m["ref"]), m.get("voice_key", "")
    voices = json.load(open(VOICES_JSON, encoding="utf-8"))
    key = next((k for k in voices
                if (voices[k].get("src_speaker") or "").lower() == src_speaker.lower()
                or k.lower() == f"{src_speaker.lower()}_male"
                or k.lower() == f"{src_speaker.lower()}_female"), None)
    if not key:
        sys.exit(f"找不到说话人 {src_speaker} 的克隆源音色")
    v = voices[key]
    return dict(wav=v["path"], ref_text=(v.get("ref_text") or "").strip(),
                bench_case=v.get("bench_case")), key


def resample24(x, sr):
    """从**原始采样率**直接重采样到 24k（别从 16k 再升上去）。"""
    if int(sr) == SR_OUT:
        return np.asarray(x, dtype=np.float32)
    g = gcd(int(sr), SR_OUT)
    return resample_poly(np.asarray(x, dtype=np.float32), SR_OUT // g,
                         int(sr) // g).astype(np.float32)


def build(args):
    out_dir = os.path.join(LORA_ROOT, args.speaker)
    os.makedirs(out_dir, exist_ok=True)
    ref, voice_key = load_ref(args.src_speaker)
    bench = args.holdout or (ref.get("bench_case") or "")
    print(f"说话人 {args.speaker}（源 {args.src_speaker} / 音色 {voice_key}）"
          f"  验证集案例 = {bench}", flush=True)
    print(f"ref = {ref['wav']}", flush=True)
    print(f"输出根 = {LORA_ROOT}  逐句覆盖率闸门 = {args.sent_cover}", flush=True)

    cases = json.load(open(CASES_JSON, encoding="utf-8"))
    cases = cases if isinstance(cases, list) else list(
        cases.get("cases", cases.values()))
    mine = [c for c in cases if (c.get("name") or "").strip() == args.src_speaker]
    if args.limit:
        mine = mine[:args.limit]
    print(f"候选案例 {len(mine)} 条", flush=True)

    train, val, qc, drop = [], [], [], {}
    for c in mine:
        cid = c["id"]
        segs = c.get("segments") or []
        bad = any((s.get("cer") or 0) > MAX_CASE_CER for s in segs)
        if bad:
            drop["case_cer"] = drop.get("case_cer", 0) + 1
            print(f"  SKIP {cid} 案例级 CER 超闸 → 整条剔除", flush=True)
            continue
        is_val = (cid == bench)
        cdir = os.path.join(CASES_DIR, cid)
        n_seg = 0
        for s in segs:
            wp = os.path.join(cdir, s["audio"])
            tp = os.path.join(cdir, s["text"])
            if not (os.path.isfile(wp) and os.path.isfile(tp)):
                drop["missing"] = drop.get("missing", 0) + 1
                continue
            text = open(tp, encoding="utf-8").read().strip()
            if not text:
                drop["empty_text"] = drop.get("empty_text", 0) + 1
                continue
            try:
                x, sr = sf.read(wp, dtype="float32", always_2d=False)
            except Exception as e:
                print(f"  !! 读不了 {wp}: {e}", flush=True)
                continue
            if getattr(x, "ndim", 1) > 1:
                x = x.mean(axis=1)
            dur = len(x) / float(sr)
            print(f"  {cid[:6]} seg{s.get('idx')} {dur:.1f}s {len(text)}字 "
                  f"-> ASR 对齐 ...", flush=True)
            words = asr_words(wp)
            spans, cover = align_spans(text, words, args.sent_cover)
            print(f"     整段覆盖率 {cover:.2f}  可信句 {len(spans)} 句",
                  flush=True)
            if not spans:
                drop["align_fail"] = drop.get("align_fail", 0) + 1
                continue
            for sents, a, b, scov in pack(spans, dur, args.target_s, args.max_s):
                txt = " ".join(sents).strip()
                # 🔴 语速按**去空格字符数**算，不是 _keep() 的词数：
                #    韩语实测约 5.7 字/秒，但只有 2~3 词/秒；用词数会让
                #    CPS_LO=4.0 把全部切片误杀（2026-10-09 试跑产出 0 片）。
                n = len(re.sub(r"\s+", "", txt))
                d = b - a
                if d < MIN_S or d > args.max_s + 2 * PAD:
                    drop["dur"] = drop.get("dur", 0) + 1
                    continue
                if n < 8 or n > MAX_CHARS:
                    drop["chars"] = drop.get("chars", 0) + 1
                    continue
                cps = n / d
                if not (CPS_LO <= cps <= CPS_HI):
                    drop["cps"] = drop.get("cps", 0) + 1
                    continue
                it = dict(wav=wp, t0=a, t1=b, sr=sr, text=txt, case=cid,
                          cover=scov, cps=round(cps, 2))
                (val if is_val else train).append(it)
                n_seg += 1
        print(f"  {'VAL ' if is_val else 'train'} {cid[:6]} 产出 {n_seg} 片",
              flush=True)

    # 验证集兜底：bench_case 缺失/被剔除时，留最后一个案例，避免 val 为空
    if train and not val:
        last = train[-1]["case"]
        val = [it for it in train if it["case"] == last]
        train = [it for it in train if it["case"] != last]
        print(f"\n⚠️ bench_case 无有效样本 → 改用 {last} 作为验证集", flush=True)

    def dump(items, split):
        sd = os.path.join(out_dir, split)
        os.makedirs(sd, exist_ok=True)
        out = []
        for n, it in enumerate(items):
            x, sr = sf.read(it["wav"], dtype="float32", always_2d=False)
            if getattr(x, "ndim", 1) > 1:
                x = x.mean(axis=1)
            a, b = int(it["t0"] * sr), int(it["t1"] * sr)
            y = resample24(x[max(0, a):min(len(x), b)], sr)
            p = os.path.join(sd, f"{n:04d}.wav")
            sf.write(p, y, SR_OUT)
            d = round(len(y) / float(SR_OUT), 2)
            out.append(dict(wav=p, text=it["text"], case=it["case"], dur=d))
            qc.append(dict(split=split, file=p, case=it["case"], dur=d,
                           chars=len(re.sub(r"\s+", "", it["text"])),
                           cps=it["cps"], cover=it["cover"]))
        return out

    print("\n落盘切片 ...", flush=True)
    train_items, val_items = dump(train, "train"), dump(val, "val")
    manifest = dict(speaker=args.speaker, src_speaker=args.src_speaker,
                    voice_key=voice_key, ref=ref, sr=SR_OUT,
                    align="asr_word_ts",
                    params=dict(sent_cover=args.sent_cover, target_s=args.target_s,
                                max_s=args.max_s, max_chars=MAX_CHARS,
                                cps=[CPS_LO, CPS_HI], max_case_cer=MAX_CASE_CER),
                    train=train_items, val=val_items)
    mp = os.path.join(out_dir, "manifest.json")
    json.dump(manifest, open(mp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(dict(drop=drop, qc=qc),
              open(os.path.join(out_dir, "qc.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    td = sum(i["dur"] for i in train_items)
    vd = sum(i["dur"] for i in val_items)
    print(f"\n✅ {mp}")
    print(f"train {len(train_items)} 片 / {td/60:.1f} 分钟；"
          f"val {len(val_items)} 片 / {vd/60:.1f} 分钟")
    print(f"丢弃统计: {drop}")
    if qc:
        cs = [q["cover"] for q in qc]
        print(f"切片句覆盖率: min {min(cs):.2f} / 中位 "
              f"{sorted(cs)[len(cs)//2]:.2f}")
    return manifest


def verify(out_dir, n=8):
    """抽 n 条切片做 ASR 回检：确认切片里的声音确实在说这段文本。

    判定：CER 中位数应低于噪声底 0.2227；**空转写直接算失败**（不能放行）。
    """
    from app.asr_check import transcribe, cer as _cer
    m = json.load(open(os.path.join(out_dir, "manifest.json"), encoding="utf-8"))
    pool = (m["train"] or []) + (m["val"] or [])
    if not pool:
        print("没有切片可验")
        return None
    random.seed(0)
    pick = random.sample(pool, min(n, len(pool)))
    outs, empty = [], 0
    for it in pick:
        hyp = transcribe(it["wav"], "ko")
        if not hyp:
            empty += 1
            outs.append(1.0)
            print(f"  🔴 空转写 {os.path.basename(it['wav'])}")
            continue
        e = _cer(it["text"], hyp)
        outs.append(e)
        print(f"  {os.path.basename(it['wav'])} CER={e:.4f} "
              f"({it['dur']}s) {it['text'][:28]}...")
    outs.sort()
    med = outs[len(outs) // 2]
    ok = (empty == 0) and (med < NOISE_FLOOR)
    print(f"\nCER 中位数 {med:.4f}（噪声底 {NOISE_FLOOR}）"
          f"  空转写 {empty} 条  {'✅' if ok else '🔴 不合格'}")
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", default="yohan_aligned",
                    help="输出说话人名（= LORA_ROOT 下的目录名，训练时 --speaker 用它）")
    ap.add_argument("--src-speaker", default="yohan",
                    help="案例库里的说话人名，用于取 ref 与该说话人的案例")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个案例（试跑用）")
    ap.add_argument("--holdout", default="", help="覆盖验证集案例 id")
    ap.add_argument("--sent-cover", type=float, default=SENT_COVER_MIN,
                    help="逐句 token 匹配覆盖率下限（低于则丢弃该句）")
    ap.add_argument("--target-s", type=float, default=TARGET_S)
    ap.add_argument("--max-s", type=float, default=MAX_S)
    ap.add_argument("--verify", type=int, default=0, help="抽 N 条做 ASR 回检")
    ap.add_argument("--verify-only", action="store_true",
                    help="不重建，只对已有 manifest 做 ASR 回检")
    a = ap.parse_args()
    if a.verify_only:
        verify(os.path.join(LORA_ROOT, a.speaker), a.verify or 8)
        return
    build(a)
    if a.verify:
        verify(os.path.join(LORA_ROOT, a.speaker), a.verify)


if __name__ == "__main__":
    main()
