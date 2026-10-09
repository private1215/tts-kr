#!/usr/bin/env python3
"""从案例库构建 LoRA 训练数据集。

切片策略（沿用案例对齐的既定做法）：
  - 按句末标点切句 → 按字符数占比把音频时长分摊到句子 → 句末切点
  - 相邻句拼成 10~28 秒的训练目标（target）
  - ref 固定用该说话人的克隆源（yohan_male.wav + 其文字稿），
    与线上部署时的参考音频完全一致 —— 训练即部署

数据清洗：任一 segment CER > 0.30 的案例整条剔除（对齐坏了不能喂模型）。
验证集：bench_case 那条案例单独留出（不参与训练），用于监控过拟合。

用法：
  venv/bin/python tools/lora_data.py yohan        # 构建 yohan 男声
  venv/bin/python tools/lora_data.py SOLA         # 构建 SOLA 女声
"""
import json
import os
import re
import sys

import numpy as np
import librosa

ROOT = "/mnt/tts"
CASES_JSON = os.path.join(ROOT, "data", "cases.json")
CASES_DIR = os.path.join(ROOT, "data", "cases")
VOICES_JSON = os.path.join(ROOT, "data", "voices", "voices.json")
OUT_ROOT = os.path.join(ROOT, "data", "lora")

MAX_CER = 0.30          # 案例级对齐质量闸
SLICE_MIN, SLICE_MAX = 10.0, 28.0   # target 目标秒数
SR_OUT = 24000


def split_sentences(text):
    parts = re.split(r"(?<=[.!?？！。])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def char_weight(s):
    return max(len(re.sub(r"\s+", "", s)), 1)


def main():
    speaker = sys.argv[1] if len(sys.argv) > 1 else "yohan"
    out_dir = os.path.join(OUT_ROOT, speaker)
    os.makedirs(out_dir, exist_ok=True)

    voices = json.load(open(VOICES_JSON, encoding="utf-8"))
    # ref = 该说话人的克隆源（训练与部署同一份）
    voice_key = next((k for k in voices
                      if (voices[k].get("src_speaker") or "").lower() == speaker.lower()
                      or k.lower() == f"{speaker.lower()}_male"
                      or k.lower() == f"{speaker.lower()}_female"), None)
    if not voice_key:
        sys.exit(f"找不到说话人 {speaker} 的克隆源音色")
    vinfo = voices[voice_key]
    ref = dict(wav=vinfo["path"], ref_text=(vinfo.get("ref_text") or "").strip(),
               bench_case=vinfo.get("bench_case"))
    print(f"ref 音色: {voice_key}  bench_case(验证集)= {ref['bench_case']}")

    cases = json.load(open(CASES_JSON, encoding="utf-8"))
    cases = cases if isinstance(cases, list) else list(cases.get("cases", cases.values()))
    mine = [c for c in cases if (c.get("name") or "").strip() == speaker]
    print(f"{speaker} 案例 {len(mine)} 条")

    train, val = [], []
    for c in mine:
        segs = c.get("segments") or []
        bad = any((s.get("cer") or 0) > MAX_CER for s in segs)
        bucket = val if c.get("id") == ref["bench_case"] and not bad else train
        tag = "VAL " if bucket is val else ("SKIP" if bad else "train")
        if bad:
            print(f"  {tag} {c.get('id')} dur={c.get('total_dur')} "
                  f"CER 超闸 {[round(s.get('cer') or 0, 2) for s in segs]} → 剔除")
            continue
        cid = c["id"]
        cdir = os.path.join(CASES_DIR, cid)
        for s in segs:
            wav_path = os.path.join(cdir, s["audio"])
            txt_path = os.path.join(cdir, s["text"])
            if not (os.path.isfile(wav_path) and os.path.isfile(txt_path)):
                print(f"  SKIP {cid} seg{s['idx']} 文件缺失")
                continue
            wav, sr = librosa.load(wav_path, sr=None, mono=True)
            text = open(txt_path, encoding="utf-8").read().strip()
            sents = split_sentences(text)
            if len(sents) < 2:
                continue
            dur = float(wav.shape[0]) / sr
            w = np.array([char_weight(x) for x in sents], dtype=np.float64)
            w = w / w.sum()
            bounds = np.concatenate([[0.0], np.cumsum(w) * dur])
            # 相邻句合并成 target
            group, g0 = [], None
            for i, sent in enumerate(sents):
                if g0 is None:
                    g0 = bounds[i]
                group.append(sent)
                g1 = bounds[i + 1]
                glen = g1 - g0
                nxt = bounds[i + 2] - g0 if i + 2 <= len(sents) else 1e9
                if glen >= SLICE_MIN and (glen >= SLICE_MAX or nxt > SLICE_MAX):
                    i0, i1 = int(g0 * sr), int(g1 * sr)
                    if i1 - i0 >= int(3.0 * sr):       # 最短 3 秒
                        cut = librosa.resample(y=wav[i0:i1].astype(np.float32),
                                                orig_sr=sr, target_sr=SR_OUT)
                        bucket.append(dict(wav=wav_path, t0=g0, t1=g1,
                                           sr=sr, text=" ".join(group),
                                           case=cid))
                    group, g0 = [], None
            if group and g0 is not None and dur - g0 >= 3.0:
                i0, i1 = int(g0 * sr), len(wav)
                cut = None
                bucket.append(dict(wav=wav_path, t0=g0, t1=dur,
                                   sr=sr, text=" ".join(group), case=cid))
            print(f"  {tag} {cid} seg{s['idx']} dur={dur:.0f}s 句数={len(sents)}")
        # 真正落盘在收集完后统一做（此版直接收集元数据）

    # 统一落盘切片
    def dump(items, split):
        paths = []
        sd = os.path.join(out_dir, split)
        os.makedirs(sd, exist_ok=True)
        for n, it in enumerate(items):
            wav, sr = librosa.load(it["wav"], sr=None, mono=True)
            i0, i1 = int(it["t0"] * sr), int(it["t1"] * sr)
            cut = librosa.resample(y=wav[i0:i1].astype(np.float32),
                                   orig_sr=sr, target_sr=SR_OUT)
            p = os.path.join(sd, f"{n:04d}.wav")
            librosa.output.write_wav(p, cut, SR_OUT) if hasattr(librosa, "output") \
                else __import__("soundfile").write(p, cut, SR_OUT)
            paths.append(dict(wav=p, text=it["text"], case=it["case"],
                              dur=round(cut.shape[0] / SR_OUT, 2)))
        return paths

    print("\n落盘 train 切片 ...")
    train_items = dump(train, "train")
    print("落盘 val 切片 ...")
    val_items = dump(val, "val")

    manifest = dict(speaker=speaker, voice_key=voice_key, ref=ref,
                    sr=SR_OUT, train=train_items, val=val_items)
    mp = os.path.join(out_dir, "manifest.json")
    json.dump(manifest, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    td = sum(x["dur"] for x in train_items)
    vd = sum(x["dur"] for x in val_items)
    print(f"\n✅ {mp}")
    print(f"train: {len(train_items)} 片 / {td/60:.1f} 分钟; "
          f"val: {len(val_items)} 片 / {vd/60:.1f} 分钟")


if __name__ == "__main__":
    main()
