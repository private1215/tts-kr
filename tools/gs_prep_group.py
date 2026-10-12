# -*- coding: utf-8 -*-
"""为 GPT-SoVITS 准备**一整个说话人组**的训练数据（支持多条案例）。

🔴 与旧脚本 prep_data.py 的区别
--------------------------------
旧的 prep_data.py 硬编码单个案例（韩华 c3117b7b0fb1），而案例库实际是
**一个说话人有十几条案例**（2026-10-12 嵌入聚类实测：SOLA 22 条 / yohan 13 条）。
只用一条案例做微调，等于放着 20 倍的素材不用。

这个脚本按**声音**聚类拿到某人的全部案例，再合并成一份训练集。

🔴 关键：**必须留出测试区**，否则模型只是背出了对比用的那几句，
对比结果毫无意义。
   - 测试区 = hold 案例的前 HOLD_SEC 秒（这条不再进训练集）
   - 训练集 = 其余案例（按 --max-minutes 截断，避免训练过久）

用法（用 qwen3-tts 的 venv 跑，因为 ASR 在那边）
------------------------------------------------
    /mnt/tts/projects/qwen3-tts/venv/bin/python \\
        tools/gs_prep_group.py --spk SOLA --max-minutes 40
    /mnt/tts/projects/qwen3-tts/venv/bin/python \\
        tools/gs_prep_group.py --spk yohan --max-minutes 40

输出
----
    /mnt/tts/projects/gpt-sovits/dataset_<spk>/{train,val}.list
"""
import os
import sys
import json
import random
import argparse
import subprocess

import numpy as np

BASE = os.environ.get("TTS_BASE", "").strip() or "/mnt/tts"
GS_DIR = os.environ.get("GS_DIR", "/mnt/tts/projects/gpt-sovits")
CASES_JSON = os.path.join(BASE, "data", "cases.json")
CASES_DIR = os.path.join(BASE, "data", "cases")

SR_OUT = 32000         # v2Pro 系列原生 32k
HOLD_SEC = 60.0        # hold 案例的前 60 秒留作测试
MIN_SEC, MAX_SEC = 2.0, 11.0
SIL_THRESH_DB = -38.0
SIL_MIN_SEC = 0.35
FRAME = 0.02


def load_wav(path, sr_out=SR_OUT):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if int(sr) != sr_out:
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
             "-ac", "1", "-ar", str(sr_out), "-f", "f32le", "-"],
            capture_output=True)
        x = np.frombuffer(r.stdout, dtype="<f4")
    return np.asarray(x, dtype=np.float32)


def slice_(x, sr):
    """按静音切成 2~11 秒短段。返回 [(start_s, end_s), ...]"""
    n = int(FRAME * sr)
    m = (len(x) // n) * n
    if m < n:
        return []
    fr = x[:m].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)
    voiced = db > SIL_THRESH_DB
    segs, s = [], None
    for i, v in enumerate(voiced):
        if v and s is None:
            s = i
        elif not v and s is not None:
            segs.append((s, i))
            s = None
    if s is not None:
        segs.append((s, len(voiced)))
    merged = []
    for a, b in segs:
        if merged and (a - merged[-1][1]) * FRAME < SIL_MIN_SEC:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    out = []
    for a, b in merged:
        st, en = a * FRAME, b * FRAME
        d = en - st
        if d < MIN_SEC:
            continue
        if d <= MAX_SEC:
            out.append((st, en))
        else:
            k = int(np.ceil(d / MAX_SEC))
            step = d / k
            for j in range(k):
                out.append((st + j * step, min(st + (j + 1) * step, en)))
    return out


def get_group(spk, thr=0.75):
    """复用 mkcase_voices 的嵌入聚类，拿到这个说话人的全部案例 id。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mkcase_voices as mk
    cases = mk.load_json(CASES_JSON, {})
    grp, meta = mk.embed_groups(cases, thr)
    for (nm, gd), cids in grp.items():
        if nm.lower() == spk.lower():
            return sorted(cids), meta.get((nm, gd), {})
    raise SystemExit("找不到说话人组 %s（现有：%s）"
                     % (spk, sorted({k[0] for k in grp})))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spk", required=True, help="说话人组名，如 SOLA / yohan")
    ap.add_argument("--max-minutes", type=float, default=40.0,
                    help="训练音频总时长上限（分钟）")
    ap.add_argument("--hold", default="",
                    help="留出作测试区的案例 id（默认取组内第一条）")
    args = ap.parse_args()

    cids, meta = get_group(args.spk)
    print("说话人组 %s：%d 条案例，组内最低相似度 %.4f"
          % (args.spk, len(cids), meta.get("sim_min", 0)), flush=True)

    hold = args.hold.strip() or cids[0]
    if hold not in cids:
        raise SystemExit("hold 案例 %s 不在这组里" % hold)
    train_cids = [c for c in cids if c != hold]
    print("留出测试区：%s（前 %.0fs），其余 %d 条进训练集"
          % (hold, HOLD_SEC, len(train_cids)), flush=True)

    out = os.path.join(GS_DIR, "dataset_" + args.spk)
    os.makedirs(os.path.join(out, "train"), exist_ok=True)
    os.makedirs(os.path.join(out, "val"), exist_ok=True)

    # ---- 累积音频，直到 max_minutes --------------------------------------
    budget = args.max_minutes * 60.0
    pieces = []          # [(cid, x)]  保持案例边界，切片时逐条处理
    got = 0.0
    for cid in train_cids:
        if got >= budget:
            break
        p = os.path.join(CASES_DIR, cid, "merged.wav")
        if not os.path.exists(p):
            print("  跳过 %s（无 merged.wav）" % cid[:8], flush=True)
            continue
        x = load_wav(p)
        d = len(x) / float(SR_OUT)
        if got + d > budget:
            keep = int((budget - got) * SR_OUT)
            if keep < SR_OUT * 5:
                break
            x = x[:keep]
            d = keep / float(SR_OUT)
        pieces.append((cid, x))
        got += d
        print("  + %s  %.1fs（累计 %.1f min）" % (cid[:8], d, got / 60.0), flush=True)
    print("训练素材合计 %.1f 分钟" % (got / 60.0), flush=True)

    # ---- ASR --------------------------------------------------------------
    sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
    os.environ.setdefault("HF_HOME", "/mnt/tts/models/asr")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from app.asr_check import transcribe
    import soundfile as sf
    print("ASR 载入完成", flush=True)

    tmp = os.path.join(out, "_tmp_seg.wav")
    rows = []
    gi = 0
    for cid, x in pieces:
        segs = slice_(x, SR_OUT)
        for (st, en) in segs:
            seg = x[int(st * SR_OUT):int(en * SR_OUT)]
            if len(seg) < SR_OUT:
                continue
            sf.write(tmp, seg, SR_OUT, subtype="PCM_16")
            txt = (transcribe(tmp, "ko") or "").strip()
            if len(txt) < 4:
                continue
            rows.append({"i": gi, "cid": cid, "x": seg, "txt": txt})
            gi += 1
        print("  %s 切出 %d 段（累计有效 %d）" % (cid[:8], len(segs), len(rows)),
              flush=True)
    if os.path.exists(tmp):
        os.remove(tmp)
    print("有效样本 %d 条" % len(rows), flush=True)
    if len(rows) < 20:
        raise SystemExit("有效样本太少（%d），放弃" % len(rows))

    # ---- 训练 / 验证分离 ---------------------------------------------------
    random.Random(42).shuffle(rows)
    n_val = max(3, int(len(rows) * 0.12))
    val, train = rows[:n_val], rows[n_val:]

    spk_id = args.spk.lower()
    lines = []
    for split, subset in (("train", train), ("val", val)):
        ls = []
        for r in subset:
            name = "%s_%05d.wav" % (spk_id, r["i"])
            p = os.path.join(out, split, name)
            sf.write(p, r["x"], SR_OUT, subtype="PCM_16")
            ls.append("%s|%s|ko|%s" % (p, spk_id, r["txt"]))
        with open(os.path.join(out, split + ".list"), "w", encoding="utf-8") as f:
            f.write("\n".join(ls) + "\n")
        lines.append(len(ls))
        print("%s: %d 条" % (split, len(ls)), flush=True)

    with open(os.path.join(out, "hold.txt"), "w", encoding="utf-8") as f:
        f.write("测试区 = 案例 %s 的前 %.1f 秒（未进训练集）\n" % (hold, HOLD_SEC))
        f.write("训练集素材来自 %d 条案例，合计 %.1f 分钟\n"
                % (len(pieces), got / 60.0))
    print("DONE -> %s（train %d / val %d，测试区案例 %s）"
          % (out, lines[0], lines[1], hold), flush=True)


if __name__ == "__main__":
    main()
