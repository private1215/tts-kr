# -*- coding: utf-8 -*-
"""说话人核验：案例库里哪些案例是同一个人。

## 为什么它必须存在

🔴 **真正的排异依据是 source 白名单 + 说话人身份，不是 tag**（tag 是人工标注，
员工新增案例忘打就"加了不生效"）。而多案例 EQ / 响度中位 / 停顿样本 /
PROSO 目标 / SFT 数据集，全都要**同一个人**才能合并 —— 混进不同说话人
就是互相污染。

r19 用它证明「4 个男声是同一个人」（余弦 0.924~0.971，同人阈值 0.75），
从而把 `CASE_BENCHMARK_TAG` 从 `标杆` 放开成 `""`。现在案例库涨到 26 条
（女声 1 → 11 例），必须重跑。

## 实现要点（都是踩过的）

1. **必须用 CosyVoice3 的 venv**（`/mnt/tts/projects/cosy/venv`），
   只有它有 `torchaudio.compliance.kaldi` + onnxruntime。
2. ⚠️ **campplus 吃 kaldi fbank(80 mel)，不是波形**。喂波形会 rank 报错。
   且必须**重采样到 16 kHz**（fbank 的 sample_frequency=16000）。
3. ⚠️ 官方 `EmbeddingExtractor.inference` 是**随机截 10 秒**
   （`random.randint(0, len - 10s)`）—— 同一条音频每次跑出来都不一样，
   直接拿来做聚类会得到随机结果。这里改成**多窗平均**：
   每 10 秒一个窗、各提一次嵌入、L2 归一化后取平均再归一化。

用法（服务器）：
    /mnt/tts/projects/cosy/venv/bin/python tools/spkchk.py
    /mnt/tts/projects/cosy/venv/bin/python tools/spkchk.py --gender female
"""
import os
import sys
import json
import glob
import argparse
import itertools
from collections import defaultdict

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import onnxruntime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")
CAMPPLUS = os.environ.get(
    "TTS_CAMPPLUS", "/mnt/tts/models/Fun-CosyVoice3-0.5B/campplus.onnx")
SR = 16000
WIN = 10 * SR            # 官方 max_len
SAME_THR = float(os.environ.get("TTS_SPK_THR", "0.75"))
WORK = "/mnt/tts/diag/spkwork"


def read16(path):
    import subprocess
    import tempfile
    import soundfile as sf
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir="/tmp")
    os.close(fd)
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path,
                        "-ac", "1", "-ar", str(SR), tmp], check=True,
                       timeout=900)
        return torch.from_numpy(sf.read(tmp, dtype="float32")[0]).unsqueeze(0)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _fbank(speech):
    f = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=SR)
    return f - f.mean(dim=0, keepdim=True)


def embed(sess, speech, nwin=12):
    """多窗平均嵌入。官方只随机取一窗，这里改成均匀取 nwin 窗后平均。"""
    n = speech.shape[1]
    if n < SR:
        return None
    if n <= WIN:
        idxs = [0]
    else:
        # 均匀铺 nwin 个窗的起点（含首尾可容纳范围）
        idxs = [int(round(i * (n - WIN) / max(nwin - 1, 1))) for i in range(nwin)]
    vs = []
    for s in idxs:
        seg = speech[:, s:s + WIN]
        if seg.shape[1] < SR // 2:
            continue
        feat = _fbank(seg).unsqueeze(0).cpu().numpy()
        e = sess.run(None, {sess.get_inputs()[0].name: feat})[0].flatten()
        e = np.asarray(e, dtype=np.float64)
        nv = np.linalg.norm(e)
        if nv > 1e-12:
            vs.append(e / nv)
    if not vs:
        return None
    v = np.mean(vs, axis=0)
    nv = np.linalg.norm(v)
    return v / nv if nv > 1e-12 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", default="")
    ap.add_argument("--thr", default=str(SAME_THR))
    ap.add_argument("--nwin", type=int, default=12)
    ap.add_argument("--save", default=os.path.join(WORK, "emb.npz"))
    ap.add_argument("--load", default="", help="复用上次算好的 npz，跳过重算")
    a = ap.parse_args()
    a.thr = float(a.thr)          # ⚠️ argparse 默认给的是 str，别拿它比 float
    os.makedirs(WORK, exist_ok=True)

    meta = {}
    p = "/tmp/cases.json"
    if os.path.exists(p):
        for c in json.load(open(p)):
            meta[c["id"]] = c

    opt = onnxruntime.SessionOptions()
    opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.intra_op_num_threads = 2
    sess = onnxruntime.InferenceSession(CAMPPLUS, sess_options=opt,
                                       providers=["CPUExecutionProvider"])

    ids = sorted(os.path.basename(d.rstrip("/"))
                 for d in glob.glob(os.path.join(CASES_DIR, "*", "")))
    embs = {}
    if a.load and os.path.exists(a.load):
        z = np.load(a.load)
        embs = {k: z[k] for k in z.files}
        print(f"复用已算好的嵌入 {a.load}：{len(embs)} 条")
    for cid in ids:
        if cid in embs and a.load:
            continue
        if a.gender and (meta.get(cid, {}).get("gender") or "") != a.gender:
            continue
        w = os.path.join(CASES_DIR, cid, "merged.wav")
        if not os.path.exists(w):
            continue
        sp = read16(w)
        if sp is None:
            continue
        e = embed(sess, sp, a.nwin)
        if e is not None:
            embs[cid] = e
            print(f"  {cid}  {sp.shape[1]/SR:7.1f}s  ok", flush=True)

    if a.save:
        os.makedirs(os.path.dirname(a.save), exist_ok=True)
        np.savez(a.save, **embs)
        print(f"\n嵌入已存 {a.save}")

    groups = defaultdict(list)
    for cid in embs:
        g = (meta.get(cid, {}).get("gender") or "unknown")
        groups[g].append(cid)

    print("\n" + "=" * 100)
    print(f"说话人核验：{len(embs)} 条，同人阈值 {a.thr}")
    print("=" * 100)
    for g, ks in sorted(groups.items()):
        ks = sorted(ks)
        if len(ks) < 2:
            print(f"\n[{g}] 只有 {len(ks)} 条，跳过")
            continue
        M = np.eye(len(ks))
        for i, j in itertools.combinations(range(len(ks)), 2):
            M[i, j] = M[j, i] = float(np.dot(embs[ks[i]], embs[ks[j]]))
        print(f"\n[{g}] {len(ks)} 条")
        print("        " + " ".join(f"{k[:6]:>7}" for k in ks))
        for i, k in enumerate(ks):
            print(f"{k[:6]:>7} " + " ".join(f"{M[i,j]:7.3f}"
                                            for j in range(len(ks))))
        off = [M[i, j] for i, j in itertools.combinations(range(len(ks)), 2)]
        print(f"  → 同性别内两两余弦：min {min(off):.3f} "
              f"中位 {np.median(off):.3f} max {max(off):.3f}")
        # 贪心聚类：与簇内**所有**成员都 >= 阈值才并入
        clusters = []
        for i, k in enumerate(ks):
            best = None
            for cl in clusters:
                if all(M[i, ks.index(m)] >= a.thr for m in cl):
                    best = cl
                    break
            if best is None:
                clusters.append([k])
            else:
                best.append(k)
        print(f"  → 分成 {len(clusters)} 个说话人簇：")
        for n, cl in enumerate(clusters, 1):
            d = sum((meta.get(m, {}).get("total_dur") or 0) for m in cl)
            print(f"     簇{n}: {len(cl)} 条  {d/60.0:6.2f} 分钟  "
                  f"{[m[:6] for m in cl]}")
        if len(clusters) > 1:
            print("  ⚠️ 不同簇 = 不同说话人，不能合并做多案例 EQ / SFT")


if __name__ == "__main__":
    main()
