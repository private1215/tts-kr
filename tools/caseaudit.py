# -*- coding: utf-8 -*-
"""案例库体检：新增案例能不能进白名单、能不能合并到同一说话人。

## 为什么需要它

案例库从 5 个涨到 26 个（2026-10-07 员工批量上传）。记忆里的硬约束：

- `PROSO_CASE_SOURCE` 默认只认 `elevenlabs` → source 不对的案例**加了不生效**
- `CASE_BENCHMARK_TAG` 默认 `""`（同性别 + source 白名单全收）
  → 于是**同性别里混进不同说话人**是最危险的：多案例 EQ、响度中位、
    停顿样本、PROSO 目标全部会被污染
- 4 个男声已核验是同一个人（campplus 余弦 0.924~0.971，同人阈值 0.75）
- 女声只有 1 例（7.65 分钟），**不够做女声多案例 EQ / SFT**

所以它做三件事：
1. **元数据体检**：性别 / source / 时长 / 六维，看有没有异常值混进来
2. **说话人核验**（`--spk 1`）：campplus 余弦矩阵，把同性别分成同人簇
   ⚠️ campplus 吃 **kaldi fbank(80 mel)** 不是波形（喂波形 rank 报错）
3. **可合并量**：每个同人簇的总时长，对照 SFT 门槛（男 34 分钟已够）

用法（服务器，qwen 环境）：
    venv/bin/python tools/caseaudit.py
    venv/bin/python tools/caseaudit.py --spk 1
"""
import os
import sys
import json
import glob
import argparse
import itertools
from collections import Counter, defaultdict

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "app"))

SR = 24000
CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")
# 同人阈值（campplus 官方 0.75）
SAME_THR = float(os.environ.get("TTS_SPK_THR", "0.75"))
# whitelist 里只认这些 source
SRC_OK = set(filter(None, os.environ.get(
    "TTS_PROSO_CASE_SOURCE", "elevenlabs").split(",")))


def load_meta():
    p = "/tmp/cases.json"
    if os.path.exists(p):
        return json.load(open(p))
    # 退路：直接扫目录
    out = []
    for d in sorted(glob.glob(os.path.join(CASES_DIR, "*", ""))):
        out.append({"id": os.path.basename(d.rstrip("/"))})
    return out


def wav_of(cid):
    p = os.path.join(CASES_DIR, cid, "merged.wav")
    return p if os.path.exists(p) else None


def read24(path):
    """ffmpeg 解码到 24k float32。"""
    import subprocess
    import tempfile
    import soundfile as sf
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir="/tmp")
    os.close(fd)
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path,
                        "-ac", "1", "-ar", str(SR), tmp], check=True,
                       timeout=600)
        return np.asarray(sf.read(tmp, dtype="float32")[0], dtype=np.float32)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def emb_of(x, sr=SR):
    """campplus 说话人嵌入。⚠️ 必须喂 kaldi fbank(80 mel)，喂波形会 rank 报错。"""
    try:
        from app.voice_match import embed, fbank
    except Exception:
        return None
    try:
        f = fbank(x, sr)
        return embed(f)
    except Exception:
        return None


def meta_report(cases):
    print("=" * 96)
    print(f"案例库体检：{len(cases)} 条")
    print("=" * 96)
    print("性别  ", dict(Counter(c.get("gender") for c in cases)))
    print("来源  ", dict(Counter(c.get("source") for c in cases)))
    print("名字  ", dict(Counter(c.get("name") for c in cases)))
    print("上传者", dict(Counter(c.get("owner") for c in cases)))
    print("标签  ", dict(Counter(c.get("tags") for c in cases)))
    bad = [c for c in cases if (c.get("source") or "") not in SRC_OK]
    if bad:
        print(f"⚠️ source 不在白名单 {SRC_OK} 的有 {len(bad)} 条（**加了不生效**）：")
        for c in bad:
            print(f"     {c['id']}  source={c.get('source')!r} "
                  f"gender={c.get('gender')}")
    nofull = [c for c in cases if not c.get("has_full_text")]
    if nofull:
        print(f"⚠️ 缺整篇文案 {len(nofull)} 条："
              f" {[c['id'] for c in nofull]}")

    hdr = (f"{'id':<14}{'性别':<8}{'时长':>7}{'字数':>6}{'cps':>7}"
           f"{'LUFS':>7}{'peak':>6}{'A':>7}{'B':>7}{'C':>7}{'E':>7}"
           f"{'F':>6}{'segs':>6}{'段':>4}")
    print()
    print(hdr)
    print("-" * len(hdr))
    tot = defaultdict(float)
    for g in ("male", "female", None):
        for c in sorted([x for x in cases if x.get("gender") == g],
                        key=lambda y: y["id"]):
            p = c.get("prosody") or {}
            print(f"{c['id']:<14}{(c.get('gender') or '-'):<8}"
                  f"{c.get('total_dur') or 0:>7.1f}{c.get('total_chars') or 0:>6}"
                  f"{c.get('cps') or 0:>7.3f}{c.get('lufs') or 0:>7.1f}"
                  f"{c.get('peak_dbfs') or 0:>6.1f}"
                  f"{p.get('A') or 0:>7.2f}{p.get('B') or 0:>7.2f}"
                  f"{p.get('C') or 0:>7.2f}{p.get('E') or 0:>7.1f}"
                  f"{p.get('F') or 0:>6.2f}{p.get('segs') or 0:>6}"
                  f"{c.get('seg_count') or 0:>4}")
            tot[c.get("gender")] += c.get("total_dur") or 0
    print()
    for g, s in tot.items():
        print(f"  {g or '-':<8} 总时长 {s/60.0:7.2f} 分钟")


def spk_report(cases, thr=SAME_THR):
    print()
    print("=" * 96)
    print(f"说话人核验（campplus 余弦，同人阈值 {thr}）")
    print("=" * 96)
    groups = defaultdict(list)
    for c in cases:
        groups[c.get("gender")].append(c["id"])
    for g, ids in groups.items():
        if len(ids) < 2:
            print(f"\n[{g}] 只有 {len(ids)} 条，跳过")
            continue
        embs = {}
        for cid in ids:
            w = wav_of(cid)
            if not w:
                continue
            x = read24(w)
            if x is None or len(x) < SR:
                continue
            e = emb_of(x)
            if e is not None:
                embs[cid] = np.asarray(e, dtype=np.float64).ravel()
        print(f"\n[{g}] {len(embs)}/{len(ids)} 条取到嵌入")
        ks = sorted(embs)
        # 两两余弦
        M = np.zeros((len(ks), len(ks)))
        for i, j in itertools.combinations(range(len(ks)), 2):
            a, b = embs[ks[i]], embs[ks[j]]
            n = min(len(a), len(b))
            c = float(np.dot(a[:n], b[:n]) /
                      (np.linalg.norm(a[:n]) * np.linalg.norm(b[:n]) + 1e-12))
            M[i, j] = M[j, i] = c
        for i in range(len(ks)):
            M[i, i] = 1.0
        print("      " + " ".join(f"{k[:6]:>7}" for k in ks))
        for i, k in enumerate(ks):
            print(f"{k[:6]:>6} " + " ".join(f"{M[i,j]:7.3f}"
                                            for j in range(len(ks))))
        # 简单聚类：与簇内第一个成员比
        clusters = []
        for i, k in enumerate(ks):
            placed = False
            for cl in clusters:
                if np.mean([M[i, ks.index(m)] for m in cl]) >= thr:
                    cl.append(k)
                    placed = True
                    break
            if not placed:
                clusters.append([k])
        print(f"  → 分成 {len(clusters)} 个说话人簇（阈值 {thr}）：")
        byid = {c["id"]: c for c in cases}
        for n, cl in enumerate(clusters, 1):
            d = sum((byid[m].get("total_dur") or 0) for m in cl)
            print(f"     簇{n}: {len(cl)} 条，{d/60.0:.2f} 分钟  "
                  f"{[m[:6] for m in cl]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spk", default="0")
    ap.add_argument("--thr", default=str(SAME_THR))
    a = ap.parse_args()
    cases = load_meta()
    meta_report(cases)
    if a.spk == "1":
        spk_report(cases, float(a.thr))


if __name__ == "__main__":
    main()
