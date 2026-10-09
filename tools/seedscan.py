# -*- coding: utf-8 -*-
"""拒绝采样可行性验证：同一文案换 N 个种子，看还原度分数的**分布**。

为什么要有这个脚本
------------------
2026-10-08 r26 里发现：同一配置、同一案例（대한광통신）两次测量 **96.70 vs 94.40**，
差 2.3 分；标杆男三次 95.18 / 95.30 / 95.02。而我们辛苦抠的 lufs+peak 机制收益
只有 ~0.5 分 —— **噪声比信号大**。

但「波动大」本身也是资源：如果同文案换种子能拿到 94.4~96.7 的分布，那
「跑 N 个种子挑最高分」就能白吃上尾 +0.5~1.0，且**完全不改模型、不改链路**。
本脚本就是先回答那个前提问题：**分布到底长什么样**。

⚠️ 比分布更要紧的是分辨「真信号」还是「噪声」：
   只挑一个案例、只跑一次，挑到的高分很可能只是运气。所以这里
   (1) 每个种子都独立评分（不是复用产物）
   (2) 支持多案例对照（--cases 逗号分隔）
   (3) 输出极差/标准差，若极差 < 单次测量噪声（约 1~2 分）就说明没戏

用法
----
    # 单案例 5 个种子，取前 1200 字（控制单轮耗时）
    venv/bin/python tools/seedscan.py --case 1a6c20a60481 --seeds 42,777,1234,2024,31415 \
        --chars 1200 --prefix ss1

    # 两案例对照（验证不是只挑到噪声）
    venv/bin/python tools/seedscan.py --cases 1a6c20a60481,ba24c0925311 \
        --seeds 42,777,1234 --chars 1200 --prefix ss2

🔴 必须在 qwen 环境跑（librosa 1.0.0）；用 http://127.0.0.1:8000
（服务器访问自己的 Tailscale IP 不通）。产物落 /mnt/tts，绝不落 /tmp。
"""
import os
import re
import sys
import glob
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.eval5 import (login, submit_job, wait_job, wait_dir, case_text,
                         to_wav, CASES_DIR, OUT_DIR, WORK, FIVE, _post)
from tools.fidelity import measure, score, WEIGHTS  # noqa: E402

CASE_BY_ID = {c[0]: c for c in FIVE}
SHORT_BY_ID = {c[0]: c[1] for c in FIVE}


def cut_text(txt, chars):
    """按字数截断，但尽量停在句子边界上——截在半句话里会把模型逼出怪韵律，
    分数就不是「短一点」而是「坏了」了。"""
    if chars <= 0 or len(txt) <= chars:
        return txt
    seg = txt[:chars]
    m = None
    for match in re.finditer(r"[.!?。！？]\s*", seg):
        m = match
    if m and m.end() > chars * 0.5:
        return seg[:m.end()].strip()
    return seg.strip()


def one(cid, voice, seed, prefix, chars, case_target=True):
    """跑一个 (案例, 种子)，返回 (分数, 逐项偏差, 成品时长)。"""
    short = SHORT_BY_ID.get(cid, cid[:6])
    txt = cut_text(case_text(cid), chars)
    title = f"{prefix}-{short}-s{seed}"
    print(f"  提交 seed={seed}  {len(txt)} 字 -> {title}", flush=True)
    job = submit_job(txt, voice, title,
                     case_id=(cid if case_target else ""), seed=seed)
    jid = job.get("id") or job.get("job_id")
    st = wait_job(jid)
    print(f"    status {st.get('status')}", flush=True)
    wait_dir(title)
    prod_dir = os.path.join(OUT_DIR, title)
    mp3s = sorted(sum([glob.glob(os.path.join(prod_dir, p))
                       for p in ("*.mp3", "*.wav")], []))
    if not mp3s:
        print("    !! 没有产物，跳过", flush=True)
        return None, None, 0.0
    wav = to_wav(mp3s, os.path.join(WORK, f"{title}.wav"))
    case_wav = os.path.join(CASES_DIR, cid, "merged.wav")
    cm = measure(case_wav, "CASE")
    pm = measure(wav, "PROD")
    s, devs = score(cm, pm)
    print(f"    -> {s:.2f}%  时长 {pm['dur']:.1f}s", flush=True)
    return s, devs, pm["dur"]


def stats(vals):
    n = len(vals)
    if not n:
        return {}
    mean = sum(vals) / n
    sd = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd,
            "min": min(vals), "max": max(vals), "range": max(vals) - min(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="", help="案例 id，逗号分隔")
    ap.add_argument("--case", default="", help="单个案例 id（与 --cases 二选一）")
    ap.add_argument("--seeds", default="42,777,1234,2024,31415")
    ap.add_argument("--chars", type=int, default=1200,
                    help="取案例文案前多少字（控制单轮耗时，0=全文）")
    ap.add_argument("--prefix", default="ss")
    ap.add_argument("--voice", default="", help="覆盖默认音色")
    args = ap.parse_args()

    ids = [x.strip() for x in (args.cases or args.case).split(",") if x.strip()]
    if not ids:
        print("至少要给一个 --case / --cases")
        return 1
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    os.makedirs(WORK, exist_ok=True)
    login()

    rows, all_devs = [], {}
    for cid in ids:
        meta = CASE_BY_ID.get(cid)
        voice = args.voice or (meta[3] if meta else "EL_male")
        label = meta[4] if meta else cid[:6]
        print(f"\n=== {label} ({SHORT_BY_ID.get(cid, cid[:6])}) 音色={voice} ===",
              flush=True)
        for sd in seeds:
            s, devs, dur = one(cid, voice, sd, args.prefix, args.chars)
            if s is None:
                continue
            rows.append((label, SHORT_BY_ID.get(cid, cid[:6]), sd, s, dur))
            all_devs[f"{SHORT_BY_ID.get(cid, cid[:6])}#{sd}"] = devs

    if not rows:
        print("没有任何有效结果")
        return 1

    print("\n" + "=" * 74)
    print(f"{'案例':<12}{'seed':>10}{'还原度':>10}{'时长':>10}")
    for label, short, sd, s, dur in rows:
        print(f"{label:<12}{sd:>10}{s:>10.2f}{dur:>9.1f}s")

    print("\n按案例汇总")
    print(f"{'案例':<12}{'n':>4}{'均值':>9}{'标准差':>9}{'最低':>9}{'最高':>9}{'极差':>9}")
    for cid in ids:
        short = SHORT_BY_ID.get(cid, cid[:6])
        vs = [r[3] for r in rows if r[1] == short]
        st = stats(vs)
        if not st:
            continue
        print(f"{short:<12}{st['n']:>4}{st['mean']:>9.2f}{st['sd']:>9.2f}"
              f"{st['min']:>9.2f}{st['max']:>9.2f}{st['range']:>9.2f}")

    allv = [r[3] for r in rows]
    st = stats(allv)
    print(f"\n全体 {st['n']} 次：均值 {st['mean']:.2f}  标准差 {st['sd']:.2f}  "
          f"极差 {st['range']:.2f}（{st['min']:.2f} ~ {st['max']:.2f}）")
    print("判据：极差 > 1.0（单次测量噪声量级）才算「挑种子有得挑」；"
          "若只有 ~0.3 就别做拒绝采样，纯属白烧时间。")

    keys = ["A", "B", "E", "D", "F", "sil", "C", "dur", "lufs", "peak", "tone"]
    print("\n逐项偏差（%，看波动主要来自哪一维）")
    names = sorted(all_devs)
    print(f"{'维度':<8}{'权重':>6}" + "".join(f"{n.split('#')[1][:8]:>10}"
                                             for n in names))
    for k in keys:
        vals = [all_devs[n].get(k) for n in names]
        print(f"{k:<8}{WEIGHTS[k]:>6.2f}" +
              "".join(f"{(('%.1f' % v) if v is not None else '-'):>10}"
                      for v in vals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
