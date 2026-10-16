# -*- coding: utf-8 -*-
"""第 47 轮：SOLA 女声微调(e30 权重) vs 现役女声(omnivoice) —— 只比女声。

## 为什么这么设计
员工反馈「女生的少了点感情」+ 音色账本显示女声剩余缺口 0.177 全在**引擎**。
SOLA_v2pro 微调权重已出炉（e10/e20/e30/e40），GS 服务已 unload。
这一轮就是把「换引擎」这一个变量单独拎出来比。

## 公平性（照抄 engine_full_cmp.py 的教训）
两边都通过 **POST /api/jobs** 提交，由主服务跑**同一条流水线**：
分块 → 引擎合成 → ASR 回检 → PBA → 慢速修复 → EQ → PROSO → 呼吸 →
响度闭环 → peakfit。只有 "谁来念" 不同。

· 同一条文案：full_00b6a45ad6aa.txt（女声案例 00b6a45ad6aa 的全篇）
· 同一个音色名：EL_female（→ GS 侧 _gs_group_of 映射到 SOLA 组）
· 同一个 case_id：00b6a45ad6aa（生产口径 = 验收口径）
· 同一个 seed：42

用法（服务器）：
    venv/bin/python /mnt/tts/diag/cmp_sola47.py --dry-run
    venv/bin/python /mnt/tts/diag/cmp_sola47.py
    venv/bin/python /mnt/tts/diag/cmp_sola47.py --engines gptsovits   # 只跑一条
"""
import argparse
import glob
import os
import subprocess
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from tools.eval5 import login, submit_job, wait_job, wait_dir   # noqa: E402

OUT_DIR = "/mnt/tts/diag"
GSOUT = "/mnt/tts/projects/gpt-sovits/out"

TEXT = os.path.join(GSOUT, "full_00b6a45ad6aa.txt")
VOICE = "EL_female"
CID = "00b6a45ad6aa"


def find_out_dir(title):
    """产物目录 = /mnt/tts/output/<title>，但**重名会被加后缀** `-N`
    （服务端 _unique_dir）。第一次跑失败也会留下一个空目录占名，
    glob 无后缀目录会抓空 —— 第47轮实测踩过：GS 首跑 500 留下了空目录，
    成品实际落在 `sola47_f_gptsovits-2/`。
    所以这里按 `<title>*` 全列出来，挑**里面 mp3 最多且最新**的那个。
    """
    cands = sorted(glob.glob(os.path.join("/mnt/tts/output", title + "*")))
    best, best_key = None, None
    for d in cands:
        fs = glob.glob(os.path.join(d, "*.mp3"))
        if not fs:
            continue
        key = (len(fs), max(os.path.getmtime(f) for f in fs))
        if best_key is None or key > best_key:
            best, best_key = d, key
    return best


def concat_mp3s(title, dst):
    """把产物目录里的 mp3 按序拼成单条 wav（32k 单声道）。"""
    d = find_out_dir(title)
    if not d:
        return None
    print("  产物目录:", d)
    fs = sorted(glob.glob(os.path.join(d, "*.mp3")))
    if not fs:
        return None
    lst = os.path.join(OUT_DIR, "_lst_%s.txt" % title.replace("/", "_"))
    with open(lst, "w") as fh:
        for f in fs:
            fh.write("file '%s'\n" % f)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", lst,
                    "-ac", "1", "-ar", "32000", dst], check=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="gptsovits,omnivoice")
    ap.add_argument("--out-prefix", default="sola47")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--voice", default=VOICE,
                    help="音色名。GS 必须用 3~10s 参考（SOLA_gs10），"
                         "omnivoice 用现役 EL_female")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    engines = [e.strip() for e in a.engines.split(",") if e.strip()]
    txt = open(TEXT, encoding="utf-8").read().strip()
    login()
    made = []
    for eng in engines:
        title = "%s_f_%s" % (a.out_prefix, eng)
        dst = os.path.join(OUT_DIR, "%s_f_%s.wav" % (a.out_prefix, eng))
        if a.dry_run:
            print("会提交 引擎=%-12s 文案=f(%d字) 音色=%s case=%s -> %s"
                  % (eng, len(txt), a.voice, CID, dst))
            continue
        if os.path.exists(dst):
            print("跳过(已存在)", dst)
            made.append(dst)
            continue
        print("=== 引擎=%-12s 字数=%d ===" % (eng, len(txt)), flush=True)
        j = submit_job(txt, a.voice, title, seed=a.seed, engine=eng, case_id=CID)
        jid = j.get("id")
        print("  job_id:", jid, flush=True)
        st = wait_job(jid, timeout=7200)
        print("  作业状态:", st.get("status"), flush=True)
        if str(st.get("status")) not in ("done", "completed"):
            print("  ❌ 失败，跳过", st.get("error") or "")
            continue
        wait_dir(title, timeout=1800)
        got = concat_mp3s(title, dst)
        if got:
            made.append(got)
            print("  ->", got, flush=True)
    print("DONE，产出 %d 条" % len(made))
    for m in made:
        print("   ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
