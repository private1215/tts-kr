# -*- coding: utf-8 -*-
"""三引擎同文案对照：**一律走线上生产链路**产出，再拿去评分。

🔴 为什么必须有这个脚本（2026-10-12 方法论缺陷修正）
----------------------------------------------------
上一轮长文案对照里，三引擎的"出身"不一样：
  · Qwen 的音频 = 线上成品（走完整后处理链）
  · GS / Cosy 的音频 = CLI 裸推理（gs_batch_cmp.py / cosy_full.py 直接写盘）
于是韵律那一列根本不是公平对比 —— 相当于让 Cosy 和 GS 裸奔上场。
`eval5.submit_job` 的 docstring 里早就写了这条，但对照脚本没遵守。

修正：**所有引擎都通过 /api/jobs 提交**，由主服务跑同一条流水线
（分块 → 引擎合成 → ASR 回检 → PBA → 慢速修复 → EQ → PROSO → 呼吸 →
 响度闭环 → peakfit）。只有"谁来念"不同，其它完全相同。

🔴 case_id 必须带
-----------------
带上它，服务端的全部目标值（峰值 / 响度 / 停顿 / 语速 / PROSO）按**这条案例自身**
取，而不是同性别案例库中位数。验收本来就是跟这条案例比，
所以「生产口径 = 验收口径」。`--case ''` 可关掉做对照。

用法
----
    python tools/engine_full_cmp.py                     # 三引擎 × 两条长文案
    python tools/engine_full_cmp.py --engines qwen,gptsovits
    python tools/engine_full_cmp.py --dry-run
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

# (标签, 文案路径, 线上音色名, 对标案例 id)
GROUPS = [
    ("f", os.path.join(GSOUT, "full_00b6a45ad6aa.txt"),
     "EL_female", "00b6a45ad6aa"),
    ("m", os.path.join(GSOUT, "full_1a6c20a60481.txt"),
     "EL_male", "1a6c20a60481"),
]

DEFAULT_ENGINES = ["qwen", "cosyvoice3", "gptsovits"]


def concat_mp3s(title, dst):
    """把产物目录里的 mp3 按序拼成单条 wav（44.1k 单声道）。"""
    d = os.path.join("/mnt/tts/output", title)
    fs = sorted(glob.glob(os.path.join(d, "*.mp3")))
    if not fs:
        return None
    lst = "/tmp/list_%s.txt" % title.replace("/", "_")
    with open(lst, "w") as fh:
        for f in fs:
            fh.write("file '%s'\n" % f)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", lst,
                    "-ac", "1", "-ar", "32000", dst], check=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default=",".join(DEFAULT_ENGINES))
    ap.add_argument("--out-prefix", default="fullp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--case", default="1",
                    help="1=带上对标案例 id（生产口径=验收口径）；0=不带")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    engines = [e.strip() for e in a.engines.split(",") if e.strip()]
    login()
    made = []
    for tag, tpath, voice, cid in GROUPS:
        txt = open(tpath, encoding="utf-8").read().strip()
        for eng in engines:
            title = "%s_%s_%s" % (a.out_prefix, tag, eng)
            dst = os.path.join(OUT_DIR, "%s_%s_%s.wav"
                               % (a.out_prefix, tag, eng))
            if a.dry_run:
                print("  会提交 引擎=%-12s 文案=%s(%d字) 音色=%s -> %s"
                      % (eng, tag, len(txt), voice, dst))
                continue
            if os.path.exists(dst):
                print("  跳过(已存在)", dst)
                made.append(dst)
                continue
            print("=== 引擎=%-12s 文案=%s 字数=%d ==="
                  % (eng, tag, len(txt)), flush=True)
            j = submit_job(txt, voice, title, seed=a.seed,
                           engine=eng,
                           case_id=cid if a.case == "1" else "")
            jid = j.get("id")
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
    if not a.dry_run:
        print("DONE，产出 %d 条" % len(made))
        for m in made:
            print("   ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
