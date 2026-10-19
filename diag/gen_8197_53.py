# -*- coding: utf-8 -*-
"""gen_8197_53.py —— 用**现役 v3.1 生产链**跑第二条男声文案，做泛化验证素材。

## 为什么
f 链（窄带 EQ + 清理）的全部结论目前只建立在**一条**文案 d172b8c4efb3 上。
项目铁律：**片段口径与全篇口径结论可以相反**，且新链路必须过第二条文案的泛化。
记忆里旧清理链也是这么验的（8197d9ac8a35 泛化也过）。

## 做法
走 `/api/jobs` 用 **EL_male**（现役 omnivoice）重跑案例 8197d9ac8a35 全文，
产出 44.1k wav 落到 /mnt/tts/diag/，供后续跑 g/f 链对照。

🔴 必须用 eval5 的 login/submit_job（直接 POST /api/jobs 会 422）。

用法: venv/bin/python -u /mnt/tts/diag/gen_8197_53.py
"""
import glob
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")

from eval5 import login, submit_job, wait_job  # noqa: E402

CASE = "8197d9ac8a35"
TXT = "/mnt/tts/data/cases/%s/full.txt" % CASE
OUT = "/mnt/tts/diag/m8197.wav"
OUT_DIRS = "/mnt/tts/data/output"


def find_out(title):
    """🔴 产物目录会被加 -N 后缀，且要**优先精确匹配**
    （glob(title+"*") 会让 gs48m_post 匹配到 gs48m_post2）。"""
    for cand in (os.path.join(OUT_DIRS, title),):
        if os.path.isdir(cand):
            return cand
    hits = sorted(glob.glob(os.path.join(OUT_DIRS, title + "*")))
    return hits[0] if hits else None


def main():
    text = open(TXT, encoding="utf-8").read().strip()
    print("文案 %d 字（%s）" % (len(text), CASE), flush=True)
    login()
    title = "m8197_53"
    j = submit_job(text, "EL_male", title, case_id=CASE)
    jid = j["id"] if isinstance(j, dict) else j
    print("job =", jid, flush=True)
    for _ in range(60):
        r = wait_job(jid, timeout=30)
        st = r.get("status") if isinstance(r, dict) else None
        if st in ("done", "finished", "completed", "success"):
            print("完成", flush=True)
            break
        if st in ("error", "failed"):
            print("失败：", r, flush=True)
            return 1
        time.sleep(5)
    d = find_out(title)
    if not d:
        print("❌ 找不到产物目录（title=%s）" % title)
        return 1
    print("产物目录：", d, flush=True)
    wavs = sorted(glob.glob(os.path.join(d, "*.wav")),
                  key=os.path.getmtime)
    if not wavs:
        print("❌ 目录里没有 wav：", os.listdir(d)[:10])
        return 1
    src = wavs[-1]
    shutil.copy2(src, OUT)
    dur = subprocess.run(["ffprobe", "-v", "error",
                          "-show_entries", "format=duration",
                          "-of", "csv=p=0", OUT],
                         capture_output=True, text=True).stdout.strip()
    sr = subprocess.run(["ffprobe", "-v", "error",
                         "-select_streams", "a:0",
                         "-show_entries", "stream=sample_rate",
                         "-of", "csv=p=0", OUT],
                        capture_output=True, text=True).stdout.strip()
    print("-> %s  %ss @ %sHz（源 %s）" % (OUT, dur, sr, src), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
