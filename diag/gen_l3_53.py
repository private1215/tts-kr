# -*- coding: utf-8 -*-
"""gen_l3_53.py —— 中频 EQ 接生产后的 **L3 线上验收**（≥3 分钟长篇）。

## 背景
`deploy/patch_mideq53b.py` 已把保采样率窄带中频 EQ（+2dB）接进生产链
（只 omnivoice + 男声，`TTS_MID_EQ_MALE_DB` 默认 2）。
离线验证已过（pipe_1019g 第一条 + pipe_8197_g 第二条泛化）：
    第一条 中频 0.86→**1.01**、click 0.11×、质心 1.01、杂音 0.85、音色 0.9331→**0.9357**
    第二条 中频 →**1.02**、click 0.02×、质心 1.02、杂音 0.95
但**只有 L3 全篇线上产物**才能验收（项目铁律：片段口径与全篇口径可相反）。

## 做法
走 /api/jobs 用 EL_male 重跑案例 d172b8c4efb3（4594 字，L3 ≈ 555s），
产物落到 /mnt/tts/diag/，之后跑 measure53 + spkcmp53 与 EL 对照。

🔴 必须用 eval5 的 login/submit_job（直接 POST /api/jobs 会 422）。
🔴 click / 杂音必须**压完 mp3 后**测（交付就是 mp3）。

用法: venv/bin/python -u /mnt/tts/diag/gen_l3_53.py
"""
import glob
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")

from eval5 import login, submit_job, wait_job  # noqa: E402

CASE = "d172b8c4efb3"
TXT = "/mnt/tts/data/cases/%s/full.txt" % CASE
OUT = "/mnt/tts/diag/l3_53.mp3"
OUT_DIRS = "/mnt/tts/output"


def find_out(title):
    """🔴 产物目录会被加 `-N` 后缀。**精确匹配会抓到上一次的旧目录**
    （2026-10-19 实测：第二次跑 l3_mideq53，产物在 l3_mideq53-2/，
    而 l3_mideq53/ 还留着第一次的旧文件 → 精确匹配拿错了）。
    → 改成：把所有匹配按 mtime 排序，取**最新**的那个。"""
    hits = sorted(glob.glob(os.path.join(OUT_DIRS, title + "*")),
                  key=os.path.getmtime)
    return hits[-1] if hits else None


def main():
    text = open(TXT, encoding="utf-8").read().strip()
    print("文案 %d 字（%s，L3）" % (len(text), CASE), flush=True)
    login()
    title = "l3_mideq53"
    j = submit_job(text, "EL_male", title, case_id=CASE)
    jid = j["id"] if isinstance(j, dict) else j
    print("job =", jid, flush=True)
    t0 = time.time()
    for _ in range(120):
        r = wait_job(jid, timeout=30)
        st = r.get("status") if isinstance(r, dict) else None
        if st in ("done", "finished", "completed", "success"):
            print("完成（%.0fs）" % (time.time() - t0), flush=True)
            break
        if st in ("error", "failed"):
            print("失败：", r, flush=True)
            return 1
        time.sleep(10)
    d = find_out(title)
    if not d:
        print("❌ 找不到产物目录（title=%s）" % title)
        return 1
    print("产物目录：", d, flush=True)
    wavs = sorted(glob.glob(os.path.join(d, "*.mp3")) +
                  glob.glob(os.path.join(d, "*.wav")),
                  key=os.path.getmtime)
    if not wavs:
        print("❌ 目录里没有音频：", os.listdir(d)[:10])
        return 1
    shutil.copy2(wavs[-1], OUT)
    dur = subprocess.run(["ffprobe", "-v", "error",
                          "-show_entries", "format=duration",
                          "-of", "csv=p=0", OUT],
                         capture_output=True, text=True).stdout.strip()
    sr = subprocess.run(["ffprobe", "-v", "error",
                         "-select_streams", "a:0",
                         "-show_entries", "stream=sample_rate",
                         "-of", "csv=p=0", OUT],
                        capture_output=True, text=True).stdout.strip()
    print("-> %s  %ss @ %sHz（源 %s）" % (OUT, dur, sr, wavs[-1]),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
