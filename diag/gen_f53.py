# -*- coding: utf-8 -*-
"""gen_f53.py —— 重生成女声 L3 现役链路产物，拿**权威当前基线**。

## 为什么必须重跑
记忆里女声基线自相矛盾，两条都标着"现役"：
  记录A：아모텍 3415字 → click **0.61×EL** / 中频 0.87 / 杂音 0.22 / 音色 0.8067
  记录B：中频 **1.01（已对齐，不需 EQ）**
  刚测 l3-f-omni-2（另一条文案）→ click **1.59×EL** / 中频 1.01 / 杂音 0.80

click 到底是 0.61×（已比 EL 好，按铁律不能动）还是 1.59×（比 EL 差 59%，该修）
**是完全相反的两种决策**。且 v3.2 之后没重生成过女声产物。
→ 用同一条文案（female_full.txt，3429 字）重出一遍，一次测全。

用法: venv/bin/python -u /mnt/tts/diag/gen_f53.py
"""
import glob
import os
import sys
import time

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from tools.eval5 import login, submit_job, wait_job  # noqa: E402

TITLE = "l3f53"
OUT_DIRS = ["/mnt/tts/output"]


def find_out(title):
    """🔴 产物目录会被加 `-N` 后缀，取 mtime 最新的。"""
    hits = sorted(glob.glob(os.path.join(OUT_DIRS[0], title + "*")),
                  key=os.path.getmtime)
    return hits[-1] if hits else None


def main():
    login()
    txt = open("/mnt/tts/diag/female_full.txt", encoding="utf-8").read().strip()
    print("文案 %d 字" % len(txt), flush=True)
    j = submit_job(txt, "EL_female", TITLE)
    print("job=%s" % j.get("id"), flush=True)
    s = wait_job(j.get("id"), timeout=7200)
    print("status=%s engine=%s" % (s.get("status"), s.get("engine")), flush=True)
    print("stats=%s" % s.get("stats"), flush=True)
    for _ in range(60):
        d = find_out(TITLE)
        if d:
            fs = sorted(glob.glob(os.path.join(d, "*.mp3")))
            if fs:
                print("产物：%s" % fs[-1], flush=True)
                return 0
        time.sleep(5)
    print("❌ 没找到产物目录 %s" % TITLE, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
