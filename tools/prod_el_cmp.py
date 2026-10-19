# -*- coding: utf-8 -*-
"""prod_el_cmp.py —— 用 EL 案例的同款文案，走**生产链**生成，供与 EL 对比。

## 为什么

此前所有优化（tailclean53b/53c/中频EQ）都只在 `/mnt/tts/diag/` 诊断产物上做，
**生产链 app/ 未接入**。要判断优化是否值得接生产，必须拿
「生产链真实产出」与 EL 同文案对照，量化差距。

⚠️ 必须走 /api/jobs（不是 CLI 裸生成）：prod_engine_ab.py 踩过的坑——
   裸生成没经过后处理链，比赢了不能信。

## 用法

    python tools/prod_el_cmp.py [voice] [title]

产物落在 /mnt/tts/output/<title>*/（服务端可能自动加 -N 后缀）。
"""
import glob
import os
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
os.chdir("/mnt/tts/projects/qwen3-tts")

from tools.eval5 import login, submit_job, wait_job  # noqa: E402

TXT = "/mnt/tts/data/cases/d172b8c4efb3/full.txt"


def main():
    voice = sys.argv[1] if len(sys.argv) > 1 else "EL_male"
    title = sys.argv[2] if len(sys.argv) > 2 else "elcmp-male"
    text = open(TXT, encoding="utf-8").read().strip()
    print("文案 %s  %d 字  voice=%s  title=%s"
          % (TXT, len(text), voice, title), flush=True)

    login()
    job = submit_job(text, voice, title)
    jid = job.get("id")
    print("job:", jid, flush=True)
    st = wait_job(jid, timeout=7200)
    print("status:", st.get("status"), flush=True)
    if st.get("status") == "error":
        raise SystemExit("生成失败: %s" % str(st.get("error"))[:300])

    base = os.environ.get("TTS_OUT_DIR", "/mnt/tts/output")
    d = None
    for c in sorted(glob.glob(os.path.join(base, title + "*"))):
        if os.path.isdir(c) and (glob.glob(os.path.join(c, "*.mp3"))
                                 or glob.glob(os.path.join(c, "*.wav"))):
            d = c
    print("OUT_DIR=%s" % d, flush=True)
    for f in sorted(glob.glob(os.path.join(d or base, "*"))):
        print("  %s  %d bytes" % (f, os.path.getsize(f)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
