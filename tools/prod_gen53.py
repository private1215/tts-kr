# -*- coding: utf-8 -*-
"""prod_gen53.py —— 用**任意案例文案**走生产链 /api/jobs 生成，供泛化验证。

## 为什么需要它
tailclean53b/53c/eqfix53/ltasmatch53 这些判据是在**单条文案**（d172b8c4efb3，
제주반도체 男声）上调出来的。接生产前必须证明：换一条文案、另一种内容，
清理判据依然安全（不削字、碎片下降、click 不变差），而不是只对那一条过拟合。

## 用法（服务端执行）
    python tools/prod_gen53.py <case_id> [voice] [title]
例：
    python tools/prod_gen53.py 8197d9ac8a35 EL_male gen2-male
    python tools/prod_gen53.py 8197d9ac8a35 EL_female gen2-female

⚠️ 必须走 /api/jobs（不能 CLI 裸生成，prod_engine_ab.py 踩过的坑）。
产物落在 /mnt/tts/output/<title>*/。
"""
import glob
import os
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
os.chdir("/mnt/tts/projects/qwen3-tts")

from tools.eval5 import login, submit_job, wait_job  # noqa: E402

CASE_DIR = "/mnt/tts/data/cases"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    case = sys.argv[1]
    voice = sys.argv[2] if len(sys.argv) > 2 else "EL_male"
    title = sys.argv[3] if len(sys.argv) > 3 else "gen53-" + case[:6]
    engine = os.environ.get("GEN_ENGINE", "") or None
    case_id = os.environ.get("GEN_CASE_ID", "")

    txt = os.path.join(CASE_DIR, case, "full.txt")
    if not os.path.exists(txt):
        raise SystemExit("文案不存在: %s" % txt)
    text = open(txt, encoding="utf-8").read().strip()
    print("文案 %s  %d 字  voice=%s  title=%s"
          % (txt, len(text), voice, title), flush=True)

    login()
    job = submit_job(text, voice, title, case_id=case_id, engine=engine)
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
