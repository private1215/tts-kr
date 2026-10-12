# -*- coding: utf-8 -*-
"""走**生产服务**的引擎对照：同一条流水线上的 Qwen vs CosyVoice3。

🔴 为什么不能拿 CLI 直出比（2026-10-12 踩到的坑）：
   GPT-SoVITS / CosyVoice3 用官方 CLI 跑出来的音频是**裸的** —— 没经过我们的
   后处理链（PBA 停顿对齐 / EQ 长时谱拟合 / proso 韵律移植 / 呼吸 / 响度闭环 /
   peakfit）。而线上 Qwen 那份是走完整链的。拿裸音频去比 = 让候选引擎裸奔上场，
   打赢了不能信，打输了也不知道是不是差在没做后处理。

   所以这个脚本一律**走 /api/jobs 提交**，让服务端按正常生产流程出片，
   两边经过完全相同的后处理，差异才只来自引擎本身。

用法：
    python tools/prod_engine_ab.py cosyvoice3 cosy-prod-long  long  42
    python tools/prod_engine_ab.py qwen       qwen-prod-long  long  42
"""
import glob
import os
import subprocess
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
os.chdir("/mnt/tts/projects/qwen3-tts")

from tools.eval5 import login, submit_job, wait_job, wait_dir  # noqa: E402

TEXTS = {
    "short": "/mnt/tts/diag/hh_short.txt",
    "long": "/mnt/tts/diag/gs_long.txt",
}


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "cosyvoice3"
    title = sys.argv[2] if len(sys.argv) > 2 else "prod-ab"
    which = sys.argv[3] if len(sys.argv) > 3 else "long"
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    voice = sys.argv[5] if len(sys.argv) > 5 else "EL_female"

    text = open(TEXTS[which], encoding="utf-8").read().strip()
    print("引擎=%s  音色=%s  %s文本 %d 字  seed=%d" % (engine, voice, which, len(text), seed),
          flush=True)

    login()
    job = submit_job(text, voice, title, seed=seed, engine=engine)
    jid = job.get("id")
    print("job:", jid, flush=True)
    st = wait_job(jid, timeout=7200)
    print("status:", st.get("status"), flush=True)
    if st.get("status") == "error":
        raise SystemExit("生成失败: %s" % str(st.get("error"))[:300])

    # 🔴 同名 title 服务端会自动加 -2/-3 后缀，直接 wait_dir(title) 会等在上一次
    #    的空目录直到超时。取最新且非空的那个。
    base = os.environ.get("TTS_OUT_DIR", "/mnt/tts/output")
    d = None
    for c in sorted(glob.glob(os.path.join(base, title + "*"))):
        if os.path.isdir(c) and glob.glob(os.path.join(c, "*.mp3")):
            d = c
    if d is None:
        d = os.path.join(base, title)
    print("产物目录:", d, flush=True)
    fs = wait_dir(os.path.basename(d), timeout=7200)
    print("分段数:", len(fs), flush=True)

    lst = "/mnt/tts/diag/%s.list.txt" % title
    with open(lst, "w") as fh:
        for f in sorted(fs):
            fh.write("file '%s'\n" % f)
    wav = "/mnt/tts/diag/%s.wav" % title
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", lst, "-ac", "1", "-ar", "24000", wav], check=True)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", wav], capture_output=True, text=True)
    print("-> %s  %ss" % (wav, r.stdout.strip()), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
