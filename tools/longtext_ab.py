# -*- coding: utf-8 -*-
"""长文本 A/B：Qwen3-TTS（当前生产） vs GPT-SoVITS（候选引擎）。

短文本（304 字）上 GS 微调版说话人相似度 0.9362 明显高于 Qwen 的 0.8441，
但那只是 40 秒素材。真实业务是**单篇 3000 字左右、成片 3~8 分钟**，
而长文本发散/吞字正是 GPT-SoVITS 的已知弱项。这一关不过，相似度再高也不能换引擎。

两边都用同一份文本 /mnt/tts/diag/gs_long.txt（2902 字，取自另一个案例 산일전기，
对 GS 微调模型是完全没见过的文本，且与线上文案同分布），产出拼接成一条 wav，
交给 longtext_score.py 统一打分。

用法：
    python tools/longtext_ab.py EL_female gs-long-qwen
"""
import os
import sys
import glob
import subprocess

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
os.chdir("/mnt/tts/projects/qwen3-tts")

from tools.eval5 import login, submit_job, wait_job, wait_dir  # noqa: E402

TEXT = open("/mnt/tts/diag/gs_long.txt", encoding="utf-8").read().strip()


def main():
    voice = sys.argv[1] if len(sys.argv) > 1 else "EL_female"
    title = sys.argv[2] if len(sys.argv) > 2 else "gs-long-qwen"
    print("文本字数:", len(TEXT), flush=True)
    print("音色:", voice, " 产物目录:", title, flush=True)

    login()
    job = submit_job(TEXT, voice, title, seed=42)
    jid = job.get("id")
    print("job:", jid, flush=True)
    st = wait_job(jid, timeout=7200)
    print("job status:", st.get("status"), flush=True)
    if st.get("status") == "error":
        raise SystemExit("生成失败: %s" % str(st.get("error"))[:300])
    # 🔴 同名 title 被用过时服务端会自动加 -2 / -3 后缀，所以不能直接 wait_dir(title)，
    #    否则等的是上一次的空目录，一直卡到超时。这里取最新且非空的那个。
    base = os.environ.get("TTS_OUT_DIR", "/mnt/tts/output")
    cands = sorted(glob.glob(os.path.join(base, title + "*")))
    d = None
    for c in cands:
        if os.path.isdir(c) and glob.glob(os.path.join(c, "*.mp3")):
            d = c
    if d is None:
        d = os.path.join(base, title)
    print("产物目录:", d, flush=True)
    fs = wait_dir(os.path.basename(d), timeout=7200)
    print("分段数:", len(fs), flush=True)
    for f in sorted(fs):
        print("   ", os.path.basename(f), flush=True)

    # 拼成一条 wav，方便和 GS 那条对齐口径地做 ASR / 时长比较
    lst = "/mnt/tts/diag/%s.list.txt" % title
    with open(lst, "w") as fh:
        for f in sorted(fs):
            fh.write("file '%s'\n" % f)
    wav = "/mnt/tts/diag/%s.wav" % title
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", lst, "-ac", "1", "-ar", "24000", wav],
        check=True,
    )
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", wav],
        capture_output=True, text=True,
    )
    print("拼接后时长:", r.stdout.strip(), "s ->", wav, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
