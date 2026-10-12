# -*- coding: utf-8 -*-
"""生成 Qwen 侧的对照音频：与 GS 推理用**同一条文本**，便于公平比音色。

文本取 GS 推理时用的那条（测试区 60s 的 ASR 转写），保证两边一字不差。
"""
import sys
import os
import glob
import subprocess

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")

from tools.eval5 import login, submit_job, wait_job, wait_dir  # noqa: E402

TEXT = "/mnt/tts/projects/gpt-sovits/out/text_sola_00b6a45a.txt"
OUT_DIR = "/mnt/tts/diag"
VOICES = ["SOLA_female", "EL_female"]


def main():
    txt = open(TEXT, encoding="utf-8").read().strip()
    print("文本 %d 字" % len(txt), flush=True)
    login()
    for v in VOICES:
        title = "gs-cmp-" + v
        j = submit_job(txt, v, title, seed=42)
        jid = j.get("id")
        st = wait_job(jid, timeout=1800)
        print(v, st.get("status"), flush=True)
        base = "/mnt/tts/output"
        d = None
        for c in sorted(glob.glob(os.path.join(base, title + "*"))):
            if os.path.isdir(c) and glob.glob(os.path.join(c, "*.mp3")):
                d = c
        if d is None:
            print("  ❌ 没找到产物目录", flush=True)
            continue
        fs = wait_dir(os.path.basename(d), timeout=1800)
        lst = "/tmp/list_%s.txt" % v
        with open(lst, "w") as fh:
            for f in sorted(fs):
                fh.write("file '%s'\n" % f)
        out = os.path.join(OUT_DIR, "qwen_cmp_%s.wav" % v)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                        "-f", "concat", "-safe", "0", "-i", lst,
                        "-ac", "1", "-ar", "32000", out], check=True)
        print("  ->", out, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
