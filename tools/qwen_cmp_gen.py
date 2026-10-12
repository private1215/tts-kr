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

OUT_DIR = "/mnt/tts/diag"


def main():
    """用法: qwen_cmp_gen.py <文本文件> <音色,逗号分隔> <标题前缀>"""
    argv = sys.argv[1:]
    text_path = argv[0] if argv else \
        "/mnt/tts/projects/gpt-sovits/out/text_sola_00b6a45a.txt"
    voices = (argv[1].split(",") if len(argv) > 1
              else ["SOLA_female", "EL_female"])
    prefix = argv[2] if len(argv) > 2 else "gs-cmp"
    txt = open(text_path, encoding="utf-8").read().strip()
    print("文本 %d 字  音色 %s" % (len(txt), voices), flush=True)
    login()
    for v in voices:
        title = "%s-%s" % (prefix, v)
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
        # ⚠️ 文件名必须带 prefix：男女两条长文案会各跑一次，
        #    只按音色命名会把女声结果覆盖掉（2026-10-12 长文案对照）。
        out = os.path.join(OUT_DIR, "qwen_%s_%s.wav" % (prefix, v))
        if os.path.exists(out):
            print("  跳过(已存在)", out, flush=True)
            continue
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                        "-f", "concat", "-safe", "0", "-i", lst,
                        "-ac", "1", "-ar", "32000", out], check=True)
        print("  ->", out, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
