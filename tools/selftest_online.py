# -*- coding: utf-8 -*-
"""线上正式链路一键自测：**用案例的文案走员工那条路，再跟案例原声比**。

为什么要有这个脚本
------------------
2026-10-13 之前，"我们生成的 vs ElevenLabs 案例"这件事是手工做的：
手敲 curl 提交、手拼成品、手写 spec，跑完的结论只留在聊天里，
下次想复现又得重新拼一遍（历史上 fidelity3.py / eval5r20.py 就因为只放
在 /tmp 被服务器清空丢过一次，r20 基线一度无法复现）。

它做的事（每一步都与员工实际操作一致，没有评测专用便利）：
    1. 登录（账号密码从环境变量取，不落盘）
    2. 把目标案例的 seg*.txt 拼成完整稿
    3. POST /api/jobs，参数与 UI 提交完全一致（mode=clone / speed 1.0 /
       emotion standard / seed 42，不传 engine、不传 case_id）
    4. 等任务完成
    5. 把线上落盘的 mp3 原样 concat + 解码成 44.1k wav（不做二次处理）
    6. 生成 engine_score_all 的 spec

用法
----
    # 真实口径（推荐做验收）：被测案例 **不要** 用参考音频的来源案例
    python tools/selftest_online.py --case d172b8c4efb3 --voice EL_male --run

    # 开卷口径（乐观上限，只用来做对照，不能当验收数字）
    python tools/selftest_online.py --case 1a6c20a60481 --voice EL_male --open

    # 只提交不等（任务留在服务器队列里，稍后自己查）
    python tools/selftest_online.py --case ce1a3fda551a --voice EL_female --no-wait

环境变量：
    TTS_SELFTEST_USER / TTS_SELFTEST_PASS   自测账号（默认 selftest）
    TTS_BASE                                默认 /mnt/tts

⚠️ 口径铁律（改代码前先读）
--------------------------
同一个说话人的参考音频（EL_male / EL_female）是从 **bench_case** 那条案例里
切出来的。拿那条案例自己的文案去测 = **开卷考试**，测的是"记忆"不是"还原"。
验收必须用别的案例（同一个人、但不是参考来源）。两者本次实测只差 0.004，
但结论性质完全不同 —— 开卷分数不能拿去回答"员工听不听得出"。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

BASE = os.environ.get("TTS_BASE", "").strip() or "/mnt/tts"
API = "http://127.0.0.1:8000"
CASES_DIR = os.path.join(BASE, "data", "cases")
OUTPUT_DIR = os.path.join(BASE, "output")
TOOLS = os.path.dirname(os.path.abspath(__file__))


def _natkey(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def case_text(cid):
    """案例全部片段文案拼成一篇完整稿。"""
    d = os.path.join(CASES_DIR, cid)
    segs = sorted([f for f in os.listdir(d)
                   if f.startswith("seg") and f.endswith(".txt")], key=_natkey)
    parts = []
    for f in segs:
        t = open(os.path.join(d, f), encoding="utf-8").read().strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def login(cj):
    u = os.environ.get("TTS_SELFTEST_USER", "selftest")
    p = os.environ.get("TTS_SELFTEST_PASS", "")
    if not p:
        sys.exit("❌ 缺 TTS_SELFTEST_PASS（自测账号密码不写进代码）")
    if os.path.exists(cj):
        os.unlink(cj)
    r = subprocess.run(["curl", "-s", "-c", cj, "-X", "POST", API + "/api/login",
                        "-d", "username=%s&password=%s" % (u, p)],
                       capture_output=True, text=True)
    if '"username"' not in r.stdout:
        sys.exit("❌ 登录失败：%s" % r.stdout[:200])


def submit(cj, title, text, voice):
    tf = "/tmp/_selftest_upload.txt"
    with open(tf, "w", encoding="utf-8") as f:
        f.write(text)
    r = subprocess.run(
        ["curl", "-s", "-b", cj, "-X", "POST", API + "/api/jobs",
         "-F", "title=" + title,
         "-F", "text=<" + tf,
         "-F", "mode=clone", "-F", "language=Korean",
         "-F", "voice=" + voice, "-F", "speed=1.0",
         "-F", "emotion=standard", "-F", "seed=42"],
        capture_output=True, text=True)
    try:
        return json.loads(r.stdout)["id"]
    except Exception:
        sys.exit("❌ 提交失败：%s" % r.stdout[:300])


def job(cj, jid):
    r = subprocess.run(["curl", "-s", "-b", cj, API + "/api/jobs/" + jid],
                       capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


def dur(p):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", p], capture_output=True, text=True).stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="被测案例 id")
    ap.add_argument("--voice", required=True, help="线上音色名（如 EL_male）")
    ap.add_argument("--title", default="")
    ap.add_argument("--out-dir", default="/tmp/selftest")
    ap.add_argument("--open", action="store_true",
                    help="标注为开卷口径（参考音频取自本案例），仅作对照")
    ap.add_argument("--no-wait", action="store_true", help="只提交不等")
    ap.add_argument("--run", action="store_true",
                    help="spec 生成后立刻跑 engine_score_all")
    args = ap.parse_args()

    cid, voice = args.case, args.voice
    os.makedirs(args.out_dir, exist_ok=True)
    tag = "open" if args.open else "real"
    cj = "/tmp/_cj_selftest"

    text = case_text(cid)
    if not text.strip():
        sys.exit("❌ 案例 %s 没有文案" % cid)
    tf = os.path.join(args.out_dir, "case_%s.txt" % cid)
    with open(tf, "w", encoding="utf-8") as f:
        f.write(text)
    print("案例 %s：%d 字" % (cid, len(text)))

    login(cj)
    title = args.title or ("自测-%s-%s-%s" % (tag, voice, cid))
    jid = submit(cj, title, text, voice)
    print("已提交：%s（%s）" % (jid, title))
    if args.no_wait:
        return

    for i in range(200):
        j = job(cj, jid)
        st = j.get("status")
        if st in ("done", "error"):
            break
        if i % 4 == 0:
            print("  [%s] %s %s" % (time.strftime("%H:%M:%S"), st, j.get("chunk") or ""))
        time.sleep(30)
    j = job(cj, jid)
    if j.get("status") != "done":
        sys.exit("❌ 任务未成功：%s %s" % (j.get("status"), j.get("error")))
    st = j.get("stats") or {}
    print("完成：%d 字 / %.1fs / RTF %.2f / 异常段 %d" %
          (st.get("chars", 0), st.get("audio_seconds", 0), st.get("rtf", 0),
           st.get("abnormal_segments", 0)))
    for f in j.get("files", []):
        print("   %s %.1fs cer=%.4f" % (f["name"], f["seconds"], f.get("cer") or 0))

    # 成品拼回整条
    files = [os.path.join(OUTPUT_DIR, f["path"]) for f in j.get("files", [])]
    files = [p for p in files if os.path.exists(p)]
    files.sort(key=lambda p: os.path.basename(p))
    wav = os.path.join(args.out_dir, "%s_%s.wav" % (tag, voice))
    lst = os.path.join(args.out_dir, "list_%s_%s.txt" % (tag, voice))
    with open(lst, "w", encoding="utf-8") as f:
        for p in files:
            f.write("file '%s'\n" % p.replace("'", "'\\''"))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", lst, "-ac", "1", "-ar", "44100", wav], check=True)
    ref = os.path.join(CASES_DIR, cid, "merged.wav")
    print("成品 %.1fs（%d 段）  参照 %.1fs" % (dur(wav), len(files), dur(ref)))

    is_male = (voice == "EL_male") or ("male" in voice.lower())
    spec = [
        {"name": "案例原声(参照)", "path": ref, "is_male": is_male,
         "ref": True, "text_file": tf},
        {"name": "线上正式链路(%s)" % voice, "path": wav, "is_male": is_male,
         "ref": False, "text_file": tf},
    ]
    sp = os.path.join(TOOLS, "score_selftest_%s_%s.json" % (tag, voice))
    json.dump(spec, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("SPEC:", sp)
    if args.run:
        subprocess.run([os.path.join(os.path.dirname(TOOLS), "venv", "bin", "python"),
                        "-u", os.path.join(TOOLS, "engine_score_all.py"),
                        "--spec", sp])


if __name__ == "__main__":
    main()
