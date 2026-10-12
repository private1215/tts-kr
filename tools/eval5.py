# -*- coding: utf-8 -*-
"""五案例线上验收驱动（站规：每轮改完必须跑）。

两种模式
--------
1) `--prefix r20`：只**重评分**已有的线上产物（/mnt/tts/output/<prefix>-<短id>/）。
   代码没动、只想复核分数时用这个，几分钟出结果。
2) `--submit`：走**生产 API** 重新生成（POST /api/jobs），下载产物后再评分。
   这是真正的验收路径，不绕过业务代码。

为什么必须走 API 而不是直接调模型：绕过 server.py 就等于没验到
响度归一 / EQ / 停顿 reshuffle / PROSO 这条整链，历史上在这上面吃过亏。

🔴 必须在 qwen 环境跑（librosa 1.0.0），eval 用 http://127.0.0.1:8000
（服务器访问自己的 Tailscale IP 不通）。
"""
import os
import re
import sys
import json
import time
import glob
import argparse
import subprocess
import urllib.request
import urllib.parse
import http.cookiejar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fidelity import measure, score, WEIGHTS  # noqa: E402

BASE = os.environ.get("TTS_BASE", "http://127.0.0.1:8000")
USER = os.environ.get("TTS_USER", "admin")
PASS = os.environ.get("TTS_PASS", "qwer1234")
CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")
OUT_DIR = os.environ.get("TTS_OUT_DIR", "/mnt/tts/output")
# 🔴 默认目录**不能**是 /tmp：2026-10-07 重启把 /tmp 清空，上一版的
#    fidelity3.py / eval5r20.py / submit5.py 全没了，93.52% 基线一度无法复现。
#    中间产物（拼好的 wav、评分 json）也落持久盘，重启后还能接着对账。
WORK = os.environ.get("TTS_EVAL_WORK", "/mnt/tts/diag/eval5work")

# 五案例固定清单（id, 短名, 性别, 生成用音色）
FIVE = [
    ("1a6c20a60481", "1a6c20", "male", "EL_male", "标杆男"),
    ("00b6a45ad6aa", "00b6a4", "female", "EL_female", "标杆女"),
    ("ba24c0925311", "ba24c0", "male", "EL_male", "대한광통신"),
    ("8197d9ac8a35", "8197d9", "male", "EL_male", "세미파이브"),
    ("6cdd25bc75f2", "6cdd25", "male", "EL_male", "리노공업y"),
]

_opener = None


def _op():
    global _opener
    if _opener is None:
        cj = http.cookiejar.CookieJar()
        _opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cj))
    return _opener


def login():
    """Form 登录拿 cookie（API 鉴权是 cookie，不是 Bearer）。"""
    data = urllib.parse.urlencode({"username": USER, "password": PASS}).encode()
    _op().open(urllib.request.Request(BASE + "/api/login", data=data), timeout=30)
    return True


def _get(path, timeout=60):
    return _op().open(urllib.request.Request(BASE + path), timeout=timeout).read()


def _post(path, form, timeout=120):
    data = urllib.parse.urlencode(form).encode()
    r = _op().open(urllib.request.Request(BASE + path, data=data), timeout=timeout)
    return json.loads(r.read().decode("utf-8"))


def case_text(cid):
    """取案例全文：优先 full.txt，否则按 segN.txt 顺序拼。"""
    d = os.path.join(CASES_DIR, cid)
    full = os.path.join(d, "full.txt")
    if os.path.exists(full):
        return open(full, encoding="utf-8").read()
    segs = sorted(glob.glob(os.path.join(d, "seg*.txt")),
                  key=lambda p: int(re.findall(r"seg(\d+)", p)[0]))
    return "\n".join(open(p, encoding="utf-8").read().strip() for p in segs)


def submit_job(text, voice, title, case_id="", seed=None, engine=None):
    """提交生成任务。⚠️ 不能用 multipart（-F text=@file），那是文件上传，
    服务端 Form(str) 会报 "Input should be a valid string"。

    `title` 同时就是产物目录名（/mnt/tts/output/<title>/），所以带短名
    方便后面按目录取。mode 必须显式给 clone，否则走 custom 不会用参考音频。

    `case_id`：对标案例。带上它，服务端全部目标值（峰值/响度/停顿/语速/PROSO）
    就按**这条案例自身**取，而不是同性别案例库的中位数。验收时本来就是跟
    这条案例比，所以带上是「生产口径=验收口径」；`--case 0` 可关掉做对照。

    `engine`：显式指定 qwen / cosyvoice3。
    🔴 做引擎对照时**必须走生产服务提交，不能拿 CLI 直出比**：
    CLI 出来的音频没经过后处理链（PBA 停顿对齐 / EQ / proso / 呼吸 / 响度闭环），
    而 Qwen 那份是走完整链的，两者比等于让 Cosy 裸奔上场，结论不可信。
    走生产提交才是同一条流水线上的公平对照。
    """
    form = {
        "title": title,
        "text": text,
        "mode": "clone",
        "voice": voice,
        "language": "Korean",
        "speed": "1.0",
    }
    if case_id:
        form["case_id"] = case_id
    if engine:
        form["engine"] = engine
    # 拒绝采样实验用：不传就是服务端默认 42
    if seed is not None:
        form["seed"] = str(int(seed))
    return _post("/api/jobs", form)


def wait_job(jid, timeout=7200, poll=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            st = json.loads(_get("/api/jobs/" + jid).decode("utf-8"))
        except Exception:
            time.sleep(poll)
            continue
        s = str(st.get("status", ""))
        if s in ("done", "completed", "success", "finished"):
            return st
        if s in ("error", "failed"):
            return st
        time.sleep(poll)
    return {"status": "timeout"}


def wait_dir(title, timeout=7200, poll=15):
    """等产物目录出现且 mp3 不再增多（服务端本机直接看文件系统，
    比走 HTTP 下载简单，也避免 /api/download 的鉴权细节）。"""
    d = os.path.join(OUT_DIR, title)
    t0 = time.time()
    stable, last = 0, -1
    while time.time() - t0 < timeout:
        fs = glob.glob(os.path.join(d, "*.mp3"))
        if fs and len(fs) == last:
            stable += poll
            if stable >= 60:
                return sorted(fs)
        else:
            stable = 0
        last = len(fs)
        time.sleep(poll)
    return sorted(glob.glob(os.path.join(d, "*.mp3")))


def download(job, dest_dir):
    """下载某任务的全部成品到 dest_dir，返回文件列表。"""
    os.makedirs(dest_dir, exist_ok=True)
    files = []
    outs = job.get("outputs") or job.get("files") or []
    if not outs:
        d = job.get("dir") or job.get("out_dir") or ""
        for f in (job.get("files_list") or []):
            outs.append({"dir": d, "file": f})
    for o in outs:
        d = o.get("dir") or job.get("dir") or ""
        f = o.get("file") or o.get("name")
        if not f:
            continue
        url = f"{BASE}/api/download/{d}/{urllib.parse.quote(f)}"
        dst = os.path.join(dest_dir, os.path.basename(f))
        try:
            with _op().open(urllib.request.Request(url), timeout=600) as r, \
                    open(dst, "wb") as w:
                w.write(r.read())
            files.append(dst)
        except Exception as e:
            print("  !! 下载失败", f, repr(e)[:120])
    return files


def to_wav(files, dst):
    """把成品 mp3 拼成 24k 单声道 wav 用于测量。

    ⚠️ 只做解码与拼接，**不做任何后处理**（陷阱 #16：离线 finalize.py 的
    alimiter 会把 E 打掉 18%，离线试算的 E/A 不可与平台口径比）。
    """
    if len(files) == 1:
        src = ["-i", files[0]]
    else:
        lst = dst + ".txt"
        with open(lst, "w") as w:
            for f in sorted(files):
                w.write("file '%s'\n" % f)
        src = ["-f", "concat", "-safe", "0", "-i", lst]
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y"] + src +
                   ["-ac", "1", "-ar", "24000", dst], check=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="", help="已有产物前缀，如 r20")
    ap.add_argument("--submit", action="store_true", help="重新走线上生成")
    ap.add_argument("--only", default="", help="只跑部分短名，逗号分隔")
    ap.add_argument("--voice", default="", help="覆盖 FIVE 里的音色（A/B 另一个引擎时用："
                    "引擎由音色的 engine 字段决定，所以要跑 CosyVoice3 必须传一个 "
                    "engine=cosyvoice3 的临时音色，不能直接改 EL_male）")
    ap.add_argument("--cer", action="store_true", help="额外跑 ASR 回检（慢）")
    ap.add_argument("--case", type=int, default=1,
                    help="1=提交时带对标案例（生产目标=该案例自身值，默认）；"
                         "0=不带（目标=同性别库中位数，做对照用）")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    if a.submit:
        login()

    rows, all_devs = [], {}
    for cid, short, gender, voice, label in FIVE:
        if a.only and short not in a.only.split(","):
            continue
        if a.voice:
            voice = a.voice
        print(f"\n=== {label} ({short}) ===", flush=True)
        prod_dir = os.path.join(OUT_DIR, (a.prefix + "-" + short) if a.prefix
                                else ("eval5-" + short))
        if a.submit:
            title = (a.prefix or "eval5") + "-" + short
            txt = case_text(cid)
            print(f"  提交 {len(txt)} 字 -> {title}", flush=True)
            job = submit_job(txt, voice, title,
                             case_id=(cid if a.case else ""))
            jid = job.get("id") or job.get("job_id")
            st = wait_job(jid)
            print("  status", st.get("status"), flush=True)
            wait_dir(title)
            prod_dir = os.path.join(OUT_DIR, title)

        mp3s = sorted(sum([glob.glob(os.path.join(prod_dir, p))
                           for p in ("*.mp3", "*.wav")], []))
        if not mp3s:
            print("  !! 没有产物，跳过")
            continue
        wav = to_wav(mp3s, os.path.join(WORK, short + ".wav"))
        case_wav = os.path.join(CASES_DIR, cid, "merged.wav")
        cm = measure(case_wav, "CASE")
        pm = measure(wav, "PROD")
        s, devs = score(cm, pm)
        all_devs[short] = devs
        cer = None
        if a.cer:
            from app.asr_check import transcribe, cer as _cer
            hyp = transcribe(wav, "ko")
            cer = _cer(case_text(cid), hyp)
        rows.append((label, short, gender, s, cm["dur"], pm["dur"],
                     cm["lufs"], pm["lufs"], cm["peak"], pm["peak"], cer))
        print(f"  {label:<12} {s}%   时长 {pm['dur']:.1f}/{cm['dur']:.1f}  "
              f"LUFS {pm['lufs']}/{cm['lufs']}  峰值 {pm['peak']}/{cm['peak']}",
              flush=True)

    if not rows:
        return
    print("\n" + "=" * 78)
    print(f"{'案例':<12}{'性别':>6}{'还原度':>9}{'时长(成品/案例)':>18}"
          f"{'LUFS':>14}{'峰值':>14}{'CER':>8}")
    for label, short, gender, s, cd, pd_, cl, pl, cp, pp, cer in rows:
        print(f"{label:<12}{gender:>6}{s:>9.2f}{('%0.1f/%0.1f' % (pd_, cd)):>18}"
              f"{('%0.1f/%0.1f' % (pl, cl)):>14}{('%0.1f/%0.1f' % (pp, cp)):>14}"
              f"{(('%.4f' % cer) if cer is not None else '-'):>8}")
    mean = sum(r[3] for r in rows) / len(rows)
    print(f"{'均值':<12}{'':>6}{mean:>9.2f}")
    bench = [r[3] for r in rows if r[1] in ("1a6c20", "00b6a4")]
    yoh = [r[3] for r in rows if r[1] in ("ba24c0", "8197d9", "6cdd25")]
    if bench and yoh:
        bm = sum(bench) / len(bench)
        ym = sum(yoh) / len(yoh)
        print(f"  标杆 in-sample {bm:.2f} / yohan out-of-sample {ym:.2f} "
              f"/ 过拟合量 {bm - ym:+.2f}")

    print("\n逐项偏差（%，带符号均值用于分辨「系统性缺陷」vs「噪声」）")
    keys = ["A", "B", "E", "D", "F", "sil", "C", "dur", "lufs", "peak", "tone"]
    print(f"{'维度':<8}{'权重':>6}" + "".join(f"{s:>10}" for s in all_devs))
    for k in keys:
        vals = [all_devs[s].get(k) for s in all_devs]
        print(f"{k:<8}{WEIGHTS[k]:>6.2f}" +
              "".join(f"{(('%.1f' % v) if v is not None else '-'):>10}"
                      for v in vals))

    if a.json:
        json.dump({"rows": rows, "devs": all_devs, "mean": mean},
                  open(a.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
