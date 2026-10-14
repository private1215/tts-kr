# -*- coding: utf-8 -*-
"""提交「线上完整链-Qwen」长稿任务到生产 /api/jobs。

用法：
    python tools/longsubmit_qwen.py --case d172b8c4efb3 --voice EL_male --title 长稿多方案-M1
    python tools/longsubmit_qwen.py --job <jid> --poll      # 只轮询已有任务
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from tools.eval5 import login, submit_job, wait_job, wait_dir, _get  # noqa: E402

CASES = "/mnt/tts/data/cases"
STATE = "/mnt/tts/diag/longsubmit-20261013"


def queue_state():
    """返回当前排队/运行中的任务概况，用于生产优先判断。"""
    st = json.loads(_get("/api/jobs?scope=all&page_size=300").decode("utf-8"))
    act = [j for j in st.get("items", [])
           if j.get("status") in ("queued", "running", "pending")]
    return st.get("total", 0), act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="")
    ap.add_argument("--voice", default="")
    ap.add_argument("--engine", default="qwen")
    ap.add_argument("--title", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--job", default="", help="只轮询已有任务")
    ap.add_argument("--poll", action="store_true")
    a = ap.parse_args()

    login()
    total, act = queue_state()
    print("提交前：队列总数=%d  活动中=%d" % (total, len(act)), flush=True)
    for j in act:
        print("   活动任务 %s owner=%s %s" % (j["id"], j.get("owner"),
                                            (j.get("title") or "")[:40]),
              flush=True)
    if act:
        print("!! 有员工活动任务，按规则让位，不提交", flush=True)
        return 2

    if a.poll and a.job:
        jid = a.job
    else:
        txt = open(os.path.join(CASES, a.case, "full.txt"), encoding="utf-8").read()
        print("文案 %d 字 sha256=%s" % (len(txt),
              hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]), flush=True)
        form = {
            "title": a.title, "text": txt, "mode": "clone", "voice": a.voice,
            "language": "Korean", "speed": "1.0", "engine": a.engine,
            "case_id": a.case, "seed": str(a.seed),
        }
    if not (a.poll and a.job):
        from urllib.parse import urlencode
        import tools.eval5 as E
        r = E._post("/api/jobs", form)
        jid = r.get("id")
        print("已提交 job=%s title=%s" % (jid, a.title), flush=True)

    t0 = time.time()
    rec = {"job": jid, "title": a.title, "case": a.case, "voice": a.voice,
           "engine": a.engine, "case_id": a.case, "seed": a.seed,
           "submitted_at": t0}
    last = None
    TERM = ("done", "completed", "success", "finished", "error", "failed")
    while time.time() - t0 < 10800:
        try:
            st = json.loads(_get("/api/jobs/" + jid).decode("utf-8"))
        except Exception:
            time.sleep(20)
            continue
        s = str(st.get("status", ""))
        el = round(time.time() - t0, 1)
        if (s, st.get("stage")) != last:
            print("  [%6.1fs] status=%s stage=%s progress=%s"
                  % (el, s, st.get("stage"), st.get("progress")), flush=True)
            last = (s, st.get("stage"))
        if s in TERM:
            rec["status"] = s
            rec["elapsed_sec"] = el
            rec["final"] = {k: st.get(k) for k in
                            ("status", "stage", "engine", "error",
                             "fallback_from", "fallback_reason",
                             "outputs", "stats", "dir")}
            break
        time.sleep(20)
    if "status" not in rec:
        rec["status"] = "timeout"

    if rec.get("status") in ("done", "completed"):
        fs = wait_dir(a.title, timeout=1800)
        rec["files"] = fs
        print("产物 %d 个" % len(fs), flush=True)
        for f in fs:
            print("   ", f, flush=True)
    json.dump(rec, open(os.path.join(STATE, jid + ".json"), "w"),
              ensure_ascii=False, indent=1)
    print("OK elapsed=%.1fs" % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
