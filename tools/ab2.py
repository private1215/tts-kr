#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两阶段还原实验：案例对标（阶段一）+ 未见文案迁移（阶段二）。

为什么分两阶段
--------------
阶段一回答「同一条案例、同一份文案，本项目能不能生成出接近案例的成品」；
阶段二回答「固定同一个参考音色，换成没参与过的文案还成不成」。
把阶段一的分数直接当成阶段二的泛化能力，是本项目此前最容易犯的错。

阶段一（--stage 1）
  - text = 案例原文；voice = 该说话人的克隆音色
  - 同时跑 **带 case_id** 和 **不带 case_id** 两组：
      case_id 会把该案例自身的后处理目标（峰值/响度/停顿/语速/PROSO）注入链路，
      所以「带 case_id」是案例条件对标成绩，**不能**当成默认生产或泛化能力。
  - 多个随机种子：单次生成波动约 1~2 分（历史记录 Qwen 同案例不同 seed 差 0.95），
    只看一次或只挑最高分会误判。

阶段二（--stage 2）
  - 完全复用阶段一的 voice/参考音频/引擎/参数，**只换 text**
  - 默认**不带 case_id**（对新文案套用别的案例目标是错配）
  - 文案必须是没参与参考音频文字稿、训练、调参、bench_case 拟合的文本
  - 多条文案 × 多个 seed，报告均值与离散，不只报最好一次

判定口径（与 12 项验收清单一致）
------------------------------
- **完整性/内容是闸门**：ASR 空转写、CER 超阈、时长异常 → 直接判失败，
  任何声学高分都不能抵消。
- fidelity 只在**同文案、同长度**下可比；阶段二不同文案之间只用 CER/时长/
  语速做描述性对比，不做整段排名。
- 物理指标分数不是听感还原百分比；最终仍以人工盲听为准。

用法（服务器 qwen 环境）：
  venv/bin/python tools/ab2.py --stage 1 --case 6cdd25bc75f2 --voice EL_male \
      --seeds 42,2026,7 --prefix r26s1
  venv/bin/python tools/ab2.py --stage 2 --voice EL_male \
      --texts-json /mnt/tts/diag/unseen.json --seeds 42,2026 --prefix r26s2
"""
import os
import re
import sys
import json
import glob
import time
import argparse
import urllib.request
import urllib.parse
import http.cookiejar
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

BASE = os.environ.get("TTS_BASE", "http://127.0.0.1:8000")
USER = os.environ.get("TTS_USER", "admin")
PASS = os.environ.get("TTS_PASS", "qwer1234")
CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")
OUT_DIR = os.environ.get("TTS_OUT_DIR", "/mnt/tts/output")
WORK = os.environ.get("TTS_AB2_WORK", "/mnt/tts/diag/ab2")

_opener = None


def _op():
    global _opener
    if _opener is None:
        cj = http.cookiejar.CookieJar()
        _opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cj))
    return _opener


def login():
    data = urllib.parse.urlencode({"username": USER, "password": PASS}).encode()
    _op().open(urllib.request.Request(BASE + "/api/login", data=data),
               timeout=30)


def _get(path, timeout=60):
    return _op().open(urllib.request.Request(BASE + path),
                      timeout=timeout).read()


def _post(path, form, timeout=180):
    data = urllib.parse.urlencode(form).encode()
    r = _op().open(urllib.request.Request(BASE + path, data=data),
                   timeout=timeout)
    return json.loads(r.read().decode("utf-8"))


def case_text(cid):
    d = os.path.join(CASES_DIR, cid)
    full = os.path.join(d, "full.txt")
    if os.path.exists(full):
        return open(full, encoding="utf-8").read()
    segs = sorted(glob.glob(os.path.join(d, "seg*.txt")),
                  key=lambda p: int(re.findall(r"seg(\d+)", p)[0]))
    return "\n".join(open(p, encoding="utf-8").read().strip() for p in segs)


def submit(text, voice, title, seed, case_id="", extra=None):
    """走生产 API 提交。固定 speed=1.0 / Korean / clone，其余参数仅在显式
    传入时才设置，避免悄悄改变线上默认行为。"""
    form = {"title": title, "text": text, "mode": "clone", "voice": voice,
            "language": "Korean", "speed": "1.0"}
    if seed is not None:
        form["seed"] = str(int(seed))
    if case_id:
        form["case_id"] = case_id
    for k in ("max_chars", "batch_size", "temperature",
              "subtalker_temperature"):
        if extra and extra.get(k) is not None:
            form[k] = str(extra[k])
    if extra and extra.get("enhance") is not None:
        form["enhance"] = "true" if extra["enhance"] else "false"
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
        if s in ("done", "completed", "success", "finished", "error", "failed"):
            return st
        time.sleep(poll)
    return {"status": "timeout"}


def wait_dir(title, timeout=7200, poll=15):
    d = os.path.join(OUT_DIR, title)
    t0, stable, last = time.time(), 0, -1
    while time.time() - t0 < timeout:
        fs = glob.glob(os.path.join(d, "*.mp3")) + \
             glob.glob(os.path.join(d, "*.wav"))
        if fs and len(fs) == last:
            stable += poll
            if stable >= 60:
                return sorted(fs)
        else:
            stable = 0
        last = len(fs)
        time.sleep(poll)
    return sorted(glob.glob(os.path.join(d, "*.mp3")))


def to_wav(files, dst):
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


def asr_cer(wav, ref_text):
    """内容回检。空转写 = 验收失败（不能当作 skipped 放行）。"""
    from app.asr_check import transcribe, cer as _cer
    hyp = transcribe(wav, "ko")
    if not hyp:
        return 1.0, "", True
    return _cer(ref_text, hyp), hyp, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--case", default="", help="阶段一：案例 id（可逗号分隔多个）")
    ap.add_argument("--voice", required=True, help="克隆音色名（决定引擎）")
    ap.add_argument("--seeds", default="42,2026,7")
    ap.add_argument("--prefix", default="ab2")
    ap.add_argument("--case-id", default="both",
                    help="阶段一：both=带与不带各跑一遍；1=只带；0=只不带")
    ap.add_argument("--texts-json", default="",
                    help='阶段二：JSON 数组 [{"id":"t1","text":"..."}]')
    ap.add_argument("--max-chars", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--no-fidelity", action="store_true",
                    help="跳过 fidelity（跨文案不可比时用）")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    login()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    extra = {"max_chars": a.max_chars, "batch_size": a.batch_size}
    rows = []

    if a.stage == 1:
        if not a.case:
            sys.exit("阶段一需要 --case")
        conds = []
        if a.case_id == "both":
            conds = [1, 0]
        else:
            conds = [int(a.case_id)]
        for cid in [c.strip() for c in a.case.split(",") if c.strip()]:
            txt = case_text(cid)
            case_wav = os.path.join(CASES_DIR, cid, "merged.wav")
            for ci in conds:
                for sd in seeds:
                    title = f"{a.prefix}-{cid[:6]}-ci{ci}-s{sd}"
                    print(f"[阶段一] {title}  {len(txt)}字", flush=True)
                    job = submit(txt, a.voice, title, sd,
                                 case_id=(cid if ci else ""), extra=extra)
                    jid = job.get("id") or job.get("job_id")
                    st = wait_job(jid)
                    if str(st.get("status")) in ("error", "failed", "timeout"):
                        print("  !! 任务失败", st.get("status"))
                        continue
                    fs = wait_dir(title)
                    if not fs:
                        print("  !! 无产物")
                        continue
                    wav = to_wav(fs, os.path.join(WORK, title + ".wav"))
                    cer, hyp, empty = asr_cer(wav, txt)
                    rec = dict(stage=1, case=cid, case_id=bool(ci), seed=sd,
                               title=title, chars=len(txt), cer=cer,
                               asr_empty=empty, engine=job.get("engine"))
                    if not a.no_fidelity and os.path.exists(case_wav):
                        from tools.fidelity import measure, score
                        cm, pm = measure(case_wav, "CASE"), measure(wav, "PROD")
                        s, devs = score(cm, pm)
                        rec.update(fidelity=s, dur_case=cm["dur"],
                                   dur_prod=pm["dur"], devs=devs)
                    rows.append(rec)
                    print(f"  CER={cer:.4f}{'（空转写=失败）' if empty else ''}"
                          f"  fidelity={rec.get('fidelity')}", flush=True)
    else:
        if not a.texts_json:
            sys.exit("阶段二需要 --texts-json")
        texts = json.load(open(a.texts_json, encoding="utf-8"))
        print("⚠️ 确认：这些文案未参与参考音频文字稿 / 训练 / 调参 / bench_case "
              "拟合，否则不是 out-of-sample。", flush=True)
        for t in texts:
            tid, txt = t.get("id"), t.get("text", "")
            if not txt.strip():
                continue
            for sd in seeds:
                title = f"{a.prefix}-{tid}-s{sd}"
                print(f"[阶段二] {title}  {len(txt)}字", flush=True)
                job = submit(txt, a.voice, title, sd, case_id="", extra=extra)
                jid = job.get("id") or job.get("job_id")
                st = wait_job(jid)
                if str(st.get("status")) in ("error", "failed", "timeout"):
                    print("  !! 任务失败", st.get("status"))
                    continue
                fs = wait_dir(title)
                if not fs:
                    print("  !! 无产物")
                    continue
                wav = to_wav(fs, os.path.join(WORK, title + ".wav"))
                cer, hyp, empty = asr_cer(wav, txt)
                # 阶段二只用内容/语速做描述性对比，不跨文案排 fidelity
                from tools.fidelity import measure
                pm = measure(wav, "PROD")
                rows.append(dict(stage=2, text_id=tid, case_id=False, seed=sd,
                                 title=title, chars=len(txt), cer=cer,
                                 asr_empty=empty, dur=pm["dur"],
                                 cps=round(len(txt) / max(pm["dur"], 1e-6), 2),
                                 engine=job.get("engine")))
                print(f"  CER={cer:.4f}{'（空转写=失败）' if empty else ''}"
                      f"  dur={pm['dur']:.1f}s cps={rows[-1]['cps']}", flush=True)

    if not rows:
        print("没有有效结果")
        return
    # 汇总：按 (case/条件) 或 (文案) 聚合多种子
    if a.stage == 1:
        groups = {}
        for r in rows:
            k = (r["case"], r["case_id"])
            groups.setdefault(k, []).append(r)
        print("\n" + "=" * 74)
        print(f"{'案例':<14}{'case_id':>9}{'种子数':>7}{'CER中位':>9}"
              f"{'fidelity均值':>13}{'fidelity极差':>13}{'失败':>6}")
        for k, rs in groups.items():
            cs = sorted(r["cer"] for r in rs)
            fs_ = [r.get("fidelity") for r in rs if r.get("fidelity") is not None]
            fail = sum(1 for r in rs if r["asr_empty"])
            fm = sum(fs_) / len(fs_) if fs_ else None
            fr = (max(fs_) - min(fs_)) if len(fs_) > 1 else 0.0
            print(f"{k[0][:12]:<14}{str(k[1]):>9}{len(rs):>7}"
                  f"{cs[len(cs)//2]:>9.4f}"
                  f"{(('%.2f' % fm) if fm is not None else '-'):>13}"
                  f"{(('%.2f' % fr) if fs_ else '-'):>13}{fail:>6}")
    else:
        groups = {}
        for r in rows:
            groups.setdefault(r["text_id"], []).append(r)
        print("\n" + "=" * 74)
        print(f"{'文案':<14}{'种子数':>7}{'CER中位':>9}{'时长均值':>10}"
              f"{'语速均值':>10}{'失败':>6}")
        for k, rs in groups.items():
            cs = sorted(r["cer"] for r in rs)
            d = sum(r["dur"] for r in rs) / len(rs)
            cps = sum(r["cps"] for r in rs) / len(rs)
            fail = sum(1 for r in rs if r["asr_empty"])
            print(f"{k:<14}{len(rs):>7}{cs[len(cs)//2]:>9.4f}{d:>10.1f}"
                  f"{cps:>10.2f}{fail:>6}")

    out = a.json or os.path.join(WORK, f"{a.prefix}-stage{a.stage}.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(f"\n✅ {out}")


if __name__ == "__main__":
    main()
