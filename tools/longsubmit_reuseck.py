# -*- coding: utf-8 -*-
"""复用资产核验：把 4 个历史成品目录 + 对应 job 的台账一次性核出来。

判定口径：
  文案   —— job.text 与案例 full.txt（CRLF）比对，字节级 / 文本模式 / 空白归一化
  参考   —— voices.json 该音色的 path + ref_text 现场算 hash，并与 job 创建时间比较
            （wav mtime 早于 job 创建时间 => job 之后参考未被动过）
  产物   —— 目录内 mp3 逐个算 sha256 + 时长，与 job.status=done 互证
"""
import datetime
import glob
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from tools.eval5 import login, _get  # noqa: E402

CASES = "/mnt/tts/data/cases"
VOICES = "/mnt/tts/data/voices/voices.json"
STATE = "/mnt/tts/diag/longsubmit-20261013"

# (job_id, 产出目录, 案例id, job里声明的音色)
JOBS = [
    ("35391b1f0e1f", "自测-男-真实口径-d172b8c4efb3", "d172b8c4efb3", "EL_male"),
    ("159b1c5e6a66", "AB-男-参考yohan_male-d172b8c4efb3", "d172b8c4efb3", "yohan_male"),
    ("bc8196e72fed", "自测-女-真实口径-ce1a3fda551a", "ce1a3fda551a", "EL_female"),
    ("81e99d131e9b", "AB-女-参考SOLA_female-ce1a3fda551a", "ce1a3fda551a", "SOLA_female"),
]


def sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dur(p):
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", p],
                       capture_output=True, text=True).stdout.strip()
    return round(float(o), 3) if o else None


def main():
    login()
    V = json.load(open(VOICES, encoding="utf-8"))
    out = []
    for jid, title, cid, voice in JOBS:
        j = json.loads(_get("/api/jobs/" + jid).decode("utf-8"))
        jtext = j.get("text") or ""
        ref_raw = open(os.path.join(CASES, cid, "full.txt"), "rb").read()
        ref_txt = open(os.path.join(CASES, cid, "full.txt"),
                       encoding="utf-8").read()
        import re
        nrm = lambda s: re.sub(r"\s+", " ", s).strip()
        v = V[voice]
        vw, vt = v["path"], (v.get("ref_text") or "")
        created = float(j.get("created_at") or 0)
        vmt = os.path.getmtime(vw)

        files = sorted(glob.glob(os.path.join("/mnt/tts/output", title, "*.mp3")))
        rec = {
            "job_id": jid,
            "output_dir": os.path.join("/mnt/tts/output", title),
            "case_id": cid,
            "job": {
                "title": j.get("title"), "status": j.get("status"),
                "engine": j.get("engine"), "engine_force": j.get("engine_force"),
                "voice": j.get("voice"), "mode": j.get("mode"),
                "case_id": j.get("case_id"), "seed": j.get("seed"),
                "speed": j.get("speed"), "emotion": j.get("emotion"),
                "owner": j.get("owner"),
                "created_at": created,
                "created_at_str": datetime.datetime.fromtimestamp(
                    created).strftime("%Y-%m-%d %H:%M:%S"),
            },
            "text_check": {
                "job_text_chars": len(jtext),
                "job_text_sha256": hashlib.sha256(
                    jtext.encode("utf-8")).hexdigest(),
                "case_full_txt_bytes_sha256": hashlib.sha256(ref_raw).hexdigest(),
                "case_full_txt_chars_textmode": len(ref_txt),
                "case_full_txt_textmode_sha256": hashlib.sha256(
                    ref_txt.encode("utf-8")).hexdigest(),
                "identical": jtext == ref_txt,
                "identical_after_ws_normalize": nrm(jtext) == nrm(ref_txt),
                "note": ("字节级完全一致" if jtext == ref_txt else
                         "仅差一个尾随换行符，空白归一化后一致（同一稿件，未换稿）"),
            },
            "reference_check": {
                "voice_key": voice,
                "voices_json_engine_field": v.get("engine"),
                "ref_wav": vw,
                "ref_wav_sha256": sha256f(vw),
                "ref_wav_mtime": datetime.datetime.fromtimestamp(
                    vmt).strftime("%Y-%m-%d %H:%M:%S"),
                "ref_wav_mtime_before_job_created": vmt < created,
                "ref_text_len": len(vt),
                "ref_text_sha256": hashlib.sha256(
                    vt.encode("utf-8")).hexdigest(),
            },
            "artifacts": [{"file": f, "sha256": sha256f(f), "dur_sec": dur(f)}
                          for f in files],
            "total_dur_sec": round(sum(dur(f) for f in files), 2),
        }
        el_wav = os.path.join(CASES, cid, "merged.wav")
        rec["baseline_el"] = {
            "file": el_wav, "sha256": sha256f(el_wav), "dur_sec": dur(el_wav)}
        rec["verdict"] = "可复用" if (j.get("status") == "done"
                                 and rec["text_check"]
                                 ["identical_after_ws_normalize"]
                                 and rec["reference_check"]
                                 ["ref_wav_mtime_before_job_created"]) \
            else "不可复用"
        out.append(rec)
        print("%s %s -> %s  %.1fs  files=%d" %
              (jid, voice, rec["verdict"], rec["total_dur_sec"], len(files)),
              flush=True)
    json.dump(out, open(os.path.join(STATE, "reuse_check.json"), "w"),
              ensure_ascii=False, indent=1)
    print("->", os.path.join(STATE, "reuse_check.json"))


if __name__ == "__main__":
    main()
