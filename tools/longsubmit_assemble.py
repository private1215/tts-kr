# -*- coding: utf-8 -*-
"""把各方案的原始产物拼成「一条候选一个文件」的盲标音频，并落盘台账。

🔴 拼接口径：只做解码与连接（concat），**不做任何后处理**
（陷阱 #16：离线 finalize 的 alimiter 会把 E 打掉 18%，离线试算的
 E/A 不可与平台口径比）。与 app 里的 eval5.to_wav 保持一致。

产出：/mnt/tts/diag/longsubmit-20261013/deliver/*.wav (24k 单声道 PCM16)
台账：/mnt/tts/diag/longsubmit-20261013/deliver_manifest.json
"""
import glob
import hashlib
import json
import os
import subprocess
import sys

STATE = "/mnt/tts/diag/longsubmit-20261013"
DELIVER = os.path.join(STATE, "deliver")
CASES = "/mnt/tts/data/cases"


def sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ffprobe_dur(p):
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", p],
                       capture_output=True, text=True).stdout.strip()
    return round(float(o), 3) if o else None


def join(files, dst):
    """解码 + 拼接 -> 24k 单声道 PCM16 wav。"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if len(files) == 1:
        src = ["-i", files[0]]
    else:
        lst = dst + ".list.txt"
        with open(lst, "w") as w:
            for f in files:
                w.write("file '%s'\n" % f)
        src = ["-f", "concat", "-safe", "0", "-i", lst]
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y"] + src +
                   ["-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", dst],
                   check=True)
    return dst


def sources(spec):
    if spec.get("wav"):
        return [spec["wav"]]
    return sorted(glob.glob(os.path.join(spec["dir"], "*.mp3")))


def main():
    plan = json.load(open(os.path.join(STATE, "blind_plan.json"),
                          encoding="utf-8"))
    os.makedirs(DELIVER, exist_ok=True)
    out = {"cases": {}}
    for cid, info in plan["cases"].items():
        recs = []
        for name, spec in info["candidates"].items():
            fs = sources(spec)
            if not fs:
                print("!! 无产物", name, spec, flush=True)
                recs.append({"blind_name": name, "status": "no_artifact",
                             "spec": spec})
                continue
            dst = os.path.join(DELIVER, name + ".wav")
            join(fs, dst)
            recs.append({
                "blind_name": name,
                "status": "ok",
                "file": dst,
                "sha256": sha256f(dst),
                "dur_sec": ffprobe_dur(dst),
                "size_bytes": os.path.getsize(dst),
                "source_files": fs,
                "source_file_sha256": [sha256f(f) for f in fs],
                "source_count": len(fs),
            })
            print("  %s -> %s  %.1fs" % (name, dst,
                                         ffprobe_dur(dst) or -1), flush=True)
        # 基准（案例原声 EL）
        base = os.path.join(CASES, cid, "merged.wav")
        bd = os.path.join(DELIVER, info["ref_name"] + ".wav")
        join([base], bd)
        out["cases"][cid] = {
            "ref": {"blind_name": info["ref_name"], "file": bd,
                    "source": base, "sha256": sha256f(bd),
                    "dur_sec": ffprobe_dur(bd)},
            "candidates": recs,
        }
        print("  %s(基准) -> %.1fs" % (info["ref_name"], ffprobe_dur(bd)),
              flush=True)
    json.dump(out, open(os.path.join(STATE, "deliver_manifest.json"), "w"),
              ensure_ascii=False, indent=1)
    print("-> deliver_manifest.json")


if __name__ == "__main__":
    main()
