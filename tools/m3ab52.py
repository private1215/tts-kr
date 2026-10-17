# -*- coding: utf-8 -*-
"""M3/M4 离线 A/B：同一裸输出分别走不同后处理组合。

组合
----
A  现役完整链（自适应激进 EQ + 停顿清洗）
B  温和 EQ + 停顿清洗
C  自适应激进 EQ + 无停顿清洗（半极简）
D  温和 EQ + 无停顿清洗（半极简 + 温和 EQ = 推荐方案）

用法（服务器，zheng 可直接写 projects）：
    cd /mnt/tts/projects/qwen3-tts
    /mnt/tts/projects/qwen3-tts/venv/bin/python -u \
        /tmp/m3ab52.py /tmp/voxprod_st10_0.wav 785 /mnt/tts/diag/m3ab52
"""
import json
import os
import subprocess
import sys
import tempfile

FIX_CHAIN = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"
EQ_MILD = ("highpass=f=110:poles=2,"
           "equalizer=f=90:t=q:w=2.0:g=3.0,"
           "equalizer=f=350:t=q:w=1.5:g=1.5,"
           "equalizer=f=1400:t=q:w=1.5:g=1.5,"
           "highshelf=f=4000:g=-2,lowpass=f=14000")
TARGET_CPS = 8.37


def run(in_wav, out_wav, gender, chars, no_clean=False, no_adaptive=False, eq=None):
    cmd = [sys.executable, FIX_CHAIN, in_wav, out_wav,
           "--gender", gender, "--target-cps", str(TARGET_CPS),
           "--chars", str(chars)]
    if no_clean:
        cmd.append("--no-clean")
    if no_adaptive:
        cmd.append("--no-adaptive-hf")
    if eq:
        cmd += ["--eq", eq]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print("FAIL %s: %s" % (out_wav, r.stderr[-300:]), flush=True)
        return False
    print("OK %s" % out_wav, flush=True)
    return True


def main():
    in_wav = sys.argv[1]
    chars = int(sys.argv[2])
    outdir = sys.argv[3] if len(sys.argv) > 3 else "/mnt/tts/diag/m3ab52"
    os.makedirs(outdir, exist_ok=True)

    variants = [
        ("A_baseline", False, False, None),
        ("B_mild_eq", False, True, EQ_MILD),
        ("C_no_clean", True, False, None),
        ("D_mild_no_clean", True, True, EQ_MILD),
    ]
    report = []
    for name, nc, na, eq in variants:
        dst = os.path.join(outdir, "%s.wav" % name)
        ok = run(in_wav, dst, "m", chars, nc, na, eq)
        report.append({"name": name, "out": dst, "ok": ok,
                       "no_clean": nc, "no_adaptive": na,
                       "eq": eq})
    json.dump(report, open(os.path.join(outdir, "ab_report.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nDONE", json.dumps(report, ensure_ascii=False)[:600])


if __name__ == "__main__":
    sys.exit(main())
