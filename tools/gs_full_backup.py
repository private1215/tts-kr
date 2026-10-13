# -*- coding: utf-8 -*-
"""用 GS 备份权重（_bak_e40）生成整篇候选，供员工盲听比较。

背景
----
2026-10-13 早先判定「GS 不可用」，依据是 pick_weight 对 SOLA/yohan 解析为空。
复核发现：权重其实存在，只是放在 SoVITS_weights_v2ProPlus/_bak_e40/ 子目录，
而 pick_weight 只扫根目录（glob 非递归）。各文件 sha256 不同，是不同 checkpoint。
本脚本通过**独立目录 + 独立端口(8201)**的 GS 实例使用这些权重，
不改动生产 GS 服务、不改动生产权重目录。

⚠️ 结论性质
----------
这些权重来自备份目录，**训练收敛与数据标注未经验证**。产出只能作为
实验候选参与盲听比较，不得直接视为可交付方案。

用法
----
    python -u tools/gs_full_backup.py
"""
import json
import os
import subprocess
import sys
import time

OUT = "/mnt/tts/diag/gs_backup-20261013"
CASES = "/mnt/tts/data/cases"
GSOUT = "/mnt/tts/projects/gpt-sovits/out"
API = "http://127.0.0.1:8201"
os.makedirs(OUT, exist_ok=True)

# (输出名, 案例id, 性别, spk, 参考wav, 参考txt)
JOBS = [
    ("G1_m", "d172b8c4efb3", "male", "yohan",
     os.path.join(GSOUT, "prompt_yohan_1a6c20a6.wav"),
     os.path.join(GSOUT, "prompt_yohan_1a6c20a6.txt")),
    ("G2_f", "ce1a3fda551a", "female", "SOLA",
     os.path.join(GSOUT, "prompt_sola_00b6a45a.wav"),
     os.path.join(GSOUT, "prompt_sola_00b6a45a.txt")),
]


def case_text(cid):
    d = os.path.join(CASES, cid)
    segs = sorted([f for f in os.listdir(d)
                   if f.startswith("seg") and f.endswith(".txt")])
    return "\n\n".join(open(os.path.join(d, f), encoding="utf-8").read().strip()
                       for f in segs if open(os.path.join(d, f),
                                             encoding="utf-8").read().strip())


def main():
    report = []
    for name, cid, gender, spk, rw, rt in JOBS:
        print("\n==== %s %s spk=%s ====" % (name, gender, spk), flush=True)
        txt = case_text(cid)
        dst = os.path.join(OUT, name + ".wav")
        t0 = time.time()
        ref_text = open(rt, encoding="utf-8").read().strip()
        r = subprocess.run(
            ["curl", "-s", "--max-time", "3600", API + "/synth",
             "-F", "text=%s" % txt, "-F", "spk=%s" % spk,
             "-F", "ref_wav=%s" % rw, "-F", "ref_text=%s" % ref_text,
             "-F", "seed=42", "-F", "seg=50", "-o", dst],
            capture_output=True)
        el = time.time() - t0
        sz = os.path.getsize(dst) if os.path.exists(dst) else 0
        dur = None
        if sz:
            try:
                dur = float(subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries",
                     "format=duration", "-of", "csv=p=0", dst],
                    capture_output=True, text=True).stdout.strip())
            except Exception:
                pass
        ok = bool(sz and dur and dur > 30)
        print("-> %s %d bytes %.1fs 用时 %.1fs ok=%s" % (dst, sz, dur or 0,
                                                          el, ok), flush=True)
        if r.returncode != 0:
            print("curl rc=%s stderr=%s" % (r.returncode,
                                            r.stderr.decode()[:300]))
        report.append({"name": name, "case": cid, "gender": gender, "spk": spk,
                       "chars": len(txt), "bytes": sz,
                       "audio_seconds": dur, "elapsed_s": round(el, 1),
                       "status": "ok" if ok else "failed",
                       "weight_dir": "SoVITS_weights_v2ProPlus/_bak_e40",
                       "api": API, "out": dst})
        # 卸载释放显存，避免两个说话人同时驻留
        subprocess.run(["curl", "-s", "-X", "POST", API + "/unload"],
                       capture_output=True)
    json.dump(report, open(os.path.join(OUT, "gs_backup_report.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nDONE", json.dumps(report, ensure_ascii=False)[:600])
    return 0


def POST_API():
    return API + "/synth"


if __name__ == "__main__":
    raise SystemExit(main())
