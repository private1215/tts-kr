# -*- coding: utf-8 -*-
"""l1_zerofill53 - 数字零填充的 L1 闸门（15~30s，改动有没有生效）

预期（S4 写在前）：
  ≥20ms 全零段：男 52 -> 0 / 女 44 -> 0（同素材离线实测 zerofix53b）
  时长：零变化；LUFS：不变；CER：不变差

服务端执行：
  /mnt/tts/projects/qwen3-tts/venv/bin/python /tmp/l1_zerofill53.py m
  /mnt/tts/projects/qwen3-tts/venv/bin/python /tmp/l1_zerofill53.py f
"""
import glob
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
os.chdir("/mnt/tts/projects/qwen3-tts")

from tools.eval5 import login, submit_job, wait_job  # noqa: E402
import numpy as np  # noqa: E402

SR = 16000
CASE = {"m": ("1a6c20a60481", "EL_male"), "f": ("00b6a45ad6aa", "EL_female")}
NCHARS = 250          # 250 字 / 8.27 字每秒 ≈ 30s（L1 区间上限）


def decode(path, sr=SR):
    out = tempfile.mktemp(suffix=".raw")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    "-ac", "1", "-ar", str(sr), "-c:a", "pcm_f32le",
                    "-f", "f32le", out], check=True)
    y = np.fromfile(out, dtype="<f4").astype(np.float32)
    os.remove(out)
    return y


def zero_runs(y, sr, min_ms=20.0):
    z = np.abs(y) < 1e-9
    d = np.diff(np.concatenate(([0], z.astype(np.int8), [0])))
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return [(a / sr, (b - a) / sr * 1000.0)
            for a, b in zip(s, e) if (b - a) / sr * 1000.0 >= min_ms]


def steep_drops(y, sr, ms=30.0, drop=25.0):
    w = int(ms / 1000.0 * sr)
    env = 10.0 * np.log10(np.convolve(y * y, np.ones(w) / w, mode="same") + 1e-20)
    return sum(1 for i in range(0, len(env) - 2 * w, w)
               if env[i:i + w].max() - env[i + w:i + 2 * w].max() >= drop)


def main():
    g = sys.argv[1] if len(sys.argv) > 1 else "m"
    case_id, voice = CASE[g]
    txtp = "/mnt/tts/data/cases/%s/full.txt" % case_id
    if os.path.exists(txtp):
        text = open(txtp, encoding="utf-8").read().strip()
    else:
        # 有些 case 只有分段文案（seg0.txt / seg1.txt ...），拼起来用
        parts = sorted(glob.glob("/mnt/tts/data/cases/%s/seg*.txt" % case_id))
        text = " ".join(open(p, encoding="utf-8").read().strip()
                        for p in parts)
    text = text[:NCHARS]
    title = "l1zf53-%s" % g
    print("case=%s voice=%s chars=%d" % (case_id, voice, len(text)), flush=True)

    login()
    job = submit_job(text, voice, title, case_id=case_id, engine="omnivoice")
    jid = job.get("id")
    print("job:", jid, flush=True)
    st = wait_job(jid, timeout=3600)
    print("status:", st.get("status"), flush=True)
    if st.get("status") == "error":
        raise SystemExit("生成失败: %s" % str(st.get("error"))[:300])

    d = None
    for c in sorted(glob.glob("/mnt/tts/output/%s*" % title)):
        if os.path.isdir(c) and glob.glob(os.path.join(c, "*.mp3")):
            d = c
    if not d:
        raise SystemExit("没找到产物目录")
    mp3 = sorted(glob.glob(os.path.join(d, "*.mp3")))[-1]
    print("MP3:", mp3, os.path.getsize(mp3), "bytes", flush=True)

    y = decode(mp3)
    db = 20 * np.log10(np.abs(y) + 1e-12)
    pause = db < -45
    q = np.percentile(db[pause], [5, 50]) if pause.sum() > 10 else [np.nan] * 2
    zr = zero_runs(y, SR)
    print("dur=%.1fs  zero>=20ms=%d  steep30ms=%d  p5=%.0f  med=%.0f  peak=%.1fdB"
          % (len(y) / SR, len(zr), steep_drops(y, SR), q[0], q[1], db.max()))
    print("PASS" if not zr else "FAIL(zero remain)  %s" % zr[:5])
    return 0


if __name__ == "__main__":
    sys.exit(main())
