# -*- coding: utf-8 -*-
"""定位 0.953(裸推理) -> 0.844(线上成品) 这 0.109 丢在哪。

同一案例 1a6c20a60481、同一参照、同一文案，只有"走没走线上后处理"不同。
三项诊断：
  1. 两者互相比：如果互相似也很低 → 声学上确实被改了（后处理）；
     如果互相似很高 → 是 campplus/长度类测量偏差。
  2. 分段看漂移：是不是某一段崩了（线上分 chunk 生成）。
  3. 各自整条 vs 参照。
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = "/mnt/tts/projects/qwen3-tts"
sys.path.insert(0, os.path.join(ROOT, "tools"))
import spkchk  # noqa: E402

CAMPPLUS = os.environ.get(
    "TTS_CAMPPLUS", "/mnt/tts/models/Fun-CosyVoice3-0.5B/campplus.onnx")
_SESS = None


def sess():
    global _SESS
    if _SESS is None:
        import onnxruntime
        _SESS = onnxruntime.InferenceSession(
            CAMPPLUS, providers=["CPUExecutionProvider"])
    return _SESS


def wav16(path, ss=None, t=None):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if ss is not None:
        cmd += ["-ss", str(ss)]
    if t is not None:
        cmd += ["-t", str(t)]
    cmd += ["-i", path, "-ar", "16000", "-ac", "1", tmp.name]
    subprocess.run(cmd, check=True)
    return tmp.name


def emb(path, ss=None, t=None):
    w = wav16(path, ss, t)
    try:
        return spkchk.embed(sess(), spkchk.read16(w))
    finally:
        os.unlink(w)


def cos(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


REF = "/mnt/tts/data/cases/1a6c20a60481/merged.wav"
BARE = "/mnt/tts/projects/gpt-sovits/out/cosy_full_m.wav"     # 裸推理 0.953
ONLINE = "/tmp/selftest_open/open_male.wav"                # 线上成品 0.844


def dur(p):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", p], capture_output=True, text=True).stdout.strip())


def main():
    for tag, p in (("参照(案例原声)", REF), ("裸推理", BARE), ("线上成品", ONLINE)):
        print("%-16s %8.1fs" % (tag, dur(p)))
    e_ref = emb(REF)
    e_b = emb(BARE)
    e_o = emb(ONLINE)
    print("\n整条对参照：裸推理 %.4f   线上成品 %.4f" % (cos(e_ref, e_b), cos(e_ref, e_o)))
    print("两者互相似：%.4f  ← 高=测量偏差，低=声学真被改了" % cos(e_b, e_o))

    print("\n分段漂移（每段 120s，看是不是某段崩了）：")
    for tag, p in (("裸推理", BARE), ("线上成品", ONLINE)):
        d = dur(p)
        row = []
        for st in range(0, int(d) - 60, 120):
            row.append("%3ds:%.4f" % (st, cos(e_ref, emb(p, ss=st, t=120.0))))
        print("  %-10s %s" % (tag, "  ".join(row)))


if __name__ == "__main__":
    main()
