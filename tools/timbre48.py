# -*- coding: utf-8 -*-
"""timbre48.py —— 同长度音色对比（campplus，×EL 男声标杆）。

🔴 记忆铁律：音色对比必须**同素材同长度**。
成品 201s 与 EL 标杆 509s 直接比是不公平的（多窗平均会稀释）。
做法：把 EL 标杆**截到与候选同样长**再比。

用法：python3 timbre48.py <候选1> [候选2 ...] --el <标杆> --dur <秒>
      dur 省略则自动取候选的时长（截 EL 到同样长）。
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")
import onnxruntime  # noqa: E402
import spkchk  # noqa: E402

opt = onnxruntime.SessionOptions()
opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
opt.intra_op_num_threads = 2
SESS = onnxruntime.InferenceSession(spkchk.CAMPPLUS, sess_options=opt,
                                    providers=["CPUExecutionProvider"])


def emb(p):
    return spkchk.embed(SESS, spkchk.read16(p))


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def trim(src, dur, tag):
    d = "/tmp/t48_%s.wav" % tag
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-t", "%.2f" % dur, "-ar", "16000", "-ac", "1", d],
                   check=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cands", nargs="+")
    ap.add_argument("--el", default="/mnt/tts/data/cases/6cdd25bc75f2/merged.wav")
    ap.add_argument("--dur", type=float, default=0.0,
                    help="统一时长(s)；0=自动按候选")
    a = ap.parse_args()

    import soundfile as sf
    print("=== 音色（campplus 16k，同长度，×EL 男声标杆）===")
    print("  同人阈值 0.75")
    for c in a.cands:
        dur = a.dur or sf.info(c).duration
        el = trim(a.el, dur, os.path.basename(c)[:12])
        s = cos(emb(c), emb(el))
        print("  %-34s %.2fs  %.4f  %s"
              % (os.path.basename(c)[:34], dur, s,
                 "同人 OK" if s >= 0.75 else "**不同人**"))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
