# -*- coding: utf-8 -*-
"""spkcmp53.py —— 音色相似度（campplus 余弦）对照：我们 vs EL 同文案。

## 为什么现在才测它
measure53 的 5 个指标（click/质心/中频/停顿/杂音）全对齐了，但**音色相似度
一直没进日常测量**。记忆里现役男声只有 **0.858**，EL 上限 **0.9628**
→ 这可能是当前最大的剩余差距，而且它**完全客观，不需要员工**。

更紧迫的是：本轮新加的 LTAS（改频谱）和 clickfix（改波形）都可能伤音色，
**必须先确认没改坏**，否则前面的优化是拆东墙补西墙。

## 口径（严格照 spkchk.py 的踩坑记录）
1. 必须用 **CosyVoice3 的 venv**（`/mnt/tts/projects/cosy/venv`），
   只有它有 `torchaudio.compliance.kaldi` + onnxruntime。
2. campplus 吃 **kaldi fbank(80 mel)**，不是波形；必须 **16 kHz**。
3. 官方 `inference` 是**随机截 10 秒** → 改成**多窗平均**（nwin=12，均匀铺开）。
4. 同人阈值 **0.75**。

## 用法（服务器）
    /mnt/tts/projects/cosy/venv/bin/python tools/spkcmp53.py \\
        <ours.wav> <el.wav> [更多成对文件...]

成对给：第 1 个是我们、第 2 个是 EL，以此类推。
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import onnxruntime
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi

CAMPPLUS = os.environ.get(
    "TTS_CAMPPLUS", "/mnt/tts/models/Fun-CosyVoice3-0.5B/campplus.onnx")
SR = 16000
WIN = 10 * SR
SAME_THR = 0.75


def read16(path):
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir="/tmp")
    os.close(fd)
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path,
                        "-ac", "1", "-ar", str(SR), tmp], check=True,
                       timeout=1800)
        return torch.from_numpy(sf.read(tmp, dtype="float32")[0]).unsqueeze(0)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _fbank(speech):
    f = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=SR)
    return f - f.mean(dim=0, keepdim=True)


def embed(sess, speech, nwin=12):
    n = speech.shape[1]
    if n < SR:
        return None
    if n <= WIN:
        idxs = [0]
    else:
        idxs = [int(round(i * (n - WIN) / max(nwin - 1, 1)))
                for i in range(nwin)]
    vs = []
    for s in idxs:
        seg = speech[:, s:s + WIN]
        if seg.shape[1] < SR // 2:
            continue
        feat = _fbank(seg).unsqueeze(0).cpu().numpy()
        e = sess.run(None, {sess.get_inputs()[0].name: feat})[0].flatten()
        e = np.asarray(e, dtype=np.float64)
        nv = np.linalg.norm(e)
        if nv > 1e-12:
            vs.append(e / nv)
    if not vs:
        return None
    v = np.mean(vs, axis=0)
    nv = np.linalg.norm(v)
    return v / nv if nv > 1e-12 else None


def main():
    fs = sys.argv[1:]
    if len(fs) < 2 or len(fs) % 2:
        print(__doc__)
        return 1
    opt = onnxruntime.SessionOptions()
    opt.graph_optimization_level = \
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.intra_op_num_threads = 2
    sess = onnxruntime.InferenceSession(
        CAMPPLUS, sess_options=opt, providers=["CPUExecutionProvider"])

    print("%-28s %-28s %8s" % ("我们", "EL", "音色余弦"))
    print("-" * 68)
    for i in range(0, len(fs), 2):
        a, b = fs[i], fs[i + 2 - 2] if False else fs[i + 1]
        ea = embed(sess, read16(a))
        eb = embed(sess, read16(b))
        if ea is None or eb is None:
            print("%-28s %-28s  %8s" % (os.path.basename(a),
                                        os.path.basename(b), "NA"))
            continue
        cos = float(np.dot(ea, eb))
        flag = " ✅同人" if cos >= SAME_THR else " ❌不同人"
        print("%-28s %-28s %8.4f%s"
              % (os.path.basename(a)[:28], os.path.basename(b)[:28],
                 cos, flag))
    print("-" * 68)
    print("同人阈值 %.2f" % SAME_THR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
