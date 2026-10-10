# -*- coding: utf-8 -*-
"""运行官方 2-get-sv.py，但把 torchaudio.load 换成 soundfile。

🔴 为什么要有这个包装：
    这台服务器上 PyTorch torchaudio 2.11 一律走 torchcodec 后端，而 wheel 编的
    libtorchcodec_core4.so 要 FFmpeg 4 的 libavcodec.so.58 / libavformat.so.58
    ... ；系统只有 FFmpeg 6 的 .so.60，所以 `torchaudio.load` 直接抛
    "Could not load libtorchcodec"。
    这里用 runpy 跑官方脚本、但先 monkeypatch 掉 torchaudio.load，
    **不改上游一行代码**（升级仓库不会冲突）。

用法：python gs_run_sv.py
"""
import os
import sys
import runpy

REPO = "/mnt/tts/projects/gpt-sovits/repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))

import numpy as np
import torch
import torchaudio
import soundfile as sf


def _load(path):
    w, sr = sf.read(path, dtype="float32")
    t = torch.from_numpy(np.asarray(w))
    if t.ndim == 1:
        t = t.unsqueeze(0)          # [N] -> [1, N]
    else:
        t = t.transpose(0, 1)       # [N, C] -> [C, N]
    return t, sr


torchaudio.load = _load

os.chdir(REPO)
runpy.run_path(os.path.join(REPO, "GPT_SoVITS", "prepare_datasets", "2-get-sv.py"),
               run_name="__main__")
