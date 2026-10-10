# -*- coding: utf-8 -*-
"""GPT-SoVITS 长文本稳定性验证。

短文本（304 字）上 GS 微调版说话人相似度 0.9362，明显高于 Qwen 的 0.8441。
但 **我们真实业务是单篇 3000 字左右、成片 3~8 分钟**，而"长文本发散/吞字"
恰恰是 GPT-SoVITS 的已知弱项。所以在决定要不要动生产之前，必须先跑这一关。

测试文本刻意取自**另一个案例**（산일전기, 799e4d7060ce）的全文前 2902 字：
  - 对 GS 微调模型来说是完全没见过的文本（韩华案例只喂了 이민정 那份）
  - 内容是真实的韩语财经口播稿，和线上文案同分布

判定口径（和 Qwen 长文验收同一套）：
  1. 完整性 —— 结尾那句有没有被吃掉 / 有没有截断
  2. CER    —— 用 faster-whisper 转回来和原文比，看吞字/重复
  3. 时长   —— dur_ratio = 生成时长 / 参考时长，越接近 1 越好
  4. 音色   —— 抽头/中/尾三段和案例参照比 campplus 相似度，看会不会中途跑掉

用法：
    python gs_infer_long.py                 # 默认 e30 权重
    python gs_infer_long.py hh_v2pro_e20_s320.pth
"""
import os
import sys
import numpy as np

REPO = "/mnt/tts/projects/gpt-sovits/repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(REPO)

import yaml  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402


# 🔴 torchaudio 在这台机器上一律走 torchcodec，缺 libavcodec.so.58 / libnppicc.so.12，
#    直接用 soundfile 顶掉，不动上游代码。
def _sf_load(path):
    w, sr = sf.read(path, dtype="float32")
    t = torch.from_numpy(np.asarray(w))
    if t.ndim == 1:
        t = t.unsqueeze(0)
    else:
        t = t.transpose(0, 1)
    return t, sr


torchaudio.load = _sf_load

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

OUT = "/mnt/tts/projects/gpt-sovits/out"
os.makedirs(OUT, exist_ok=True)

TEXT = open("/mnt/tts/diag/gs_long.txt", encoding="utf-8").read().strip()
PROMPT = os.path.join(OUT, "prompt_10s.wav")
PROMPT_TXT = open(os.path.join(OUT, "prompt_10s.txt"), encoding="utf-8").read().strip()

GPT_W = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
SOVITS_DIR = "SoVITS_weights_v2ProPlus"


def build(sovits_path):
    cfg = yaml.safe_load(open("GPT_SoVITS/configs/tts_infer.yaml"))
    c = cfg.get("custom") or {}
    c["version"] = "v2ProPlus"
    c["device"] = "cuda"
    c["is_half"] = True
    c["t2s_weights_path"] = GPT_W
    c["vits_weights_path"] = sovits_path
    cfg["custom"] = c
    return TTS_Config(cfg)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "hh_v2pro_e30_s480.pth"
    wpath = os.path.join(SOVITS_DIR, name)
    print("权重:", wpath, "存在:", os.path.exists(wpath), flush=True)
    print("文本字数:", len(TEXT), flush=True)
    print("尾句:", TEXT[-60:], flush=True)
    print("提示文本:", PROMPT_TXT, flush=True)

    tts = TTS(build(wpath))
    out = None
    n = 0
    # ⚠️ 这个版本 TTS.run 是生成器：短文本 yield 一包，长文本按分段 yield 多包。
    #    这里把每一包都记下来，既能看分段数，也能确认最后一段真的产出来了。
    segs = []
    for item in tts.run({
        "text": TEXT,
        "text_lang": "ko",
        "ref_audio_path": PROMPT,
        "prompt_text": PROMPT_TXT,
        "prompt_lang": "ko",
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut1",
        "batch_size": 1,
        "speed_factor": 1.0,
        "fragment_interval": 0.3,
        "seed": 42,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "use_cuda_graph": False,          # 这台机器上 cuda_graph 捕获会卡住
    }):
        sr, audio = item
        a = np.asarray(audio)
        segs.append(a)
        n += 1
        print("  分段 %d: %.2fs" % (n, len(a) / sr), flush=True)
        out = (sr, a)

    if not segs:
        raise RuntimeError("推理没有产出音频")

    sr = out[0]
    full = np.concatenate(segs) if len(segs) > 1 else segs[0]
    dst = os.path.join(OUT, "gs_long_%s.wav" % name.replace(".pth", ""))
    sf.write(dst, full, sr, subtype="PCM_16")
    print("分段数:", len(segs), flush=True)
    print("总时长: %.1fs @%dHz" % (len(full) / sr, sr), flush=True)
    print("->", dst, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
