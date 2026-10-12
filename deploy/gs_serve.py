# -*- coding: utf-8 -*-
"""GPT-SoVITS 推理服务（端口 8200）—— 把「纯 CLI」的 GS 接进生产链路。

🔴 为什么是独立进程而不是 import
----------------------------------
GS 与 Qwen 用的是**两套互不兼容的 venv**：
    · GS   : /mnt/tts/projects/gpt-sovits/venv
    · Qwen : /mnt/tts/projects/qwen3-tts/venv
两边 torch / transformers / numpy 版本都不一样，直接 import 必炸。
所以沿用 Cosy 的既有做法：独立进程 + HTTP，主服务通过 gs_client 调用。

🔴 GPU 隔离
-----------
    · Qwen 主服务 : 3060  GPU-94853612-464c-9177-c374-a9a82e321229
    · Cosy        : 3090  GPU-b67eee2c-c672-6915-910d-617c62abebee
    · GS（本服务）: 3090  同上
本服务只跑 s2（SoVITS）推理，峰值约 3~4 GiB，与 Cosy 共用 3090（24 GiB）可行。

🔴 必须外部分段（默认 50 字/段）
--------------------------------
整段长文案直接喂给 GS 会触发 GPT 自回归发散：女声 321 字实测覆盖率
**0.150 / CER 0.808**（60 秒文案只出 33.6 秒，大半是胡话）。
外部分段后 0.956 / 0.045。分段不是优化，是**正确性的前提**。

接口
----
    GET  /health                       {"ok":true,"loaded":[...]}
    POST /synth  form: text, spk, ref_wav, ref_text, seed, seg
                → audio/wav
    POST /unload                       释放显存
"""
import glob
import io
import os
import sys
import threading

import numpy as np
import soundfile as sf
import yaml
import torch                                              # noqa: E402
import torchaudio                                         # noqa: E402

REPO = os.environ.get("GS_REPO", "/mnt/tts/projects/gpt-sovits/repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(REPO)


# 🔴 本机 torchaudio 走 torchcodec，缺 libav*.so.58 / libnppicc.so.12
def _sf_load(path, *a, **k):
    w, sr = sf.read(path, dtype="float32")
    t = torch.from_numpy(np.asarray(w))
    if t.ndim == 1:
        t = t.unsqueeze(0)
    else:
        t = t.transpose(0, 1)
    return t, sr


torchaudio.load = _sf_load

from fastapi import FastAPI, Form, HTTPException            # noqa: E402
from fastapi.responses import Response                     # noqa: E402
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

PORT = int(os.environ.get("GS_PORT", "8200"))
GPT_W = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
PRETRAINED_SOVITS = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
SEG_DEFAULT = int(os.environ.get("GS_SEG", "50"))
EPOCH_DEFAULT = os.environ.get("GS_EPOCH", "40")
SEG_GAP_SEC = 0.30
_END = "。！？.!?\n"

app = FastAPI()


def pick_weight(spk, epoch):
    d = os.path.join(REPO, "SoVITS_weights_v2ProPlus")
    fs = sorted(glob.glob(os.path.join(d, "%s_v2pro_e%s_s*.pth" % (spk, epoch))))
    if not fs:
        fs = sorted(glob.glob(os.path.join(d, "%s_v2pro_e*.pth" % spk)))
    if not fs:
        raise HTTPException(500, "找不到 %s 的权重（目录 %s）" % (spk, d))
    return os.path.relpath(fs[-1], REPO)


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


def _mk_req(text, prompt, prompt_text, seed=42):
    return {
        "text": text,
        "text_lang": "ko",
        "ref_audio_path": prompt,
        "prompt_text": prompt_text,
        "prompt_lang": "ko",
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut1",
        "batch_size": 1,
        "speed_factor": 1.0,
        "fragment_interval": 0.3,
        "seed": seed,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "use_cuda_graph": False,     # 本机 cuda_graph 捕获会卡住
    }


def split_text(text, max_chars=SEG_DEFAULT):
    """按句末标点切成 ≤ max_chars 字的片段。

    ⚠️ 只在句末标点处切，绝不硬断词 —— 硬断会让 GPT 学到错误的韵律边界。
    ⚠️ 连续句末符（ASR 常见的 "。。"）要合并，否则切出 2 字碎段。
    """
    parts, buf = [], ""
    for ch in text:
        if ch in _END and not buf.strip():
            if parts:
                parts[-1] = parts[-1] + ch
            continue
        buf += ch
        if ch in _END:
            parts.append(buf.strip())
            buf = ""
        elif len(buf) >= max_chars:
            sp = buf.rfind(" ")
            if sp > max_chars * 0.5:
                parts.append(buf[:sp].strip())
                buf = buf[sp + 1:]
            else:
                parts.append(buf.strip())
                buf = ""
    if buf.strip():
        parts.append(buf.strip())
    return [p for p in parts if p]


def _run(tts, text, ref_wav, ref_text, seed=42, seg=SEG_DEFAULT):
    if seg and seg > 0:
        chunks, sr0, gap = [], None, None
        segs = split_text(text, seg)
        for i, s in enumerate(segs):
            out = None
            for item in tts.run(_mk_req(s, ref_wav, ref_text, seed)):
                out = item
            if out is None:
                raise RuntimeError("第 %d 段没有产出音频" % (i + 1))
            sr, audio = out
            audio = np.asarray(audio)
            sr0 = sr
            if gap is None:
                gap = np.zeros(int(SEG_GAP_SEC * sr), dtype=audio.dtype)
            chunks.append(audio)
            if i < len(segs) - 1:
                chunks.append(gap)
        return sr0, np.concatenate(chunks)
    out = None
    for item in tts.run(_mk_req(text, ref_wav, ref_text, seed)):
        out = item
    if out is None:
        raise RuntimeError("推理没有产出音频")
    sr, audio = out
    return sr, np.asarray(audio)


_TTS = {}
_LOCK = threading.Lock()


def _get(key):
    if key in _TTS:
        return _TTS[key]
    with _LOCK:
        if key in _TTS:
            return _TTS[key]
        if key == "__zero__":
            w = PRETRAINED_SOVITS
        else:
            # ⚠️ spk 允许两种写法：「SOLA」或「SOLA@30」。
            #    早期版本直接 rsplit("@",1) 解包成两个变量，传纯组名时会抛
            #    ValueError: not enough values to unpack —— 前端点一下就 500。
            spk, _, ep = key.partition("@")
            w = pick_weight(spk, ep or EPOCH_DEFAULT)
        _TTS[key] = TTS(build(w))
        return _TTS[key]


@app.get("/health")
def health():
    return {"ok": True, "loaded": sorted(_TTS.keys())}


@app.post("/unload")
def unload():
    _TTS.clear()
    torch.cuda.empty_cache()
    return {"ok": True}


@app.post("/synth")
def synth(text: str = Form(...), spk: str = Form("SOLA"),
          ref_wav: str = Form(""), ref_text: str = Form(""),
          seed: int = Form(42), seg: int = Form(SEG_DEFAULT)):
    """text 可以很长，内部会按段推理再拼接（见文件头说明）。"""
    if not text.strip():
        raise HTTPException(400, "文本为空")
    if not ref_wav or not os.path.exists(ref_wav):
        raise HTTPException(400, "参考音频不存在: %s" % ref_wav)
    try:
        sr, y = _run(_get(spk), text, ref_wav, ref_text,
                     seed=seed, seg=seg)
    except Exception as e:
        raise HTTPException(500, "推理失败: %s" % repr(e)[:200])
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    print("GS 服务启动 :%d  loaded=%s" % (PORT, sorted(_TTS.keys())),
          flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
