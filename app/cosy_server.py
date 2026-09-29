"""CosyVoice3 独立推理服务（端口 8100）。

为什么要独立成一个服务、而不是 import 进 tts-kr：
    CosyVoice3 的 requirements 锁 torch==2.3.1+cu121 / numpy==1.26.4，
    与现网 Qwen3-TTS 的 torch 2.14+cu130 直接冲突，两个环境无法共存于
    同一个 Python 进程。所以走「独立 venv + 独立进程 + HTTP 调用」，
    和 tts-kr 主服务解耦，互不影响。

部署位置：/mnt/tts/projects/cosy/cosy_server.py
启动 venv：/mnt/tts/projects/cosy/venv
模型：    /mnt/tts/models/Fun-CosyVoice3-0.5B

对外只暴露三个接口（极简，参数不拼装）：
    GET  /health   -> 健康检查 + 显存
    POST /synth    -> 输入文本与参考音频，直接返回 wav 二进制
    POST /voices   -> 注册/覆盖一个音色模板（存 spk2info，可复用）
"""
import os
import sys
import re
import io
import gc
import json
import time
import logging

# CosyVoice 的代码与 Matcha-TTS 子模块都要在 sys.path 里
_COSY_ROOT = os.environ.get("COSY_ROOT", "/mnt/tts/projects/cosy/repo")
for _p in (_COSY_ROOT, os.path.join(_COSY_ROOT, "third_party", "Matcha-TTS")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cosy")

MODEL_DIR = os.environ.get("COSY_MODEL", "/mnt/tts/models/Fun-CosyVoice3-0.5B")
FP16 = os.environ.get("COSY_FP16", "0") == "1"
# CosyVoice3 的 zero-shot 强制要求 prompt_text 里带 <|endofprompt|>，
# 不带会在 flow 阶段报 "Kernel size can't be greater than actual input size"。
INSTRUCT_PRESETS = {
    "neutral": "You are a helpful assistant.<|endofprompt|>",
    "excited": ("Speak with an excited and passionate tone, like a lively live "
                "stock market broadcast, with rich intonation and natural "
                "rhythm.<|endofprompt|>"),
    "warm": ("Speak in a warm, calm and conversational tone, like talking to a "
             "close friend.<|endofprompt|>"),
    "serious": ("Speak in a steady, authoritative and serious tone, like a "
                "professional news anchor.<|endofprompt|>"),
}
DEFAULT_INSTRUCT = "neutral"

app = FastAPI(title="CosyVoice3 TTS Service")
_model = None
_sr = 24000
_last_used = 0.0

# 显存是共享资源：现网 Qwen3-TTS 常驻约 4.25 GiB，CosyVoice3 再常驻约 3.2 GiB。
# 两个引擎同时常驻会挤压 Qwen 的生成峰值（实测峰值 6.97 GiB）。
# 所以默认「空闲就卸载」——员工不用 CosyVoice 时，它一兆显存都不占。
IDLE_UNLOAD_SEC = float(os.environ.get("COSY_IDLE_UNLOAD_SEC", "600"))


def _release_model():
    """把模型从显存彻底释放。注意必须是「重释放」：
    del + empty_cache 不够 —— 跨进程场景（自动选型先卸 Cosy 再加载 Qwen）
    实测 unload 返回后 nvidia-smi 仍显示本进程持有 5~7.6G，
    立刻加载 Qwen 直接 OOM。gc.collect() 清引用环 + synchronize 等 CUDA 收尾
    + 二次 empty_cache，才能把显存真正还给驱动器。
    """
    global _model
    if _model is None:
        return
    try:
        del _model
    except Exception:
        pass
    _model = None
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()


def _maybe_unload():
    """空闲超过阈值就把模型从显存释放，把显存让回给 Qwen。"""
    global _last_used
    if _model is None or IDLE_UNLOAD_SEC <= 0:
        return
    if time.time() - _last_used > IDLE_UNLOAD_SEC:
        log.info("空闲 %.0fs 超过阈值 %.0fs，卸载 CosyVoice3 释放显存",
                 time.time() - _last_used, IDLE_UNLOAD_SEC)
        _release_model()


def get_model():
    """懒加载：首次请求时才把模型读进显存，避免服务启动就被卡住。"""
    global _model, _sr, _last_used
    _maybe_unload()
    if _model is None:
        from cosyvoice.cli.cosyvoice import AutoModel
        t0 = time.time()
        _model = AutoModel(model_dir=MODEL_DIR, fp16=FP16)
        _sr = _model.sample_rate
        log.info("模型加载完成 %.1fs  fp16=%s  sr=%d",
                 time.time() - t0, FP16, _sr)
    _last_used = time.time()
    return _model


@app.get("/health")
def health():
    # 健康检查不触发加载（否则心跳会把模型常驻住），顺手做一次空闲回收
    _maybe_unload()
    m = _model is not None
    vram = (torch.cuda.memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0.0
    idle = (time.time() - _last_used) if _last_used else None
    return {"status": "ok" if m else "model_not_loaded",
            "model": MODEL_DIR, "fp16": FP16, "sample_rate": _sr,
            "vram_gib": round(vram, 2),
            "idle_sec": None if idle is None else round(idle, 1),
            "idle_unload_sec": IDLE_UNLOAD_SEC,
            "instruct_presets": list(INSTRUCT_PRESETS.keys())}


@app.post("/unload")
def unload():
    """手动卸载，立即把显存让回给 Qwen。

    同步释放：返回时显存已真正还给驱动器（gc + synchronize + 二次 empty_cache），
    并携带 gpu_free_gib（整卡空闲显存），方便调用方确认可以再加载另一个引擎。
    """
    was_loaded = _model is not None
    _release_model()
    vram = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
    try:
        free_b, _ = torch.cuda.mem_get_info()
        gpu_free = round(free_b / 1024 ** 3, 2)
    except Exception:
        gpu_free = None
    return {"ok": True, "was_loaded": was_loaded,
            "vram_gib": round(vram, 2), "gpu_free_gib": gpu_free}


class SynthReq(BaseModel):
    text: str = Field(..., description="待合成文本（韩语等）")
    ref_wav: str = Field("", description="参考音频绝对路径（克隆音色用）")
    ref_text: str = Field("", description="参考音频对应的文字稿")
    instruct: str = Field(DEFAULT_INSTRUCT,
                          description="情感指令；可为 preset 名或自定义文本")
    speed: float = Field(1.0, ge=0.5, le=2.0)
    seed: int = Field(42)
    max_chars: int = Field(320, ge=20, le=400,
                           description="按句切分后单块的最大字数。"
                                       "实测（同文本 776 字韩语）块越长 CER 越低："
                                       "80→0.0892 / 160→0.0829 / 240→0.0850 / 320→0.0727，"
                                       "故默认取 320")
    text_frontend: bool = Field(False,
                                description="韩语必须 False，否则文本前端会切坏")


def _resolve_instruct(name: str) -> str:
    """指令既可以是预设名，也可以是完整自定义文本。"""
    s = (name or "").strip()
    if not s:
        return INSTRUCT_PRESETS[DEFAULT_INSTRUCT]
    if s in INSTRUCT_PRESETS:
        return INSTRUCT_PRESETS[s]
    # 自定义文本必须自带分隔符，否则模型会把指令当成参考文本的一部分
    return s if s.endswith("<|endofprompt|>") else s + "<|endofprompt|>"


def _split_sentences(text: str, max_chars: int = 160):
    """按句末标点切句，再把短句合并到 max_chars 以内。

    为什么不整段丢给模型：CosyVoice3 在超长输入上会出现念串/幻觉，
    实测整段 3699 字直接生成的 CER 明显高于按句生成。
    """
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip())
             if s.strip()]
    if not parts:
        return []
    chunks, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) + 1 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + " " + p).strip() if cur else p
    if cur:
        chunks.append(cur)
    return chunks


@app.post("/synth")
def synth(req: SynthReq):
    if not req.text.strip():
        raise HTTPException(400, "text 为空")
    if not req.ref_wav or not os.path.exists(req.ref_wav):
        raise HTTPException(400, f"参考音频不存在: {req.ref_wav}")
    cosy = get_model()
    torch.manual_seed(req.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(req.seed)

    instruct = _resolve_instruct(req.instruct)
    prompt_text = instruct + (req.ref_text or "")
    chunks = _split_sentences(req.text, max_chars=req.max_chars)
    log.info("synth: %d字 -> %d块  instruct=%s speed=%s max_chars=%d",
             len(req.text), len(chunks), req.instruct, req.speed, req.max_chars)

    outs, t0 = [], time.time()
    for i, c in enumerate(chunks):
        try:
            for o in cosy.inference_zero_shot(c, prompt_text, req.ref_wav,
                                              stream=False, speed=req.speed,
                                              text_frontend=req.text_frontend):
                outs.append(o["tts_speech"])
        except Exception as e:
            log.warning("第 %d 块生成失败: %s", i, repr(e)[:200])
    if not outs:
        raise HTTPException(500, "全部片段生成失败")

    wav = torch.cat(outs, dim=1)
    dur = wav.shape[1] / _sr
    buf = io.BytesIO()
    torchaudio.save(buf, wav, _sr, format="wav")
    body = buf.getvalue()
    return Response(
        content=body, media_type="audio/wav",
        headers={"X-Audio-Seconds": f"{dur:.3f}",
                 "X-Gen-Seconds": f"{time.time() - t0:.3f}",
                 "X-Chunks": str(len(chunks)),
                 "X-Sample-Rate": str(_sr)})


class VoiceReq(BaseModel):
    name: str
    ref_wav: str
    ref_text: str = ""


@app.post("/voices")
def register_voice(req: VoiceReq):
    """把音色注册进 spk2info，后续可直接按 id 复用，省掉每次重算 speaker 嵌入。

    注意：CosyVoice3 的 zero-shot 仍然需要 prompt_text（含情感指令），
    所以这里只缓存「音色」，情感指令在 /synth 时传入。
    """
    if not os.path.exists(req.ref_wav):
        raise HTTPException(400, f"参考音频不存在: {req.ref_wav}")
    cosy = get_model()
    ok = cosy.add_zero_shot_spk(req.ref_text, req.ref_wav, req.name)
    cosy.save_spkinfo()
    return {"ok": bool(ok), "name": req.name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1",
                port=int(os.environ.get("COSY_PORT", "8100")),
                timeout_keep_alive=600)
