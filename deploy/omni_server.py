"""OmniVoice 独立推理服务（端口 8300）—— 女声最佳链路的合成端。

为什么要独立进程：OmniVoice 自带 venv（torch 2.14+cu130、numpy 2.5），
与 Qwen3-TTS 主服务、CosyVoice3(cu121) 互相冲突，只能「独立 venv + HTTP」。

对外接口（与 cosy_server.py 同形态，便于 pipeline 统一调用）：
    GET  /health    -> 健康检查 + 显存
    POST /synth     -> {"text":..., "voice":"female"} 返回 wav 二进制
    POST /voices    -> 覆盖某音色的参考音频/文字稿
    POST /unload    -> 卸载模型释放显存
    GET  /progress  -> ?rid=... 查询进度

⚠️ 复现获胜链路的硬约束（改这里必看）：
  女声冠军 X2_f.wav 是 **整篇一次提交** 生成的（3911 字 -> 513s，段数=1）。
  OmniVoice 长文稳定，所以这里默认**不分块**；分块会改变韵律，等于换了个方案。
  参考音频必须是 gpt-sovits/out/prompt_sola_00b6a45a（= EL 女声标杆切片）。
"""
import io
import os
import gc
import sys
import time
import threading
import logging

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

BASE = "/mnt/tts/projects/omnivoice"
if BASE not in sys.path:
    sys.path.insert(0, BASE)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("omni")

MODEL_DIR = os.environ.get("OMNI_MODEL", os.path.join(BASE, "models/OmniVoice"))
GSOUT = "/mnt/tts/projects/gpt-sovits/out"
SR = 24000

# voice -> (参考音频, 文字稿)。与盲听获胜链路一致，不要随手换。
VOICES = {
    "female": (os.path.join(GSOUT, "prompt_sola_00b6a45a.wav"),
               os.path.join(GSOUT, "prompt_sola_00b6a45a.txt")),
}
DEFAULT_VOICE = "female"

app = FastAPI(title="OmniVoice TTS Service")
_model = None
_lock = threading.Lock()      # 只保护「加载/卸载」
# 生成串行锁：OmniVoice 跑在 3060 上、显存只有 ~2G 余量不大，
# 且与同卡的主服务共享。实测并发会互相挤；这里直接排队，第二个请求等着。
# （VoxCPM2 那边更严重——并发会死锁，见 vox_server.py 的说明。）
_gen_lock = threading.Lock()
_PROG = {}
_PROG_LOCK = threading.Lock()


def load_model():
    global _model
    with _lock:
        if _model is None:
            from omnivoice import OmniVoice
            t = time.time()
            log.info("加载 OmniVoice ...")
            _model = OmniVoice.from_pretrained(MODEL_DIR, device_map="cuda",
                                               dtype=torch.float16)
            log.info("加载完成 %.1fs", time.time() - t)
        return _model


def unload_model():
    global _model
    with _lock:
        if _model is not None:
            del _model
            _model = None
            gc.collect()
            torch.cuda.empty_cache()
            log.info("已卸载")


class SynthReq(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = DEFAULT_VOICE
    rid: str = ""


@app.get("/health")
def health():
    m = {}
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        m = {"device": p.name,
             "allocated_GiB": round(torch.cuda.memory_allocated() / 1024 ** 3, 2),
             "reserved_GiB": round(torch.cuda.memory_reserved() / 1024 ** 3, 2),
             "total_GiB": round(p.total_memory / 1024 ** 3, 1)}
    return {"ok": True, "loaded": _model is not None, "voices": sorted(VOICES), **m}


@app.post("/synth")
def synth(req: SynthReq):
    if req.voice not in VOICES:
        raise HTTPException(400, f"未知音色 {req.voice}，可用：{sorted(VOICES)}")
    m = load_model()
    rw, rt = VOICES[req.voice]
    with open(rt, encoding="utf-8") as f:
        rtext = f.read().strip()
    t = time.time()
    try:
        with _gen_lock:
            out = m.generate(text=req.text, ref_audio=rw, ref_text=rtext)
    except Exception as e:
        log.exception("生成失败")
        raise HTTPException(500, f"生成失败: {e!r}")
    arr = np.asarray(out[0]) if isinstance(out, (list, tuple)) else np.asarray(out)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    dur = len(arr) / SR
    buf = io.BytesIO()
    sf.write(buf, arr.astype(np.float32), SR, subtype="PCM_16", format="WAV")
    if req.rid:
        with _PROG_LOCK:
            _PROG[req.rid] = {"done": 1, "total": 1}
    torch.cuda.empty_cache()
    log.info("synth voice=%s %d字 -> %.1fs 用时 %.1fs RTF=%.3f",
             req.voice, len(req.text), dur, time.time() - t,
             (time.time() - t) / max(dur, 1e-6))
    return Response(content=buf.getvalue(), media_type="audio/wav",
                    headers={"X-Audio-Seconds": f"{dur:.2f}",
                             "X-Sample-Rate": str(SR)})


class VoiceReq(BaseModel):
    voice: str
    wav: str
    txt: str


@app.post("/voices")
def set_voice(req: VoiceReq):
    if not os.path.exists(req.wav) or not os.path.exists(req.txt):
        raise HTTPException(400, "wav/txt 路径不存在")
    VOICES[req.voice] = (req.wav, req.txt)
    return {"ok": True, "voices": sorted(VOICES)}


@app.post("/unload")
def unload():
    unload_model()
    return {"ok": True}


@app.get("/progress")
def progress(rid: str = ""):
    with _PROG_LOCK:
        p = _PROG.get(rid)
    # "found" 是 cosy_client._poll_progress 的约定字段
    return {"found": bool(p), **(p or {"done": 0, "total": 0})}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1",
                port=int(os.environ.get("OMNI_PORT", "8300")), log_level="info")
