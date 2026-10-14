"""VoxCPM2 独立推理服务（端口 8400）—— 男声最佳链路的合成端。

为什么要独立进程：VoxCPM2 自带 venv（torch 2.14+cu130、numpy 2.5），
与 Qwen3-TTS 主服务、CosyVoice3(cu121) 互相冲突，只能「独立 venv + HTTP」。

对外接口（与 cosy_server.py 同形态）：
    GET  /health    -> 健康检查 + 显存
    POST /synth     -> {"text":..., "voice":"male"} 返回 wav 二进制
    POST /voices    -> 覆盖某音色的参考音频/文字稿
    POST /unload    -> 卸载模型释放显存
    GET  /progress  -> ?rid=... 查询进度（分块生成时用）

🔴 两个必须继承的硬约束（都是踩过的坑，改这里必看）：
  1) **必须分块**。整篇一次提交会「静默截断」：2026-10-13 实测女声 3948 字
     只出 218.4s 就正常返回（EL 525s / 分块 487s）——不报错，比崩了更危险。
     这里默认强制按句边界切块（CHUNK_CHARS=600，与获胜的 X1_m 完全一致）。
  2) **峰值显存会冲到 ~10.9 GiB**（allocated 只有 5.1，其余是缓存分配器）。
     每次生成后必须 empty_cache()，否则 reserved 一直挂着挤别人。
  参考音频必须是 gpt-sovits/out/prompt_yohan_1a6c20a6（= EL 男声标杆切片）。
"""
import io
import os
import re
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

BASE = "/mnt/tts/projects/voxcpm"
if BASE not in sys.path:
    sys.path.insert(0, BASE)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vox")

MODEL_DIR = os.environ.get("VOX_MODEL", os.path.join(BASE, "models/VoxCPM2"))
GSOUT = "/mnt/tts/projects/gpt-sovits/out"

# 分块回退大小（句边界对齐）。获胜的 X1_m 就是这个值，别改。
CHUNK_CHARS = int(os.environ.get("VOX_CHUNK_CHARS", "600"))
CFG = float(os.environ.get("VOX_CFG", "2.0"))
STEPS = int(os.environ.get("VOX_STEPS", "10"))

VOICES = {
    "male": (os.path.join(GSOUT, "prompt_yohan_1a6c20a6.wav"),
             os.path.join(GSOUT, "prompt_yohan_1a6c20a6.txt")),
}
DEFAULT_VOICE = "male"

app = FastAPI(title="VoxCPM2 TTS Service")
_model = None
_sr = 48000
_lock = threading.Lock()      # 只保护「加载/卸载」
# 🔴 生成串行锁：实测两个 /synth 并发打同一个模型实例会**死锁**
#   （进度条停在 246/3490、GPU 0%、进程还在）。VoxCPM2 内部不是线程安全的，
#   必须排队。宁可让第二个请求等着，也不能让它把整卡拖死。
_gen_lock = threading.Lock()
_PROG = {}
_PROG_LOCK = threading.Lock()


def load_model():
    global _model, _sr
    with _lock:
        if _model is None:
            from voxcpm.core import VoxCPM
            t = time.time()
            log.info("加载 VoxCPM2 ...")
            _model = VoxCPM(voxcpm_model_path=MODEL_DIR,
                            enable_denoiser=False, optimize=True, device="cuda")
            _sr = int(_model.tts_model.sample_rate)
            log.info("加载完成 %.1fs sr=%d", time.time() - t, _sr)
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


def split_sentences(text, limit):
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 > limit and cur:
            chunks.append(cur.strip())
            cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


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
    return {"ok": True, "loaded": _model is not None,
            "voices": sorted(VOICES), "sr": _sr, **m}


@app.post("/synth")
def synth(req: SynthReq):
    if req.voice not in VOICES:
        raise HTTPException(400, f"未知音色 {req.voice}，可用：{sorted(VOICES)}")
    m = load_model()
    rw, rt = VOICES[req.voice]
    with open(rt, encoding="utf-8") as f:
        rtext = f.read().strip()

    chunks = split_sentences(req.text, CHUNK_CHARS)
    total = max(len(chunks), 1)
    if req.rid:
        with _PROG_LOCK:
            _PROG[req.rid] = {"done": 0, "total": total}
    t = time.time()
    pieces = []
    with _gen_lock:
        for i, c in enumerate(chunks):
            try:
                a = m.generate(text=c, reference_wav_path=rw,
                               prompt_wav_path=rw, prompt_text=rtext,
                               cfg_value=CFG, inference_timesteps=STEPS,
                               normalize=True)
            except Exception as e:
                log.exception("第 %d 块生成失败", i)
                raise HTTPException(500, f"第 {i} 块生成失败: {e!r}")
            pieces.append(np.asarray(a))
            # 每块结束就还一次缓存，避免 reserved 一路涨到 14GiB（实测过）
            torch.cuda.empty_cache()
            if req.rid:
                with _PROG_LOCK:
                    _PROG[req.rid] = {"done": i + 1, "total": total}
    arr = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    dur = len(arr) / _sr
    buf = io.BytesIO()
    sf.write(buf, arr.astype(np.float32), _sr, subtype="PCM_16", format="WAV")
    el = time.time() - t
    # 峰值 reserved 会涨到 ~10.9GiB（缓存分配器），用完必须还回去
    torch.cuda.empty_cache()
    log.info("synth voice=%s %d字/%d块 -> %.1fs 用时 %.1fs RTF=%.3f reserved=%.2fGiB",
             req.voice, len(req.text), total, dur, el, el / max(dur, 1e-6),
             torch.cuda.memory_reserved() / 1024 ** 3)
    return Response(content=buf.getvalue(), media_type="audio/wav",
                    headers={"X-Audio-Seconds": f"{dur:.2f}",
                             "X-Sample-Rate": str(_sr),
                             "X-Chunks": str(total)})


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
                port=int(os.environ.get("VOX_PORT", "8400")), log_level="info")
