"""OmniVoice 独立推理服务（端口 8300）—— 男女声合成端。

为什么要独立进程：OmniVoice 自带 venv（torch 2.14+cu130、numpy 2.5），
与 Qwen3-TTS 主服务、CosyVoice3(cu121) 互相冲突，只能「独立 venv + HTTP」。

对外接口（与 cosy_server.py 同形态，便于 pipeline 统一调用）：
    GET  /health    -> 健康检查 + 显存
    POST /synth     -> {"text":..., "voice":"female"|"male"} 返回 wav 二进制
    POST /voices    -> 覆盖某音色的参考音频/文字稿
    POST /unload    -> 卸载模型释放显存
    GET  /progress  -> ?rid=... 查询进度

⚠️ 复现获胜链路的硬约束（改这里必看）：
  冠军链路都是**整篇一次提交**（女声 X2_f 3911 字 -> 513s；男声 4594 字 -> 553s）。
  OmniVoice 长文稳定，所以默认**不分块**；分块会改变韵律，等于换了个方案。
  参考音频默认是 gpt-sovits/out/ 下的 prompt_sola_00b6a45a（女）
  / prompt_yohan_1a6c20a6（男），都是 EL 标杆切片；2026-10-21 起可用
  TTS_OMNI_PROMPTS（默认 /mnt/tts/data/voices/omni_prompts.json）**按音色名**
  覆盖，且 /voices 会写回该文件，重启不丢。
"""
import io
import os
import gc
import json
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
    # 🔴 2026-10-17 z24 员工盲听裁决：**男声 omnivoice 1.5 分 vs voxcpm 0 分**。
    #    全篇实测（4594 字同文案）：音色 0.9557 vs 0.7928、click 0.0132 ≈ EL 0.0124、
    #    cps 8.31 ≈ EL 8.27、质心 802 ≈ EL 818 —— 除 F0/停顿外全面占优。
    #    男声链路整体切到 omnivoice，参考音频用 yohan 的 10s 切片（F0 196.1，
    #    与 EL 男声 200.7 接近；voxcpm 输出会固有压到 ~180，这是引擎特性）。
    #    同样**整篇一次提交**（分块会换韵律，男声 4594 字实测 553s 正常返回）。
    "male": (os.path.join(GSOUT, "prompt_yohan_1a6c20a6.wav"),
             os.path.join(GSOUT, "prompt_yohan_1a6c20a6.txt")),
}
DEFAULT_VOICE = "female"

# ---- 参考音频可配置（2026-10-21 P0）-------------------------------------
# 历史问题：VOICES 写死两段 10s 切片，改参考 / 重录音色对 omni **完全无效**；
# 线上临时换参考只能调 POST /voices，**服务一重启就回旧值**。
# 现在：启动时读一份 JSON 覆盖表，POST /voices 也会写回它 → 重启不丢。
# 兼容性：VOICES 里仍然保留 male/female 两个键，旧客户端（只发性别）行为不变。
PROMPT_JSON = os.environ.get("TTS_OMNI_PROMPTS",
                             "/mnt/tts/data/voices/omni_prompts.json")


def _load_prompt_overrides():
    try:
        with open(PROMPT_JSON, encoding="utf-8") as f:
            d = json.load(f)
    except FileNotFoundError:
        log.info("无参考覆盖表 %s（用内置 default）", PROMPT_JSON)
        return
    except Exception as e:
        log.warning("参考覆盖表读取失败：%r", e)
        return
    n = 0
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            w, t = v.get("wav"), v.get("txt")
        elif isinstance(v, (list, tuple)) and len(v) == 2:
            w, t = v
        else:
            continue
        if w and t and os.path.exists(w) and os.path.exists(t):
            VOICES[k] = (w, t)
            n += 1
            log.info("参考覆盖 %s -> %s", k, w)
        else:
            log.warning("参考覆盖 %s 缺文件，跳过", k)
    if n:
        log.info("已加载 %d 条参考覆盖", n)


def _save_prompt_overrides():
    try:
        d = os.path.dirname(PROMPT_JSON)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(PROMPT_JSON, "w", encoding="utf-8") as f:
            json.dump({k: {"wav": w, "txt": t} for k, (w, t) in VOICES.items()},
                      f, ensure_ascii=False, indent=1)
        log.info("参考覆盖表已写回 %s", PROMPT_JSON)
    except Exception as e:
        log.warning("参考覆盖表写入失败：%r", e)


def _resolve_voice(name):
    """名字命中就用它；否则按性别兜底（旧客户端只发 male/female）。

    ⚠️ 必须先判 female：'female' 里也含 'male'。
    """
    if name in VOICES:
        return name
    n = (name or "").strip().lower()
    for key in ("female", "male"):
        if key in n and key in VOICES:
            return key
    return None


_load_prompt_overrides()

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


# ---- 语种 / 文本规范化（P2，2026-10-21）--------------------------------
# 历史问题：omni 只收到 text/ref_audio/ref_text，**没有 language、也不做规范化**。
# 而 `TTS_KO_NUM=1`（服务里确实开着）只作用于 qwen 链的 pipeline.build_segments，
# omni 这条链从来没人管韩语数字 → 「几十万」「퍼센트」这类写法可能念错。
# 默认仍关闭（TTS_OMNI_LANG 空 = 不传 language，行为与 v3.8 一致），
# 请求体也可单独覆盖，方便不重启做 A/B。
LANG_DEFAULT = os.environ.get("TTS_OMNI_LANG", "").strip()
NORM_DEFAULT = os.environ.get("TTS_OMNI_NORM", "0").strip().lower() in (
    "1", "true", "yes", "on")


# ---- 解码参数覆盖（r59，2026-10-22）-------------------------------------
# OmniVoice 是掩码扩散模型，现役一直用默认解码参数：
#   num_step=32 / guidance_scale=2.0 / t_shift=0.1 / denoise=True / class_temperature=0（纯贪心）
# r59 扫描发现：denoise=True 会对干净的 EL 参考做过量平滑（HF 能量比 EL 低 19%）；
# class_temperature=0 走 log_probs.argmax()，输出最"平均化"。
# 此处提供环境级默认 + 请求级覆盖；**全部留空 = 行为与 v3.9 完全一致**。
_GEN_KEYS = {"denoise", "guidance_scale", "class_temperature", "num_step",
             "t_shift", "position_temperature", "layer_penalty_factor",
             "postprocess_output",
             # r65 分块拼接：块间插纯零是「卡音」根因，见文件尾注释
             "audio_chunk_duration", "audio_chunk_threshold",
             "pad_duration", "fade_duration"}


def _gen_env_default():
    """读 TTS_OMNI_DENOISE / CFG / CLASS_T / STEPS / TSHIFT（空 = 不覆盖）。"""
    return {
        "denoise": os.environ.get("TTS_OMNI_DENOISE", "").strip(),
        "guidance_scale": os.environ.get("TTS_OMNI_CFG", "").strip(),
        "class_temperature": os.environ.get("TTS_OMNI_CLASS_T", "").strip(),
        "num_step": os.environ.get("TTS_OMNI_STEPS", "").strip(),
        "t_shift": os.environ.get("TTS_OMNI_TSHIFT", "").strip(),
        # r65 分块拼接（空 = 用模型默认 15.0 / 30.0 / 0.1 / 0.1，行为不变）
        "audio_chunk_duration": os.environ.get("TTS_OMNI_CHUNK", "").strip(),
        "audio_chunk_threshold": os.environ.get("TTS_OMNI_CHUNK_TH", "").strip(),
        "pad_duration": os.environ.get("TTS_OMNI_PAD", "").strip(),
        "fade_duration": os.environ.get("TTS_OMNI_FADE", "").strip(),
    }


def _gen_overrides(req_over):
    """合并 环境默认 + 请求覆盖，返回可传给 OmniVoiceGenerationConfig 的 dict。"""
    raw = _gen_env_default()
    for k, v in (req_over or {}).items():
        if k in _GEN_KEYS:
            raw[k] = v
    out = {}
    for k, v in raw.items():
        if k not in _GEN_KEYS or v in ("", None):
            continue
        if isinstance(v, bool) or str(v).strip().lower() in ("true", "false"):
            out[k] = str(v).strip().lower() == "true"
        else:
            try:
                out[k] = int(v) if k == "num_step" else float(v)
            except (TypeError, ValueError):
                continue
    return out


class SynthReq(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = DEFAULT_VOICE
    rid: str = ""
    # 留空/None = 用环境默认值
    language: str = ""
    normalize: bool | None = None
    # r59：解码参数覆盖（可选，键见 _GEN_KEYS；不传 = 环境默认 = 行为不变）
    gen: dict | None = None


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
    voice = _resolve_voice(req.voice)
    if voice is None:
        raise HTTPException(400, f"未知音色 {req.voice}，可用：{sorted(VOICES)}")
    m = load_model()
    rw, rt = VOICES[voice]
    with open(rt, encoding="utf-8") as f:
        rtext = f.read().strip()
    t = time.time()
    try:
        kw = {}
        _lang = (req.language or LANG_DEFAULT).strip()
        if _lang:
            kw["language"] = _lang
        _norm = NORM_DEFAULT if req.normalize is None else bool(req.normalize)
        if _norm:
            kw["normalize_text"] = True
        _go = _gen_overrides(req.gen)
        if _go:
            from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
            kw["generation_config"] = OmniVoiceGenerationConfig(**_go)
        with _gen_lock:
            out = m.generate(text=req.text, ref_audio=rw, ref_text=rtext, **kw)
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
    log.info("synth voice=%s(%s) %d字 lang=%s norm=%s gen=%s -> %.1fs 用时 %.1fs RTF=%.3f",
             voice, req.voice, len(req.text), kw.get("language", "-"),
             kw.get("normalize_text", False), _go or "-", dur, time.time() - t,
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
    _save_prompt_overrides()
    log.info("参考已切换 %s -> %s（已持久化）", req.voice, req.wav)
    return {"ok": True, "voices": sorted(VOICES), "persisted": PROMPT_JSON}


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
