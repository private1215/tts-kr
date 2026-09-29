"""tts-kr 调用 CosyVoice3 独立服务的客户端。

为什么要走 HTTP 而不是直接 import：
    CosyVoice3 锁 torch==2.3.1+cu121 / numpy==1.26.4，与现网 Qwen3-TTS 的
    torch 2.14+cu130 冲突，两个引擎只能各占一个进程。所以这里只做薄薄一层
    HTTP 封装，让 server.py 用起来跟本地函数差不多。

服务未启动或超时时返回 (None, reason)，调用方据此回退到本地 Qwen 引擎，
保证「CosyVoice 挂了也不影响员工正常出活」。
"""
import os
import io
import json
import logging

import numpy as np
import soundfile as sf
import urllib.request
import urllib.error

log = logging.getLogger("tts")

COSY_URL = os.environ.get("TTS_COSY_URL", "http://127.0.0.1:8100")
# 单块生成 RTF 约 1.2，一篇 3700 字约需 10 分钟，超时给足
COSY_TIMEOUT = float(os.environ.get("TTS_COSY_TIMEOUT", "2400"))
# 单块最大字数：实测块越长越容易念串/幻觉（CER 显著变差）
COSY_MAX_CHARS = int(os.environ.get("TTS_COSY_MAX_CHARS", "80"))


def health(timeout: float = 5.0):
    """探测 CosyVoice3 服务是否可用。返回 (ok, info)。"""
    try:
        r = urllib.request.urlopen(COSY_URL + "/health", timeout=timeout)
        return True, json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return False, {"error": repr(e)[:200]}


def synth(text: str, ref_wav: str, ref_text: str,
          instruct: str = "neutral", speed: float = 1.0,
          seed: int = 42, max_chars: int = None, timeout: float = None):
    """合成一段文本。

    返回 (audio:np.float32|None, sr:int, meta:dict)
    audio 为 None 表示失败，meta["error"] 写明原因。
    """
    meta = {"engine": "cosyvoice3", "error": None}
    if not text or not text.strip():
        meta["error"] = "文本为空"
        return None, 0, meta
    if not ref_wav or not os.path.exists(ref_wav):
        meta["error"] = f"参考音频不存在: {ref_wav}"
        return None, 0, meta

    body = json.dumps({
        "text": text,
        "ref_wav": ref_wav,
        "ref_text": ref_text or "",
        "instruct": instruct or "neutral",
        "speed": float(speed or 1.0),
        "seed": int(seed or 42),
        "max_chars": int(max_chars or COSY_MAX_CHARS),
    }).encode("utf-8")
    req = urllib.request.Request(
        COSY_URL + "/synth", data=body,
        headers={"Content-Type": "application/json"})

    t0 = __import__("time").time()
    try:
        resp = urllib.request.urlopen(
            req, timeout=float(timeout or COSY_TIMEOUT))
        raw = resp.read()
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8")[:300]
        except Exception:
            detail = ""
        meta["error"] = f"HTTP {e.code}: {detail}"
        log.warning("CosyVoice3 调用失败: %s", meta["error"])
        return None, 0, meta
    except Exception as e:
        meta["error"] = repr(e)[:200]
        log.warning("CosyVoice3 调用失败: %s", meta["error"])
        return None, 0, meta

    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as e:
        meta["error"] = f"wav 解析失败: {repr(e)[:200]}"
        return None, 0, meta
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    meta["gen_seconds"] = round(__import__("time").time() - t0, 2)
    meta["audio_seconds"] = round(len(audio) / sr, 2)
    meta["chars"] = len(text)
    for k in ("X-Audio-Seconds", "X-Gen-Seconds", "X-Chunks"):
        v = resp.headers.get(k)
        if v:
            meta[k[2:].lower()] = float(v) if "." in v else int(v)
    return audio, sr, meta


def unload(timeout: float = 10.0):
    """让 CosyVoice3 卸载模型、把显存让回给 Qwen。"""
    try:
        req = urllib.request.Request(COSY_URL + "/unload", data=b"{}",
                                     headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": repr(e)[:200]}
