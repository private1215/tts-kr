"""tts-kr 调用 OmniVoice 独立服务（8300，女声最佳链路）的客户端。

为什么走 HTTP：OmniVoice 自带 venv（torch 2.14+cu130 / numpy 2.5），
与 Qwen3-TTS 主服务无法共存于同一进程，只能独立进程 + HTTP。

⚠️ 与 CosyVoice3 客户端的关键差别：**不分块**。
   女声冠军 X2_f.wav 就是整篇一次提交（3911 字 -> 513s），OmniVoice 长文稳定；
   分块会改变韵律，等于换方案。所以这里把整篇文本一次送过去。

服务不可用时返回 (None, 0, meta)，调用方回退到原引擎，员工不受影响。
"""
import os
import io
import json
import uuid
import logging
import threading
import urllib.parse
import urllib.request
import urllib.error

import numpy as np
import soundfile as sf

log = logging.getLogger("tts")

OMNI_URL = os.environ.get("TTS_OMNI_URL", "http://127.0.0.1:8300")
# OmniVoice RTF 实测 0.16~0.19，一篇 4000 字约 100s；给足余量
OMNI_TIMEOUT = float(os.environ.get("TTS_OMNI_TIMEOUT", "1800"))
DEFAULT_VOICE = os.environ.get("TTS_OMNI_VOICE", "female")


def health(timeout: float = 5.0):
    try:
        r = urllib.request.urlopen(OMNI_URL + "/health", timeout=timeout)
        return True, json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return False, {"error": repr(e)[:200]}


def _poll_progress(rid, stop, on_progress, interval=0.6):
    url = OMNI_URL + "/progress?" + urllib.parse.urlencode({"rid": rid})
    last = None
    while not stop.is_set():
        stop.wait(interval)
        if stop.is_set():
            break
        try:
            r = urllib.request.urlopen(url, timeout=3.0)
            p = json.loads(r.read().decode("utf-8"))
        except Exception:
            continue
        if not p.get("found"):
            continue
        cur = (p.get("done"), p.get("total"))
        if cur != last and cur[1]:
            last = cur
            try:
                on_progress(int(cur[0]), int(cur[1]))
            except Exception:
                pass


def synth(text, voice=None, timeout=None, on_progress=None):
    """返回 (audio:np.float32|None, sr:int, meta:dict)。"""
    meta = {"engine": "omnivoice", "error": None}
    if not text or not text.strip():
        meta["error"] = "文本为空"
        return None, 0, meta

    rid = uuid.uuid4().hex[:12]
    body = json.dumps({"text": text,
                       "voice": voice or DEFAULT_VOICE,
                       "rid": rid}).encode("utf-8")
    req = urllib.request.Request(OMNI_URL + "/synth", data=body,
                                 headers={"Content-Type": "application/json"})
    stop = threading.Event()
    poller = None
    if on_progress:
        poller = threading.Thread(target=_poll_progress,
                                  args=(rid, stop, on_progress), daemon=True)
        poller.start()
    import time as _t
    t0 = _t.time()
    try:
        resp = urllib.request.urlopen(req, timeout=float(timeout or OMNI_TIMEOUT))
        raw = resp.read()
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8")[:300]
        except Exception:
            detail = ""
        meta["error"] = f"HTTP {e.code}: {detail}"
        log.warning("OmniVoice 调用失败: %s", meta["error"])
        stop.set()
        return None, 0, meta
    except Exception as e:
        meta["error"] = repr(e)[:200]
        log.warning("OmniVoice 调用失败: %s", meta["error"])
        stop.set()
        return None, 0, meta

    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as e:
        meta["error"] = f"wav 解析失败: {repr(e)[:200]}"
        stop.set()
        return None, 0, meta
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    meta["gen_seconds"] = round(_t.time() - t0, 2)
    meta["audio_seconds"] = round(len(audio) / sr, 2)
    meta["chars"] = len(text)
    meta["voice"] = voice or DEFAULT_VOICE
    for k in ("X-Audio-Seconds", "X-Chunks"):
        v = resp.headers.get(k)
        if v:
            meta[k[2:].lower()] = float(v) if "." in v else int(v)
    stop.set()
    if poller:
        poller.join(timeout=2.0)
    return audio, sr, meta


def unload(timeout: float = 10.0):
    try:
        req = urllib.request.Request(OMNI_URL + "/unload", data=b"{}",
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except Exception as e:
        return {"ok": False, "error": repr(e)[:200]}
