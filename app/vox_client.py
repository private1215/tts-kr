"""tts-kr 调用 VoxCPM2 独立服务（8400，男声最佳链路）的客户端。

为什么走 HTTP：VoxCPM2 自带 venv（torch 2.14+cu130 / numpy 2.5），
与 Qwen3-TTS 主服务无法共存于同一进程，只能独立进程 + HTTP。

🔴 分块由服务端负责，客户端不要自己切：
   VoxCPM2 整篇一次提交会「静默截断」（3948 字只出 218.4s 就正常返回，
   比报错更危险），服务端已强制按句边界 600 字分块，与获胜的 X1_m 一致。
   客户端再切一遍只会多出拼接点、破坏韵律。

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

VOX_URL = os.environ.get("TTS_VOX_URL", "http://127.0.0.1:8400")
# RTF 实测 0.53~0.72（48k 采样），一篇 4500 字约 300s；给足余量
VOX_TIMEOUT = float(os.environ.get("TTS_VOX_TIMEOUT", "2400"))
DEFAULT_VOICE = os.environ.get("TTS_VOX_VOICE", "male")


def health(timeout: float = 5.0):
    try:
        r = urllib.request.urlopen(VOX_URL + "/health", timeout=timeout)
        return True, json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return False, {"error": repr(e)[:200]}


def _poll_progress(rid, stop, on_progress, interval=0.6):
    """VoxCPM2 是分块生成的，进度是真实分块进度（不像 Qwen 只能假报）。"""
    url = VOX_URL + "/progress?" + urllib.parse.urlencode({"rid": rid})
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
    meta = {"engine": "voxcpm", "error": None}
    if not text or not text.strip():
        meta["error"] = "文本为空"
        return None, 0, meta

    rid = uuid.uuid4().hex[:12]
    body = json.dumps({"text": text,
                       "voice": voice or DEFAULT_VOICE,
                       "rid": rid}).encode("utf-8")
    req = urllib.request.Request(VOX_URL + "/synth", data=body,
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
        resp = urllib.request.urlopen(req, timeout=float(timeout or VOX_TIMEOUT))
        raw = resp.read()
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8")[:300]
        except Exception:
            detail = ""
        meta["error"] = f"HTTP {e.code}: {detail}"
        log.warning("VoxCPM2 调用失败: %s", meta["error"])
        stop.set()
        return None, 0, meta
    except Exception as e:
        meta["error"] = repr(e)[:200]
        log.warning("VoxCPM2 调用失败: %s", meta["error"])
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
        req = urllib.request.Request(VOX_URL + "/unload", data=b"{}",
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except Exception as e:
        return {"ok": False, "error": repr(e)[:200]}
