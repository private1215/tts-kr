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
import uuid
import logging
import threading
import urllib.parse

import numpy as np
import soundfile as sf
import urllib.request
import urllib.error

log = logging.getLogger("tts")

COSY_URL = os.environ.get("TTS_COSY_URL", "http://127.0.0.1:8100")
# 单块生成 RTF 约 1.2，一篇 3700 字约需 10 分钟，超时给足
COSY_TIMEOUT = float(os.environ.get("TTS_COSY_TIMEOUT", "2400"))
# 单块最大字数。实测（同文本 776 字韩语）块越长 ASR 字错率越低、速度也越快：
#   80→CER 0.0892/RTF 1.276 | 160→0.0829/1.075 | 240→0.0850/1.229 | 320→0.0727/1.057
# 最初定 80 是照抄 Qwen 的「短段保 shipment」经验，被数据推翻。
COSY_MAX_CHARS = int(os.environ.get("TTS_COSY_MAX_CHARS", "320"))


def health(timeout: float = 5.0):
    """探测 CosyVoice3 服务是否可用。返回 (ok, info)。"""
    try:
        r = urllib.request.urlopen(COSY_URL + "/health", timeout=timeout)
        return True, json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return False, {"error": repr(e)[:200]}


def _poll_progress(rid: str, stop: threading.Event, on_progress, interval=0.6):
    """后台线程：轮询 CosyVoice3 的 /progress，把分段进度回传给调用方。

    为什么需要线程：/synth 是一次性阻塞请求，返回前拿不到任何中间进度，
    而 tts-kr 的 job["progress"] 必须持续更新，否则前端进度条会一直停在 0%
    （2026-09-29 实测：走了 CosyVoice3 的任务几十秒都显示「正在分段 · 0%」）。
    """
    url = COSY_URL + "/progress?" + urllib.parse.urlencode({"rid": rid})
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


def synth(text: str, ref_wav: str, ref_text: str,
          instruct: str = "neutral", speed: float = 1.0,
          seed: int = 42, max_chars: int = None, timeout: float = None,
          on_progress=None, spk_id: str = ""):
    """合成一段文本。

    spk_id: CosyVoice3 里**已注册**的常驻音色 id（POST /voices 注册过）。
            给了它就走缓存复用通道，**不需要** ref_wav；
            为空则走零样本克隆（必须给 ref_wav）。

    on_progress: 可选回调 (done:int, total:int)，合成期间由后台线程按分段进度调用。

    返回 (audio:np.float32|None, sr:int, meta:dict)
    audio 为 None 表示失败，meta["error"] 写明原因。
    """
    meta = {"engine": "cosyvoice3", "error": None}
    if not text or not text.strip():
        meta["error"] = "文本为空"
        return None, 0, meta
    # 常驻音色走缓存复用，不必再传参考音频；零样本才校验 ref_wav
    if not spk_id and (not ref_wav or not os.path.exists(ref_wav)):
        meta["error"] = f"参考音频不存在: {ref_wav}"
        return None, 0, meta

    rid = uuid.uuid4().hex[:12]
    body = json.dumps({
        "text": text,
        "spk_id": spk_id or "",
        "ref_wav": ref_wav or "",
        "ref_text": ref_text or "",
        "instruct": instruct or "neutral",
        "speed": float(speed or 1.0),
        "seed": int(seed or 42),
        "max_chars": int(max_chars or COSY_MAX_CHARS),
        "rid": rid,
    }).encode("utf-8")
    req = urllib.request.Request(
        COSY_URL + "/synth", data=body,
        headers={"Content-Type": "application/json"})

    # 进度回传线程：on_progress 为空就不启（省一次轮询开销）
    stop = threading.Event()
    poller = None
    if on_progress:
        poller = threading.Thread(target=_poll_progress,
                                  args=(rid, stop, on_progress),
                                  daemon=True)
        poller.start()

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
        stop.set()
        return None, 0, meta
    except Exception as e:
        meta["error"] = repr(e)[:200]
        log.warning("CosyVoice3 调用失败: %s", meta["error"])
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

    meta["gen_seconds"] = round(__import__("time").time() - t0, 2)
    meta["audio_seconds"] = round(len(audio) / sr, 2)
    meta["chars"] = len(text)
    for k in ("X-Audio-Seconds", "X-Gen-Seconds", "X-Chunks"):
        v = resp.headers.get(k)
        if v:
            meta[k[2:].lower()] = float(v) if "." in v else int(v)
    stop.set()
    if poller:
        poller.join(timeout=2.0)
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
