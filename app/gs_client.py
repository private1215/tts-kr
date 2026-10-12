# -*- coding: utf-8 -*-
"""GPT-SoVITS 服务客户端（对齐 cosy_client 的用法与降级约定）。

🔴 为什么走 HTTP：GS 与 Qwen 是两套不兼容的 venv，见 deploy/gs_serve.py 文件头。

设计原则（与 cosy_client 一致）：
    1. 任何网络异常都**转成 (ok=False, ...) 元组**，绝不让调用方炸；
       引擎选型失败要能自动回落到 Qwen，不能让员工看到 500。
    2. 超时给得足够长（长文本分段推理本来就慢）。
    3. 不引入新的第三方依赖（只用 urllib）。
"""
import os
import json
import urllib.request
import urllib.parse
import urllib.error

GS_URL = os.environ.get("TTS_GS_URL", "http://127.0.0.1:8200")
GS_TIMEOUT = float(os.environ.get("TTS_GS_TIMEOUT", "2400"))


def health(timeout: float = 5.0):
    """返回 (ok, detail)。"""
    try:
        r = urllib.request.urlopen(GS_URL + "/health", timeout=timeout)
        return True, json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return False, repr(e)[:160]


def synth(text: str, ref_wav: str, ref_text: str, spk: str = "SOLA",
          seed: int = 42, seg: int = 50, timeout: float = None):
    """调用 GS 合成。

    返回 (ok, payload)：
        ok=True  → payload = wav bytes
        ok=False → payload = 错误说明字符串
    """
    if not text.strip():
        return False, "文本为空"
    if not ref_wav or not os.path.exists(ref_wav):
        return False, "参考音频不存在: %s" % ref_wav
    body = urllib.parse.urlencode({
        "text": text,
        "spk": spk,
        "ref_wav": ref_wav,
        "ref_text": ref_text or "",
        "seed": int(seed),
        "seg": int(seg),
    }).encode("utf-8")
    req = urllib.request.Request(
        GS_URL + "/synth", data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        resp = urllib.request.urlopen(
            req, timeout=float(timeout or GS_TIMEOUT))
        return True, resp.read()
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "ignore")[:200]
        except Exception:
            detail = ""
        return False, "GS %s %s" % (e.code, detail)
    except Exception as e:
        return False, repr(e)[:200]


def unload(timeout: float = 10.0):
    try:
        req = urllib.request.Request(GS_URL + "/unload", data=b"{}",
                                  headers={"Content-Type":
                                           "application/json"})
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False
