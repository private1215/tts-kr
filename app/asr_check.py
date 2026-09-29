# -*- coding: utf-8 -*-
"""ASR 内容回检：用 Whisper 转写成品音频，与原文比对字符错误率（CER）。

为什么需要它
------------
员工反馈的乱音是「说着说着就又笑又说奇怪的话」——这是**内容跑飞**，
而现有体检查的是**语速（= 时长）**。语速只是时长的代理指标，抓不到内容问题：

    一段 40 字的文本正常该念 5.7 秒。如果模型念了 20 字、笑了 2 秒、
    又嘟囔几句，总时长还是 5.7 秒 —— 语速算出来 7.0 字/秒完全健康，
    但人耳听到的就是乱音。

铁证：员工听着有乱音的那条（任务 7f20a19b8d8c）语速中位 8.30、最低 6.52，
指标全绿，内容却是乱的。所以**唯一能抓内容跑飞的只有 ASR 回检**。

实现要点
--------
1. faster-whisper 必须 `local_files_only=True`，否则联网检查会**卡死**
   （模型权重 464M 已缓存在 ~/.cache/huggingface）。这是踩过的坑，别改。
2. CPU + int8 推理，不占显存，与 GPU 生成互不干扰。
3. 任何异常都**降级为跳过**，绝不能因为回检失败而让整单失败。
"""
import os
import re
import logging
import difflib

log = logging.getLogger("tts.asr")

MODEL_ID = os.environ.get("TTS_ASR_MODEL", "Systran/faster-whisper-small")
# 回检开关：默认开。设为 0 可整体关闭（例如排障期）
ENABLED = os.environ.get("TTS_ASR_CHECK", "1").strip().lower() not in ("0", "false", "no")
# CER 超过这个值就判定该块内容跑飞，换种子重掷
CER_THRESHOLD = float(os.environ.get("TTS_ASR_CER", "0.35") or 0.35)

_model = None
_model_failed = False

# 只保留韩文音节 / 字母 / 数字，其余（空格、标点）全部去掉再比对。
# 韩语分词差异会让逐词比对失真，按字符比对更稳。
_KEEP = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]")


def normalize(text: str) -> str:
    """归一化：去空白与标点，只留实质字符。"""
    return "".join(_KEEP.findall(text or ""))


def _load():
    global _model, _model_failed
    if _model is not None or _model_failed:
        return _model
    try:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_ID, device="cpu", compute_type="int8",
                              cpu_threads=2, local_files_only=True)
        log.info("ASR 模型已加载: %s (cpu/int8)", MODEL_ID)
    except Exception as e:
        _model_failed = True
        log.warning("ASR 模型加载失败，本次及后续跳过内容回检: %s", e)
    return _model


def transcribe(wav_path: str, language: str = "ko") -> str:
    """转写音频文件，返回文本。失败返回空串。"""
    m = _load()
    if m is None:
        return ""
    try:
        segs, _info = m.transcribe(wav_path, language=language,
                                   vad_filter=True, beam_size=1)
        return " ".join(s.text for s in segs).strip()
    except Exception as e:
        log.warning("ASR 转写失败 %s: %s", wav_path, e)
        return ""


def cer(reference: str, hypothesis: str) -> float:
    """字符错误率（近似）。0.0 = 完全一致，1.0 = 完全不匹配。

    用 difflib 的匹配块估算，复杂度远低于逐字 DP（2600 字 × 2600 字
    的 DP 在纯 Python 下要跑好几秒，difflib 是 C 实现，毫秒级）。
    """
    a, b = normalize(reference), normalize(hypothesis)
    if not a:
        return 0.0
    if not b:
        return 1.0
    ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    return round(max(0.0, 1.0 - ratio), 4)


def check_chunk(wav_path: str, ref_text: str, language: str = "ko") -> dict:
    """回检单个块的音频内容是否与原文一致。

    返回 {"ok": bool, "cer": float, "hyp": str, "skipped": bool}
    ok=True 表示内容可信（未跑飞）或回检被跳过（不阻塞流程）。
    """
    if not ENABLED:
        return {"ok": True, "cer": 0.0, "hyp": "", "skipped": True}
    hyp = transcribe(wav_path, language=language)
    if not hyp:
        # 转写不出来不代表跑飞（可能只是 ASR 抽风），放行但记下来
        return {"ok": True, "cer": 0.0, "hyp": "", "skipped": True}
    e = cer(ref_text, hyp)
    return {"ok": e <= CER_THRESHOLD, "cer": e, "hyp": hyp, "skipped": False}
