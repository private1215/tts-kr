# -*- coding: utf-8 -*-
"""音色级「引擎 + 情感」自动选型。

设计目标：员工只选「音色 + 语速」，引擎和情感一律由后端决定。

为什么能自动：
  「哪个引擎更像这个人、哪种情感更合适」是**音色绑定的属性**，
  一次测完永久有效，不需要每次生成都重新判断。所以放在「新增音色」时做。

评分口径（两项都要，缺一不可）：
  1) CER（韩语 ASR 字错率）—— 念错字是硬伤，低于阈值才及格
  2) 句内音高动态（半音）—— 衡量「有没有声调」，越大越不平淡

  规则：先按 CER 过滤掉不合格的候选，再在合格候选里选音高动态最高的。
  这样既保证念得对，又尽量不平淡。

候选集合来自 server.EMOTION_PRESETS（引擎 + 情感组合）。
"""
import os
import re
import time
import logging
import tempfile

import numpy as np
import soundfile as sf

log = logging.getLogger("tts")

# 试听文本：一段中性的韩语股东播报体，足够长才能测出语调和可懂度
PREVIEW_TEXT = (
    "안녕하세요. 주주 여러분, 반갑습니다. "
    "오늘 시장은 생각보다 훨씬 더 강한 흐름을 보여주고 있습니다. "
    "외인 창구에서는 순매도를 기록하고 있지만, 개인과 기관은 매집 흐름을 이어가고 있습니다. "
    "지금 구간에서는 절대 섣불리 판단하시면 안 됩니다."
)

# CER 及格线：超过就认为念错太多，直接淘汰
CER_PASS = float(os.environ.get("TTS_SELECT_CER_PASS", "0.13"))
# 打分权重：CER 越低越好，音高动态越高越好
W_CER = float(os.environ.get("TTS_SELECT_W_CER", "10.0"))
W_DYN = float(os.environ.get("TTS_SELECT_W_DYN", "1.0"))


def _sentence_windows(x, sr, min_len=0.6, max_len=30.0, min_sil=0.25):
    """按静音把音频切成句段。"""
    hop = int(0.02 * sr)
    win = int(0.05 * sr)
    n = (len(x) - win) // hop
    if n <= 10:
        return []
    idx = np.arange(n) * hop
    rms = np.sqrt(np.array([np.mean(x[i:i + win] ** 2) for i in idx]) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    sil = db < (np.percentile(db, 25) + 6.0)
    cuts, i = [], 0
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]:
                j += 1
            if (j - i) * hop / sr >= min_sil:
                cuts.append((i * hop, j * hop))
            i = j
        else:
            i += 1
    bnd = [0]
    for a, b in cuts:
        bnd += [a, b]
    bnd.append(len(x))
    out = []
    for k in range(0, len(bnd) - 1, 2):
        s, e = bnd[k], bnd[k + 1]
        if min_len <= (e - s) / sr <= max_len:
            out.append((s, e))
    return out


def prosody_score(path):
    """句内音高动态（半音）：所有句段 F0 的 p15~p85 宽度取中位。

    抓不到就返回 None（不参与打分，不误判）。
    """
    try:
        import librosa
    except Exception:
        return None
    try:
        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if len(x) < sr:
            return None
        segs = _sentence_windows(x, sr)
        if not segs:
            segs = [(0, len(x))]
        dyn = []
        for s, e in segs[:60]:
            y = x[s:e].astype(np.float64)
            f0, _, _ = librosa.pyin(y, fmin=60, fmax=600, sr=sr,
                                    frame_length=2048, hop_length=256)
            if f0 is None:
                continue
            f = f0[~np.isnan(f0)]
            if len(f) < 8:
                continue
            dyn.append(12 * np.log2(np.percentile(f, 85) / np.percentile(f, 15) + 1e-12))
        if len(dyn) < 2:
            return None
        return float(np.median(dyn))
    except Exception as e:
        log.warning("音高动态计算失败: %s", repr(e)[:120])
        return None


def _cer(wav_path, text, language="Korean"):
    import asr_check
    try:
        r = asr_check.check_chunk(wav_path, text, language=language)
    except Exception as e:
        log.warning("ASR 失败: %s", repr(e)[:120])
        return None
    if r.get("skipped"):
        return None
    return r.get("cer")


def benchmark_voice(ref_wav, ref_text, language="Korean",
                    presets=None, text=None, tmp_dir=None):
    """对一组「引擎+情感」候选各跑一次，返回打分后的排序结果。

    presets: [{"key":..,"engine":..,"instruct":..}, ...]
    返回 [{"key","engine","instruct","cer","dyn","score","ok"}, ...]（按 score 降序）
    """
    from server import EMOTION_PRESETS
    presets = presets or [
        dict(key=k, engine=v["engine"], instruct=v.get("instruct") or "")
        for k, v in EMOTION_PRESETS.items()
    ]
    text = text or PREVIEW_TEXT
    tmp_dir = tmp_dir or tempfile.gettempdir()
    results = []

    import cosy_client
    for p in presets:
        t0 = time.time()
        wav = os.path.join(tmp_dir, f".sel_{p['key']}_{int(t0)}.wav")
        try:
            if p["engine"] == "cosyvoice3":
                audio, sr, meta = cosy_client.synth(
                    text=text, ref_wav=ref_wav, ref_text=ref_text or "",
                    instruct=p["instruct"] or "neutral")
                if audio is None:
                    log.warning("候选 %s 生成失败: %s", p["key"], meta.get("error"))
                    results.append(dict(p, cer=None, dyn=None, score=None, ok=False,
                                        error=(meta.get("error") or "")[:120]))
                    continue
                sf.write(wav, audio, sr)
            else:
                # Qwen 走本地引擎
                from tts_engine import TTSEngine
                eng = TTSEngine()
                _a, _ = sf.read(ref_wav, dtype="float32")
                if _a.ndim > 1:
                    _a = _a.mean(axis=1)
                audio, sr, _st = eng.synth_long(
                    text=text, mode="clone", language=language, speaker=None,
                    instruct="", ref_audio=_a, ref_text=ref_text,
                    max_chars=60, batch_size=8, seed=42)
                sf.write(wav, audio, sr)
        except Exception as e:
            log.warning("候选 %s 异常: %s", p["key"], repr(e)[:160])
            results.append(dict(p, cer=None, dyn=None, score=None, ok=False,
                                error=repr(e)[:120]))
            continue

        cer = _cer(wav, text, language)
        dyn = prosody_score(wav)
        try:
            os.remove(wav)
        except OSError:
            pass
        ok = (cer is not None) and (cer <= CER_PASS)
        score = None
        if cer is not None:
            score = -W_CER * cer + W_DYN * (dyn or 0.0)
        results.append(dict(p, cer=cer, dyn=dyn, score=score, ok=ok,
                            seconds=round(time.time() - t0, 1)))
        log.info("选型候选 %-9s 引擎=%-10s CER=%s 动态=%s score=%s",
                 p["key"], p["engine"],
                 "None" if cer is None else round(cer, 4),
                 "None" if dyn is None else round(dyn, 2),
                 "None" if score is None else round(score, 3))

    results.sort(key=lambda r: (r.get("score") is not None,
                                r.get("score") or -1e9), reverse=True)
    return results


def pick_best(results):
    """在及格候选里挑分数最高的；全部不及格就返回 CER 最低的那个。"""
    if not results:
        return None
    ok = [r for r in results if r.get("ok")]
    if ok:
        return max(ok, key=lambda r: r.get("score") or -1e9)
    withcer = [r for r in results if r.get("cer") is not None]
    if withcer:
        return min(withcer, key=lambda r: r["cer"])
    return results[0]
