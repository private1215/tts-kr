# -*- coding: utf-8 -*-
"""Qwen3-TTS 引擎封装：单实例、按需热切换模型、支持批量分段生成。"""
import os
import re
import time
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from pipeline import (build_segments, concat_segments, apply_speed_jitter,
                      trim_edges, prosody_factors, apply_gain)

log = logging.getLogger("tts")

# 语气指令为空时，模型会自由发挥，实测容易出现语速失控（越往后越慢、长拖音）。
# 这里按语言兜底一个中性指令，保证语速稳定。
DEFAULT_INSTRUCT = {
    "chinese": "用自然、平稳的语速清晰朗读",
    "korean": "자연스럽고 일정한 속도로 또렷하게 읽어주세요",
    "english": "Read clearly at a natural, steady pace",
    "japanese": "自然で一定の速さではっきりと読んでください",
    "german": "Lesen Sie in natürlichem, gleichmäßigem Tempo",
    "french": "Lisez à un rythme naturel et régulier",
    "spanish": "Lea a un ritmo natural y constante",
    "portuguese": "Leia em ritmo natural e constante",
    "russian": "Читайте в естественном, ровном темпе",
    "italian": "Leggi a un ritmo naturale e costante",
    "dutch": "Lees in een natuurlijk, gelijkmatig tempo",
}

# 低于此字/秒视为"语速异常慢"（正常中文播报约 4~5 字/秒）
SLOW_CPS = 2.6


class TTSEngine:
    """管理 Qwen3-TTS 模型。

    显存有限(12G)，同一时刻只加载一个模型变体；
    若任务需要另一个变体则按需热切换。
    """

    def __init__(self, model_dirs: dict, device: str = "cuda:0",
                 dtype: str = "bfloat16", default_variant: str = "custom_voice"):
        self.model_dirs = model_dirs
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.default_variant = default_variant
        self.model = None
        self.variant = None
        self.load_time = 0.0
        self._clone_prompt_cache = {}
        self._languages = None
        self._speakers = None

    def _load(self, variant: str):
        from qwen_tts import Qwen3TTSModel
        if self.variant == variant and self.model is not None:
            return
        if variant not in self.model_dirs:
            raise ValueError(f"未知模型变体: {variant}, 可用: {list(self.model_dirs)}")

        log.info("加载模型变体: %s (%s)", variant, self.model_dirs[variant])
        t0 = time.time()
        # 释放旧模型
        if self.model is not None:
            del self.model
            self.model = None
            self._clone_prompt_cache.clear()
            torch.cuda.empty_cache()

        kwargs = dict(
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        # attention 后端优先级：flash-attn > sdpa > 默认。
        # 没装 flash-attn 时 sdpa(PyTorch 原生) 也比默认 eager 快得多，
        # 且在 Ampere(sm_86) 上稳定。
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
            log.info("使用 flash-attention-2")
        except Exception:
            try:
                import torch.nn.functional as F  # noqa: F401
                if hasattr(F, "scaled_dot_product_attention"):
                    kwargs["attn_implementation"] = "sdpa"
                    log.info("flash-attn 未安装，使用 sdpa(PyTorch 原生)")
            except Exception:
                pass

        self.model = Qwen3TTSModel.from_pretrained(self.model_dirs[variant], **kwargs)
        self.variant = variant
        self.load_time = time.time() - t0
        self._languages = None
        self._speakers = None
        log.info("模型加载完成，耗时 %.1fs", self.load_time)
        try:
            self._languages = self.model.get_supported_languages()
        except Exception as e:
            log.warning("读取支持语言失败: %s", e)
        try:
            self._speakers = self.model.get_supported_speakers()
        except Exception as e:
            log.warning("读取支持音色失败: %s", e)

    @property
    def languages(self) -> List[str]:
        return self._languages or ["Korean", "Chinese", "English", "Japanese"]

    @property
    def speakers(self) -> List[str]:
        return self._speakers or []

    def warmup(self):
        self._load(self.default_variant)

    def _ensure_variant(self, mode: str):
        variant = "base" if mode == "clone" else "custom_voice"
        if variant not in self.model_dirs:
            variant = self.default_variant
        self._load(variant)

    def _get_clone_prompt(self, ref_audio: Tuple[np.ndarray, int], ref_text: Optional[str],
                          cache_key: str):
        if cache_key in self._clone_prompt_cache:
            return self._clone_prompt_cache[cache_key]
        prompts = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=False)
        item = prompts[0] if isinstance(prompts, list) else prompts
        self._clone_prompt_cache[cache_key] = item
        return item

    @staticmethod
    def _expand_instruct(instruct, n: int) -> List[str]:
        """把语气指令摊成 n 份。传列表则循环复用，传字符串则每段相同。"""
        if isinstance(instruct, (list, tuple)):
            arr = [str(x) for x in instruct if x is not None]
            if not arr:
                return [""] * n
            return [arr[i % len(arr)] for i in range(n)]
        return [instruct or ""] * n

    @staticmethod
    def _set_seed(seed: Optional[int]):
        """固定随机状态。分段生成时每段独立采样，音色/韵律会漂移，
        固定种子能显著提升长音频的前后一致性。"""
        if seed is None:
            return
        import random as _rnd
        _rnd.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def synth_batch(self, texts: List[str], mode: str = "custom",
                    language: str = "Korean", speaker: Optional[str] = None,
                    instruct=None,
                    ref_audio: Optional[Tuple[np.ndarray, int]] = None,
                    ref_text: Optional[str] = None,
                    ref_key: Optional[str] = None,
                    seed: Optional[int] = None,
                    temperature: Optional[float] = None,
                    top_p: Optional[float] = None,
                    repetition_penalty: Optional[float] = None,
                    max_new_tokens: Optional[int] = None,
                    ) -> Tuple[List[np.ndarray], int]:
        """批量生成一批文本片段。返回 (音频数组列表, 采样率)。

        instruct 可以是字符串（每段同语气）或列表（每段不同语气）。
        temperature/top_p/repetition_penalty 通过 kwargs 透传给底层 generate，
        若当前版本不支持会自动降级忽略。
        """
        self._ensure_variant(mode)
        n = len(texts)
        langs = [language] * n
        gen_kw = {}
        if temperature is not None:
            gen_kw["temperature"] = float(temperature)
        if top_p is not None:
            gen_kw["top_p"] = float(top_p)
        if repetition_penalty is not None:
            gen_kw["repetition_penalty"] = float(repetition_penalty)
        # max_new_tokens 上限：默认 2048 给了模型巨大的"自由发挥"空间，
        # 实测会出现越读越慢、尾部拖长音的失控（某段 70 字生成了 45 秒）。
        # 按文本长度估算一个宽松上限（约正常时长的 2.5 倍），既不会截断，
        # 又能挡住失控的长生成，同时明显加快生成。
        # 12Hz 模型约 12.5 tokens/秒；实测中文约 2.2~3.3 tokens/字。
        if max_new_tokens is None:
            c = max((len(t) for t in texts), default=0)
            max_new_tokens = min(2048, int(c * 6.0) + 96)
        gen_kw["max_new_tokens"] = int(max_new_tokens)

        def _call(fn, **kw):
            """带随机种子的调用；不支持的采样参数自动降级重试。"""
            self._set_seed(seed)
            kw.update(gen_kw)
            try:
                return fn(**kw)
            except TypeError as e:
                if not gen_kw:
                    raise
                log.warning("采样参数不被支持，降级忽略: %s", e)
                for k in list(gen_kw):
                    kw.pop(k, None)
                self._set_seed(seed)
                return fn(**kw)

        if mode == "clone":
            if ref_audio is None:
                raise ValueError("克隆模式必须提供参考音频")
            prompt = self._get_clone_prompt(ref_audio, ref_text, ref_key or "default")
            wavs, sr = _call(
                self.model.generate_voice_clone,
                text=texts, language=langs,
                voice_clone_prompt=[prompt] * n,
                non_streaming_mode=True,
            )
        else:
            spk = speaker or (self.speakers[0] if self.speakers else "Sohee")
            ins = self._expand_instruct(instruct, n)
            wavs, sr = _call(
                self.model.generate_custom_voice,
                text=texts, speaker=[spk] * n, language=langs,
                instruct=ins, non_streaming_mode=True,
            )
        return list(wavs), sr

    def synth_long(self, text: str, mode: str = "custom", language: str = "Korean",
                   speaker: Optional[str] = None, instruct: Optional[str] = None,
                   ref_audio: Optional[Tuple[np.ndarray, int]] = None,
                   ref_text: Optional[str] = None, ref_key: Optional[str] = None,
                   max_chars: int = 60, batch_size: int = 16,
                   pause_ms: float = 320.0, para_ms: float = 620.0,
                   jitter: float = 0.18, speed_jitter: float = 0.02,
                   seed: int = 42,
                   temperature: Optional[float] = None,
                   top_p: Optional[float] = None,
                   repetition_penalty: Optional[float] = None,
                   progress=None) -> Tuple[np.ndarray, int, dict]:
        """长文本分段生成 + 韵律拼接。

        默认单段 60 字：AR 模型存在 style self-referencing——每段开头几秒
        主导后续韵律，段越长尾部越平淡/越慢。短段能让整条音频保持一致的
        能量与语速，实测吞吐也更高（attention 随长度平方增长）。

        seed: 固定随机种子，保证各段音色/韵律一致，避免长音频前后漂移。
        progress: 可选回调 progress(done, total)
        """
        segs = build_segments(text, max_chars=max_chars)
        total = len(segs)
        log.info("文本切分为 %d 段", total)
        if progress:
            progress(0, total)

        # 语气指令可以是多条（换行或 | 分隔），按段轮换，避免通篇一个调
        multi = []
        if instruct:
            multi = [x.strip() for x in re.split(r"[\n|]+", str(instruct)) if x.strip()]
        fallback = DEFAULT_INSTRUCT.get(str(language).lower(), "")
        per_instruct = [(multi[i % len(multi)] if multi else fallback)
                        for i in range(total)]

        # 长度分桶：一个 batch 内若长短混杂，短序列会被 pad 到最长的那条，
        # 算力全浪费在 padding 上。按长度排序后再分批可显著提速。
        audios: List[Optional[np.ndarray]] = [None] * total
        sr_out = 24000
        t_start = time.time()
        order = sorted(range(total), key=lambda i: len(segs[i]["text"]))
        for i in range(0, total, batch_size):
            idxs = order[i:i + batch_size]
            chunk = [segs[k] for k in idxs]
            texts = [s["text"] for s in chunk]
            wavs, sr = self.synth_batch(
                texts, mode=mode, language=language, speaker=speaker,
                instruct=[per_instruct[k] for k in idxs],
                ref_audio=ref_audio, ref_text=ref_text,
                ref_key=ref_key, seed=seed,
                temperature=temperature, top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            sr_out = sr
            for k, w in zip(idxs, wavs):
                audios[k] = np.asarray(w, dtype=np.float32).flatten()
            if progress:
                progress(min(i + len(chunk), total), total)
        audios = [a for a in audios if a is not None]

        if not audios:
            return np.zeros(0, dtype=np.float32), sr_out, {"segments": 0}

        # 削掉每段首尾的模型静音，否则会和拼接停顿叠加成 1.5~2s 的死寂
        audios = [trim_edges(a, sr=sr_out) for a in audios]

        # 语速体检：字/秒过低的段说明模型跑偏了（拖长音/重复），
        # 切成更小的片段重生成一次，取语速更正常的版本。
        slow_fixed = self._fix_slow_segments(
            audios, segs[:len(audios)], sr_out, mode=mode, language=language,
            speaker=speaker, instruct=multi[0] if multi else fallback,
            ref_audio=ref_audio, ref_text=ref_text, ref_key=ref_key, seed=seed,
            temperature=temperature, top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # 韵律起伏：语速走"呼吸曲线"（起句略慢→中段稍快→收尾放缓）+ 段落间轻微音量差。
        # 单调的匀速朗读是长音频 AI 味最主要的来源。
        if speed_jitter > 0:
            base_seed = seed or 0
            factors = prosody_factors(len(audios), seed=base_seed + 7)
            rng2 = np.random.default_rng(base_seed + 13)
            audios = [apply_speed_jitter(a, f) for a, f in zip(audios, factors)]
            audios = [apply_gain(a, float(rng2.uniform(-0.9, 0.9))) for a in audios]

        out = concat_segments(audios, segs[:len(audios)], sr=sr_out,
                              base_ms=pause_ms, para_ms=para_ms, jitter=jitter)
        elapsed = time.time() - t_start
        dur = len(out) / sr_out

        detail = []
        for a, s in zip(audios, segs[:len(audios)]):
            d = len(a) / sr_out
            n = max(len(s["text"].strip()), 1)
            detail.append({"chars": n, "dur": round(d, 2), "cps": round(n / d, 2) if d > 0 else 0})
        total_chars = sum(d["chars"] for d in detail)
        stats = {
            "segments": len(audios),
            "gen_seconds": round(elapsed, 2),
            "audio_seconds": round(dur, 2),
            "rtf": round(elapsed / dur, 4) if dur > 0 else 0,
            "chars": total_chars,
            "cps": round(total_chars / dur, 2) if dur > 0 else 0,
            "slow_fixed": slow_fixed,
            "segments_detail": detail,
        }
        cps = stats["cps"]
        if cps and cps < SLOW_CPS:
            log.warning("整体语速偏慢: %.2f 字/秒 (阈值 %.1f)，段数 %d",
                        cps, SLOW_CPS, len(audios))
        return out, sr_out, stats

    def _fix_slow_segments(self, audios, segs, sr, mode="custom", language="Korean",
                           speaker=None, instruct="", ref_audio=None, ref_text=None,
                           ref_key=None, min_chars: int = 40, seed: Optional[int] = None,
                           temperature=None, top_p=None, repetition_penalty=None) -> int:
        """把语速异常慢的段切成更小的片段重生成，成功替换返回修复数量。"""
        fixed = 0
        for idx, (a, seg) in enumerate(zip(audios, segs)):
            text = seg["text"].strip()
            dur = len(a) / sr
            cps = len(text) / dur if dur > 0 else 0
            if dur <= 0 or cps >= SLOW_CPS or len(text) < min_chars:
                continue
            sub = build_segments(text, max_chars=max(min_chars, len(text) // 2))
            if len(sub) < 2:
                continue
            try:
                wavs, sr2 = self.synth_batch(
                    [s["text"] for s in sub], mode=mode, language=language,
                    speaker=speaker, instruct=instruct,
                    ref_audio=ref_audio, ref_text=ref_text, ref_key=ref_key,
                    seed=seed, temperature=temperature, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                new_parts = []
                ok = True
                for w, s in zip(wavs, sub):
                    arr = trim_edges(np.asarray(w, dtype=np.float32).flatten(), sr=sr2)
                    d = len(arr) / sr2
                    if d > 0 and len(s["text"].strip()) / d >= SLOW_CPS:
                        new_parts.append(arr)
                    else:
                        ok = False
                        break
                if ok and new_parts:
                    merged = np.concatenate(new_parts)
                    if len(merged) / sr2 < dur:
                        audios[idx] = merged
                        fixed += 1
                        log.info("段 %d 语速修复: %.2f -> %.2f 字/秒",
                                 idx, cps, len(text) / (len(merged) / sr2))
            except Exception as e:
                log.warning("段 %d 重生成失败: %s", idx, e)
        return fixed
