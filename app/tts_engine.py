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

# ---------------------------------------------------------------------------
# 自适应批量（员工零感知）：由代码依据「单批 token 预算 + 当前可用显存」反推
# batch_size，并对分段字数做硬上限，调用方不可能再配出会 OOM 的组合。
#
# 经验标定依据（RTX 3060 12G，模型 bf16 常驻约 4.2 GiB，见 bench_result.json）：
#   单段 token 数 = min(2048, max_chars * 6.0 + 96)   # 与 synth_batch 内估算一致
#   实测失败下限：单批 ≈ 7 560 tokens（max_chars=110 × 10 段）必 OOM
#   实测成功上限：单批 ≈ 7 296 tokens（max_chars=60  × 16 段）稳定通过
#   → 取 7 300 为阈值，再留约 50% 余量吸收碎片，得干净显存预算 ≈ 3 650 tokens
#
# batch_size 上限取 8：本机 bench 实测 batch_8 的 RTF 最优（0.49），
# batch_4 反而退化到 1.07；同时也是上述预算在 max_chars=60 下的自然解。
# ---------------------------------------------------------------------------
SEG_MAX_CHARS = 60          # 单段字数硬上限：超过会韵律漂移、越读越慢
SEG_MIN_CHARS = 20
MAX_AUTO_BATCH = 8
MIN_AUTO_BATCH = 1
SAFE_BATCH_TOKENS = 3650    # 干净显存下的单批 token 预算
REF_HEADROOM_GIB = 7.4      # 模型加载后应有的净可用显存（11.63 - 4.2）
OOM_MAX_RETRY = 3           # OOM 降批重试次数（每次批量减半）
OOM_THRESHOLD_TOKENS = 7300  # 实测崩溃阈值，仅用于日志对照

# ---------------------------------------------------------------------------
# 克隆模式（ICL）的参考音频前缀成本
#
# generate_voice_clone 的 voice_clone_prompt=[prompt]*n —— **每个样本都背一份
# 完整前缀**，prompt 长度会按 batch 倍数放大 KV cache，这是克隆模式唯一的
# 额外显存项，也是 329s 参考音频必 OOM 的直接原因。
#
# 实测参数（Qwen3-TTS-12Hz-1.7B-Base，服务端真机读取 config.json）：
#   codec 帧率 = input_sample_rate / encode_downsample_rate = 24000 / 1920 = 12.5 Hz
#   talker: 28 层 / 16 注意力头 / 8 KV 头 / head_dim 128 / bf16
#   → KV = 2(K,V) × 28 × 8 × 128 × 2B = 114 688 B/token ≈ 0.109375 MiB/token
#   韩文文本实测 0.634 token/字（AutoTokenizer 三段样本均值）
#
# 329.33s + 全文稿 ≈ 5 420 prompt tokens → 593 MiB/样本 → batch8 需 4.63 GiB
# > 空余 4.14 GiB ⇒ 必 OOM；batch2（1.16 GiB）才安全。
# 裁到 25s 后 ≈ 430 tokens → 47 MiB/样本 → batch8 仅 0.37 GiB，吞吐零损失。
# ---------------------------------------------------------------------------
CODEC_HZ = 12.5                 # 语音 codec 帧率（帧/秒）
KO_TOKENS_PER_CHAR = 0.634      # 韩文 token/字（实测）
KV_MIB_PER_TOKEN = 0.109375     # 2 × 28层 × 8KV头 × 128维 × 2B / 1024 / 1024
CLONE_PROMPT_OVERHEAD = 24      # prompt 里的特殊 token / 结构开销
ACT_GIB_PER_ITEM = 0.30         # 单样本激活显存：实测 batch8 峰值 6.97 - 常驻 4.17 ≈ 2.8
KV_HEADROOM_SHARE = 0.40        # prompt KV 最多吃掉的净余量比例
MEM_SAFETY = 0.85               # 显存估算安全系数


def estimate_clone_prompt_tokens(ref_audio=None, ref_text: str = "") -> int:
    """估算单条样本的 ICL 前缀 token 数（codec 帧 + 文字稿 + 结构开销）。"""
    codec = 0
    if ref_audio is not None:
        try:
            data, sr = ref_audio
            codec = int(max(0.0, len(data) / float(sr)) * CODEC_HZ)
        except Exception:
            codec = 0
    return int(codec + len(ref_text or "") * KO_TOKENS_PER_CHAR + CLONE_PROMPT_OVERHEAD)


def seg_token_cap(max_chars: int) -> int:
    """单段 token 上限，必须与 synth_batch 里的估算公式保持一致。"""
    return min(2048, int(max_chars * 6.0) + 96)


def gpu_headroom_gib() -> Optional[float]:
    """返回「总显存 - 当前已分配」的净余量(GiB)；无 CUDA 时返回 None。

    用已分配量而非 nvidia-smi 的 free：PyTorch 缓存分配器 reserved 的部分
    是可以被后续分配复用的，算作余量才不会误判。
    """
    try:
        if not torch.cuda.is_available():
            return None
        total = torch.cuda.get_device_properties(0).total_memory
        return (total - torch.cuda.memory_allocated(0)) / (1024 ** 3)
    except Exception:
        return None


def gpu_free_gib() -> Optional[float]:
    """真正尚未被占用的显存(GiB)，不含 PyTorch 已 reserved 但可复用的部分。

    KV cache 是「硬占用」——分配后不能被别的张量挤走，所以它只能用真正的
    空余显存来算，不能像 token 预算那样把 reserved 计为余量。
    """
    try:
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    except Exception:
        return None


def is_oom(exc: BaseException) -> bool:
    """跨 torch 版本的 OOM 判定（老版本只有 RuntimeError + 英文文案）。"""
    try:
        oom_type = getattr(torch, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg or "cudnn_status_alloc_failed" in msg


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

    # ---- 自适应配置 -------------------------------------------------------
    def plan_generation(self, max_chars=None, batch_size=None,
                        clone_prompt_tokens: int = 0) -> dict:
        """把调用方传入的分段字数/批量大小收敛到安全区间。

        命中过的故障：调用方传 max_chars=110 / batch_size=16 或 1000 时，
        单批 token 预算会冲到 7 560~31 680，必然 OOM。这里统一兜住，
        实际生效值写进返回值的 max_chars/batch_size，供 stats 落盘审计。

        clone_prompt_tokens：克隆模式下每个样本额外背负的前缀 token 数。
        不传时按普通模式处理（前缀为 0）。
        """
        try:
            req_mc = int(max_chars) if max_chars else SEG_MAX_CHARS
        except Exception:
            req_mc = SEG_MAX_CHARS
        try:
            req_bs = int(batch_size) if batch_size else MAX_AUTO_BATCH
        except Exception:
            req_bs = MAX_AUTO_BATCH

        # 分段字数：超过上限一律压回 60（过长会引发韵律漂移与拖长音）
        mc = max(SEG_MIN_CHARS, min(req_mc, SEG_MAX_CHARS))

        headroom = gpu_headroom_gib()
        if headroom is None:
            ratio = 1.0
        else:
            # 服务长期运行后 PyTorch reserved 会累积，显存越脏批量越小
            ratio = max(0.30, min(1.0, headroom / REF_HEADROOM_GIB))

        # 克隆前缀用「显存」而不是「token 预算」来卡：前缀只做一次 prefill，
        # 不产生 decode 步和 vocoder 开销，按 token 预算折算会过度保守，
        # 把 25s 参考音频（完全安全）也从 batch8 压到 batch4，白白损失吞吐。
        prompt_kv_gib = KV_MIB_PER_TOKEN * max(0, int(clone_prompt_tokens)) / 1024.0
        mem_bs = MAX_AUTO_BATCH
        # KV 用「真正空余」而不是 headroom 来算：headroom 把 PyTorch reserved
        # 计成余量，对 token 预算成立，对不可挤压的 KV 不成立。
        free = gpu_free_gib()
        kv_base = headroom if free is None else min(headroom, free)
        if kv_base is not None:
            per_item = (ACT_GIB_PER_ITEM
                        + prompt_kv_gib
                        + KV_MIB_PER_TOKEN * seg_token_cap(mc) / 1024.0)
            mem_bs = int((kv_base * MEM_SAFETY) // per_item)
            if prompt_kv_gib > 0:
                # 前缀 KV 按 batch 倍数放大，单独设一道更硬的天花板
                mem_bs = min(mem_bs, int((kv_base * KV_HEADROOM_SHARE) // prompt_kv_gib))
            mem_bs = max(MIN_AUTO_BATCH, mem_bs)

        budget = SAFE_BATCH_TOKENS * ratio
        auto_bs = int(budget // seg_token_cap(mc))
        auto_bs = max(MIN_AUTO_BATCH, min(auto_bs, MAX_AUTO_BATCH, mem_bs))
        # 调用方显式要求更小的批量（调试用）则尊重；要求更大的一律无效
        bs = min(auto_bs, req_bs) if req_bs >= 1 else auto_bs
        bs = max(MIN_AUTO_BATCH, bs)

        per_seg = seg_token_cap(mc)
        info = {
            "max_chars": mc,
            "batch_size": bs,
            "requested_max_chars": req_mc,
            "requested_batch_size": req_bs,
            "headroom_gib": round(headroom, 2) if headroom is not None else None,
            "memory_scale": round(ratio, 3),
            "seg_token_cap": per_seg,
            "batch_token_budget": int(budget),
            "batch_tokens": bs * per_seg,
            "oom_threshold_tokens": OOM_THRESHOLD_TOKENS,
            "clone_prompt_tokens": int(clone_prompt_tokens),
            "clone_prompt_kv_gib": round(prompt_kv_gib, 3),
            "clone_prompt_kv_total_gib": round(prompt_kv_gib * bs, 3),
            "free_gib": round(free, 2) if free is not None else None,
        }
        log.info(
            "自适应配置: max_chars=%d batch_size=%d | 请求值 %d/%d | "
            "净余量 %s GiB -> 预算 %d tokens，实际单批 %d tokens（崩溃阈值 %d）%s",
            mc, bs, req_mc, req_bs, info["headroom_gib"],
            info["batch_token_budget"], info["batch_tokens"],
            OOM_THRESHOLD_TOKENS,
            (" | 克隆前缀 %d tokens -> KV %.2f GiB/样本，合计 %.2f GiB"
             % (int(clone_prompt_tokens), prompt_kv_gib, prompt_kv_gib * bs))
            if clone_prompt_tokens else "")
        return info

    def _gen_batch_safe(self, texts, batch_size, **kw):
        """生成一批文本；显存不足时自动降批重试，返回与 texts 等长的结果。

        返回 (wavs, sr, used_bs)。used_bs 是本次实际跑通的批量，
        调用方据此收缩后续批次，避免每批都重新踩一次 OOM。
        """
        n = len(texts)
        if n <= 0:
            return [], 24000, max(1, int(batch_size or MAX_AUTO_BATCH))
        bs = max(1, min(int(batch_size or MAX_AUTO_BATCH), n))
        pos = 0
        wavs_all = []
        sr_out = 24000
        used_bs = bs
        while pos < n:
            end = min(pos + bs, n)
            sub = texts[pos:end]
            tries = 0
            while True:
                try:
                    wavs, sr = self.synth_batch(sub, **kw)
                except BaseException as e:  # noqa: BLE001 - 只放行 OOM
                    if not is_oom(e):
                        raise
                    tries += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if len(sub) <= 1 or tries > OOM_MAX_RETRY:
                        log.error("显存不足且已无法继续降批（当前 %d 段），放弃: %s",
                                  len(sub), str(e)[:200])
                        raise
                    bs = max(1, len(sub) // 2)
                    used_bs = bs
                    end = min(pos + bs, n)
                    sub = texts[pos:end]
                    log.warning("显存不足 -> 批量降至 %d 后重试（第 %d 次）: %s",
                                bs, tries, str(e)[:120])
                    continue
                sr_out = sr
                wavs_all.extend(list(wavs))
                break
            pos = end
        return wavs_all, sr_out, used_bs

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
                   max_chars: int = SEG_MAX_CHARS, batch_size: int = MAX_AUTO_BATCH,
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
        # 自适应兜底：无论调用方（含前端）传了什么参数，都先收敛到安全区间
        # 克隆模式下把参考音频前缀的 token 数一并计入，否则长参考音频会让
        # batch 停留在 8，直接把 KV cache 撑爆（329s 参考音频 = 4.63 GiB > 空余 4.14 GiB）
        _prompt_tokens = 0
        if mode == "clone" and ref_audio is not None:
            _prompt_tokens = estimate_clone_prompt_tokens(ref_audio, ref_text or "")
        plan = self.plan_generation(max_chars, batch_size,
                                    clone_prompt_tokens=_prompt_tokens)
        max_chars = plan["max_chars"]
        batch_size = plan["batch_size"]

        segs = build_segments(text, max_chars=max_chars)
        total = len(segs)
        log.info("文本切分为 %d 段", total)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
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
        pos = 0
        bs = batch_size
        batch_no = 0
        while pos < total:
            idxs = order[pos:pos + bs]
            chunk = [segs[k] for k in idxs]
            texts = [s["text"] for s in chunk]
            batch_no += 1
            wavs, sr, used_bs = self._gen_batch_safe(
                texts, bs, mode=mode, language=language, speaker=speaker,
                instruct=[per_instruct[k] for k in idxs],
                ref_audio=ref_audio, ref_text=ref_text,
                ref_key=ref_key, seed=seed,
                temperature=temperature, top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            sr_out = sr
            for k, w in zip(idxs, wavs):
                audios[k] = np.asarray(w, dtype=np.float32).flatten()
            # 批间释放 PyTorch 缓存块，阻断 reserved 跨批累积——
            # 这是服务长期不重启时显存只涨不跌的主因
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 本批若因显存降过级，后续批次沿用，避免每批重复试错
            if used_bs < bs:
                bs = max(MIN_AUTO_BATCH, used_bs)
            pos += len(idxs)
            if progress:
                progress(min(pos, total), total)
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
        if torch.cuda.is_available():
            plan["peak_alloc_gib"] = round(
                torch.cuda.max_memory_allocated(0) / (1024 ** 3), 2)
            plan["alloc_now_gib"] = round(
                torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
        plan["batch_size_final"] = bs
        plan["batches"] = batch_no

        stats = {
            "segments": len(audios),
            "gen_seconds": round(elapsed, 2),
            "audio_seconds": round(dur, 2),
            "rtf": round(elapsed / dur, 4) if dur > 0 else 0,
            "chars": total_chars,
            "cps": round(total_chars / dur, 2) if dur > 0 else 0,
            "slow_fixed": slow_fixed,
            "segments_detail": detail,
            "plan": plan,
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
            except BaseException as e:  # noqa: BLE001
                if is_oom(e) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log.warning("段 %d 重生成失败: %s", idx, e)
        return fixed
