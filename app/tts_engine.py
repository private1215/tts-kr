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
                      trim_edges, prosody_factors, apply_gain, split_for_retry)
from voice_match import match_segments

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

# 低于此字/秒视为"语速异常慢"（旧判据，仅作兜底下限；新判据见 ABNORMAL_RATIO）
SLOW_CPS = 2.6

# ---------------------------------------------------------------------------
# 解码策略（2026-09-28 定版）
#
# 根因：Qwen3-TTS 在 ICL 克隆下存在官方已知的退化循环（degenerate loop）——
# 模型不吐 EOS，进入重复/拖长音。默认 do_sample=True + temperature=0.9 + top_k=50
# 会放大触发概率；而 max_new_tokens 又给得极松（2048，或 字数*6.0+96 ≈ 允许
# 慢到 2.08 字/秒，是韩语正常值的 3.7 倍），失控段于是一路跑到上限被硬截断。
#
# 现网实锤（任务 f38869875eb4，4133 字 81 段）：
#   31/81 段语速跑飞，区间 1.51 ~ 17.52 字/秒（差 11.6 倍）——
#   快的段吞字含混（员工反馈"乱码"），慢的段拖长重复（"呀呀呀呀"），
#   最慢 4 段 45.0~46.4s，正好卡在旧上限 min(2048, 75*6+96=546) 上被截断。
#
# 失败是**随机性**的，不是确定性的：同一篇文章、同一套参数、连 plan 都完全一致
# （batch 8→1、batches 11、headroom 7.68）的两次运行，
#   7f20 → 中位 8.30 字/秒，异常 1/81
#   f388 → 中位 6.17 字/秒，异常 31/81
# 若存在一个确定的致病因子，两次运行不可能差这么远。所以对策不是"调参调到最好"，
# 而是**拒绝采样**：生成后体检，语速异常的段换个随机种子重来，直到落进健康区间。
#
# 关于贪心解码（曾被寄予厚望，实测证伪）：
#   离线对照（同 7 段文本，batch=1）：
#     采样+旧默认   104.5s，异常 4/7
#     贪心+惩罚+收紧 123.4s，异常 5/7   ← 反而更差
#   原因：贪心在这批文本上**不吐 EOS**，多段直接生成到 max_new_tokens 被硬截断
#   （不同长度的段齐刷刷都是 19.60s = 246 token 上限，就是被截断的铁证）。
#   贪心没有随机扰动，一旦进入重复循环反而跳不出来。故默认仍用采样。
#
# ⚠️ 待验证的嫌疑：repetition_penalty=1.1 可能让整体语速系统性变慢
#
#   证据（2026-09-28 现网，同文章、job 参数逐项一致，只有代码版本不同）：
#     旧代码（无 rp / 无体检）  7f20 → 中位 8.30 字/秒，实际/预期比中位 0.84
#     新代码（rp=1.1 + 体检）   A   → 中位 5.65 字/秒，实际/预期比中位 1.24
#   整体慢了约 48%，而且不只是异常段偏慢——81 段整体分布都右移了，
#   符合"某个参数系统性影响所有段"的特征，rp 是唯一的全局新变量。
#   另一个候选 token 上限已排除：健康段最多只需预期时长的 1.07 倍，
#   而新上限给到 1.6 倍，不存在截断。
#
#   → 必须做 rp 1.05 / 1.1 的对照才能定论。**未经验证前不要动这个值**。
# ---------------------------------------------------------------------------
# TTS_GEN_DO_SAMPLE=0 可切到贪心（保留开关，便于将来对照实验）
GEN_DO_SAMPLE = os.environ.get("TTS_GEN_DO_SAMPLE", "1") not in ("0", "", "false", "False")
GEN_REPETITION_PENALTY = float(os.environ.get("TTS_GEN_REPETITION_PENALTY", "1.1") or 0) or None
# 体检不达标的段，换随机种子最多重试几次（拒绝采样）。
# 2 次而非 3 次：异常率约 25% 时，第 3 次的边际收益很低，但要多付 1/3 的重试时间
# （现网 A 任务 RTF 2.49，其中重试是大头）。
ABNORMAL_MAX_RETRY = int(os.environ.get("TTS_ABNORMAL_MAX_RETRY", "2") or 2)
# 达到这个字数的段，体检失败时**第一次重试就直接二分**，不再先整段重掷。
# 实测依据：现网 C 组 `9423eadfe805` 残留的最长段是段 08（81 字 / 25.81s），
# 它在两次重试机会里，第一次是"整段换种子"——而长段跑飞是**段长本身**
# 让模型迟迟不吐 EOS（上下文越长越容易拖长音），换种子不改变段长，
# 等于把一半机会浪费在一次注定失败的重掷上。
# 二分则直接缩短上下文，是唯一能改变结果的动作，所以长段要优先用它。
# 短段相反：短段跑飞是随机波动，二分会破坏韵律连贯，仍先整段重掷。
LONG_SEG_BISECT_CHARS = int(os.environ.get("TTS_LONG_SEG_BISECT_CHARS", "60") or 60)

# ---------------------------------------------------------------------------
# 流式 vs 非流式生成 —— ⚠️ 已排除，不是问题的根因
#
# 曾怀疑这里是元凶：官方 Issue #239 的机制追踪里，同一份参考音频
# non_streaming 语速漂移 0.0%、streaming 漂移 +16.7%，而 generate_voice_clone
# 的 non_streaming_mode **默认是 False（流式）**。
# 但核查 synth_batch 发现：**代码里早就显式传了 non_streaming_mode=True**，
# 现网所有任务一直是非流式。所以这条线索是空的，不要再往这个方向查。
#
# 保留这段记录是因为：A 任务整体偏慢 48%（中位 5.65 vs 健康基准 8.30），
# 症状与流式漂移高度相似，很容易误判到这里。真正的嫌疑见文件头的
# 「repetition_penalty 会让整体变慢」一节。
# ---------------------------------------------------------------------------

# 各语言的正常语速（字/秒），用于反推 token 上限与生成后体检。
# 韩语取本业务实测中位（健康样本 8.30、产能模型 6.15），折中 7.0 留余量。
CPS_REF = {
    "korean": 7.0, "chinese": 5.0, "japanese": 6.0, "english": 9.0,
    "german": 8.0, "french": 8.0, "spanish": 9.0, "portuguese": 9.0,
    "russian": 8.0, "italian": 8.0, "dutch": 8.0,
}
CPS_REF_DEFAULT = 6.0
# 由健康样本实测标定，不是拍脑袋：
#   健康样本 7f20 共 81 段，「实际时长 ÷ 预期时长」的分布是
#     中位 0.84、p90 0.93、p99 1.07、**最大 1.07**
#   → 健康段最多只需要预期时长的 1.07 倍，取 2.0 有近一倍安全余量，零误伤（0/81）。
#
# 为什么不取更激进的 1.6：模型输出有约 33% 的随机节奏波动（同文本三次跑出
# 7.67 / 5.75 / 4.95 字/秒，见 diag_cross.py），整体偏慢的批次里「实际/预期」
# p90 能到 1.96；用 1.6 会误截 27% 的段，反而触发大量无谓重试。
# 2.0 只截断真正失控的 8.6%，而且失败段更早止损，整体仍省时间。
TOKEN_SAFETY = float(os.environ.get("TTS_TOKEN_SAFETY", "2.0") or 2.0)
TOKEN_FLOOR = 32       # 极短文本的最低额度
ABNORMAL_RATIO = 2.2   # 体检下限：慢于 正常语速/此值 判为异常
FAST_RATIO = 1.8       # 体检上限：快于 正常语速×此值 判为异常
# 阈值标定依据（韩语 → 3.18 ~ 12.6 字/秒）：
#   健康样本 7f20 共 81 段：中位 8.30，p10 7.52，p90 9.26，
#   低于 3.18 的 0 段、高于 12.6 的 1 段 —— 两端误伤率都极低。
#   乱音样本 0c1e7b839b55 逐段 2.70 / 47.98 / 9.77，两端都能抓到。


def cps_reference(language: str = "Korean") -> float:
    """该语言的正常语速（字/秒）。"""
    return CPS_REF.get(str(language or "").strip().lower(), CPS_REF_DEFAULT)


def expected_seconds(text: str, language: str = "Korean") -> float:
    """一段文本按正常语速应有的发声时长（秒）。用于体检，不是硬约束。"""
    n = len((text or "").strip())
    return n / cps_reference(language) if n > 0 else 0.0


def token_cap_for(text: str, language: str = "Korean") -> int:
    """按语言语速反推 max_new_tokens 上限。

    12Hz codec → 12.5 token/秒。一段正常时长 = 字数/cps 秒，
    对应 字数/cps×12.5 个 token，再乘 TOKEN_SAFETY 作为兜底。
    旧公式 字数*6.0+96 等价于允许慢到 2.08 字/秒，退化循环能一路跑到上限。
    """
    n = max(len((text or "").strip()), 1)
    est = n / cps_reference(language) * CODEC_HZ * TOKEN_SAFETY + TOKEN_FLOOR
    return int(min(2048, max(TOKEN_FLOOR * 2, est)))

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
    """显存规划用的单段 token 上限（保守值，不参与实际生成截断）。

    实际截断已改用 token_cap_for()（按语言语速反推，韩语约为本值的 0.63 倍），
    这里故意沿用旧的宽松公式：显存预算宁可估高，也不要把 batch 放得太激进。
    """
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

        ⚠️ used_bs 的语义是"本批实际跑了几段"，**不等于"显存只够跑这么多"**：
        尾批不足整批时它天然小于请求值。判定是否真降级由调用方负责
        （需同时满足 len(本批) >= 请求的 bs），别在这里误读。
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
                    do_sample: Optional[bool] = None,
                    ) -> Tuple[List[np.ndarray], int]:
        """批量生成一批文本片段。返回 (音频数组列表, 采样率)。

        instruct 可以是字符串（每段同语气）或列表（每段不同语气）。
        temperature/top_p/repetition_penalty/do_sample 通过 kwargs 透传给底层
        generate，若当前版本不支持会自动降级忽略。

        默认保留采样解码（官方文档与多个社区实现都指出：一旦进入重复循环，
        贪心没有随机扰动反而跳不出来，实测也证伪，见文件头注释块）。
        同时默认带 repetition_penalty，并对语速异常的段自动换种子重试。
        理由与实测数据见文件头"解码策略"注释块。
        """
        self._ensure_variant(mode)
        n = len(texts)
        langs = [language] * n
        gen_kw = {}
        # 贪心解码：克隆的退化循环（不吐 EOS → 重复/拖长音）主要由采样引入。
        # 关掉采样后输出是确定性的，同输入必得同结果，批量大小也不再影响结果。
        # subtalker 是独立的第二路采样开关（拼写为 subtalker_dosample），
        # 必须一起关，否则随机性只关掉一半。
        ds = GEN_DO_SAMPLE if do_sample is None else bool(do_sample)
        gen_kw["do_sample"] = ds
        gen_kw["subtalker_dosample"] = ds
        if temperature is not None:
            gen_kw["temperature"] = float(temperature)
        if top_p is not None:
            gen_kw["top_p"] = float(top_p)
        rp = repetition_penalty if repetition_penalty is not None else GEN_REPETITION_PENALTY
        if rp:
            gen_kw["repetition_penalty"] = float(rp)
        # max_new_tokens 上限：按语言语速反推（见 token_cap_for）。
        # 旧公式 字数*6.0+96 允许慢到 2.08 字/秒，退化循环能一路跑到上限被硬截断，
        # 员工听到的就是"呀呀呀呀"然后突然断掉。
        if max_new_tokens is None:
            longest = max(texts, key=lambda t: len(t)) if texts else ""
            max_new_tokens = token_cap_for(longest, language)
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
                   do_sample: Optional[bool] = None,
                   voice_match: Optional[np.ndarray] = None,
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
        downgrades = 0          # 因显存不足而降批的次数（留痕，便于事后判断质量风险）
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
                repetition_penalty=repetition_penalty, do_sample=do_sample,
            )
            sr_out = sr
            for k, w in zip(idxs, wavs):
                audios[k] = np.asarray(w, dtype=np.float32).flatten()
            # 批间释放 PyTorch 缓存块，阻断 reserved 跨批累积——
            # 这是服务长期不重启时显存只涨不跌的主因
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 本批若因显存降过级，后续批次沿用，避免每批重复试错。
            # 守卫 len(idxs) >= bs：尾批常常不足整批——81 段按 8 分最后只剩 1 段，
            # 此时 used_bs 返回 1，但那是"没那么多段可跑"，不是显存降级。
            # 少了这个守卫会把 bs 一路压到 1，且 plan["batch_size_final"] 记成 1
            # （排查时误以为全程单段串行，实际前 10 批都是 8）。
            if used_bs < bs and len(idxs) >= bs:
                downgrades += 1
                bs = max(MIN_AUTO_BATCH, used_bs)
            pos += len(idxs)
            if progress:
                progress(min(pos, total), total)
        audios = [a for a in audios if a is not None]

        if not audios:
            return np.zeros(0, dtype=np.float32), sr_out, {"segments": 0}

        # 削掉每段首尾的模型静音，否则会和拼接停顿叠加成 1.5~2s 的死寂
        audios = [trim_edges(a, sr=sr_out) for a in audios]

        # 语速体检：超出预期时长 ABNORMAL_RATIO 倍的段说明模型跑偏了（拖长音/重复），
        # 切成更小的片段重生成一次，取语速更正常的版本。
        slow_fixed, abnormal = self._fix_abnormal_segments(
            audios, segs[:len(audios)], sr_out, mode=mode, language=language,
            speaker=speaker, instruct=multi[0] if multi else fallback,
            ref_audio=ref_audio, ref_text=ref_text, ref_key=ref_key, seed=seed,
            temperature=temperature, top_p=top_p,
            repetition_penalty=repetition_penalty, do_sample=do_sample,
        )

        # 韵律起伏：语速走"呼吸曲线"（起句略慢→中段稍快→收尾放缓）+ 段落间轻微音量差。
        # 单调的匀速朗读是长音频 AI 味最主要的来源。
        if speed_jitter > 0:
            base_seed = seed or 0
            factors = prosody_factors(len(audios), seed=base_seed + 7)
            rng2 = np.random.default_rng(base_seed + 13)
            audios = [apply_speed_jitter(a, f) for a, f in zip(audios, factors)]
            audios = [apply_gain(a, float(rng2.uniform(-0.9, 0.9))) for a in audios]

        # 保真增强：逐段把频谱对齐到该音色的原始录音，补回生成环节丢掉的 2~4kHz。
        # 原理与实测数据见 app/voice_match.py 模块头。放在拼接之前，
        # 保证每段各自匹配（段间偏离度能差 4.7 倍，整条用一条曲线会推过头）。
        voice_matched = 0
        if voice_match is not None and audios:
            try:
                audios, voice_matched = match_segments(audios, voice_match, sr_out)
                log.info("保真增强: %d/%d 段已对齐源录音频谱", voice_matched, len(audios))
            except Exception as e:
                # 后处理出任何问题都不能让任务失败，宁可原样输出
                log.warning("保真增强失败，按原样输出: %s", e)
                voice_matched = 0

        out = concat_segments(audios, segs[:len(audios)], sr=sr_out,
                              base_ms=pause_ms, para_ms=para_ms, jitter=jitter)
        elapsed = time.time() - t_start
        dur = len(out) / sr_out

        detail = []
        for a, s in zip(audios, segs[:len(audios)]):
            d = len(a) / sr_out
            n = max(len(s["text"].strip()), 1)
            detail.append({"chars": n, "dur": round(d, 2),
                           "cps": round(n / d, 2) if d > 0 else 0,
                           # 预期时长：排查时一眼看出这段慢了几倍/快了几倍
                           "exp": round(expected_seconds(s["text"], language), 2)})
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
            "abnormal_segments": abnormal,
            "batch_downgraded": downgrades,
            "segments_detail": detail,
            "plan": plan,
            "voice_matched": voice_matched,
        }
        cps = stats["cps"]
        if cps and cps < SLOW_CPS:
            log.warning("整体语速偏慢: %.2f 字/秒 (阈值 %.1f)，段数 %d",
                        cps, SLOW_CPS, len(audios))
        return out, sr_out, stats

    def _gen_one(self, texts, mode, language, speaker, instruct, ref_audio,
                 ref_text, ref_key, seed, temperature, top_p,
                 repetition_penalty, do_sample):
        """生成一个 batch 并做边缘裁剪，返回 (音频数组列表, sr)。"""
        wavs, sr2 = self.synth_batch(
            texts, mode=mode, language=language, speaker=speaker,
            instruct=instruct, ref_audio=ref_audio, ref_text=ref_text,
            ref_key=ref_key, seed=seed, temperature=temperature, top_p=top_p,
            repetition_penalty=repetition_penalty, do_sample=do_sample,
        )
        return [trim_edges(np.asarray(w, dtype=np.float32).flatten(), sr=sr2)
                for w in wavs], sr2

    def _fix_abnormal_segments(self, audios, segs, sr, mode="custom", language="Korean",
                               speaker=None, instruct="", ref_audio=None, ref_text=None,
                               ref_key=None, min_chars: int = 6, seed: Optional[int] = None,
                               temperature=None, top_p=None, repetition_penalty=None,
                               do_sample=None) -> Tuple[int, int]:
        """逐段语速体检，不达标的段换随机种子重试。返回 (修复数, 异常数)。

        双侧判据（旧版只看"慢"，且阈值 2.6 字/秒太松）：
          过慢 —— 拖长音/重复循环，模型迟迟不吐 EOS；
          过快 —— 吞字含混，员工反馈的"乱码"主要来自这里。

        必须逐段判，不能只看整体：现网 0c1e7b839b55 整体 6.49 字/秒看似健康，
        逐段却是 2.70 / 47.98 / 9.77（50 字只念了 1.04 秒），两个极端被均值抹平。

        修复策略是**拒绝采样**而不是"细分文本"：跑飞是随机事件（同一文本同一参数
        两次跑出 1/81 与 31/81 的差距），换种子重掷一次大概率正常；细分会改变
        韵律连贯性，且对本就念得太快的段没用。奇数次重试用原文换种子，
        偶数次退化为半段二分并换种子，两条路都给机会。

        ⚠️ 2026-09-28 修 bug：这里的"二分"曾长期静默失效。
        旧代码用 build_segments(text, max_chars=len(text)//2)，而它只按
        **句末标点**切句——体检对象本就是已分好的单句段，再切一次切不开
        （实测 93 字真实韩语 -> 仍 1 段），于是两次重试全是整段换种子。
        现改用 split_for_retry（标点 > 停顿 > 空格 > 硬切，保证切成两段）。

        min_chars 由 25 降到 6（2026-09-28 修 bug）：
        旧下限会把短段整段跳过，而短段恰恰最容易跑飞且最刺耳——
        现网 A 任务 `499080dcde4d` 的四段漏网里，段 20（10 字 / 1.46 字/秒）
        和段 40（18 字 / 2.99 字/秒）都是因为字数少于 25 被直接 continue 放过的。
        段 20 起点在 215 秒（3 分 35 秒），正是员工反馈"4 分钟后开始有乱音"的位置。
        低于 6 字不体检：过短片段的语速估计本身不可靠，容易误判。
        """
        fixed = 0
        abnormal = 0
        ref_cps = cps_reference(language)
        # 韩语 = 7.0/2.2 = 3.18 字/秒；中文 = 5.0/2.2 = 2.27 -> 兜底到旧阈值 2.6
        slow_cps = max(SLOW_CPS, ref_cps / ABNORMAL_RATIO)
        fast_cps = ref_cps * FAST_RATIO

        def _cps(arr, text):
            d = len(arr) / sr
            return (len(text) / d) if d > 0 else 0.0

        for idx, (orig, seg) in enumerate(zip(audios, segs)):
            text = seg["text"].strip()
            if len(text) < min_chars or len(orig) == 0:
                continue
            best = orig
            best_dev = abs(_cps(orig, text) - ref_cps)
            flagged = False
            attempt = 0
            while True:
                c = _cps(best, text)
                if slow_cps <= c <= fast_cps:
                    break
                if not flagged:
                    flagged = True
                    abnormal += 1
                    log.warning("段 %d 语速%s: %d 字 %.2fs = %.2f 字/秒"
                                "（正常约 %.1f，判定区间 %.2f~%.2f）",
                                idx, "过慢" if c < slow_cps else "过快",
                                len(text), len(best) / sr, c, ref_cps, slow_cps, fast_cps)
                if attempt >= ABNORMAL_MAX_RETRY:
                    log.warning("段 %d 重试 %d 次仍未回到正常区间，保留最接近的 "
                                "%.2f 字/秒", idx, attempt, c)
                    break
                attempt += 1
                # 奇数次：整段换种子重来；偶数次：二分后换个种子。
                # 例外：长段第一次就二分（理由见 LONG_SEG_BISECT_CHARS）。
                base = (seed or 0) + attempt * 7919
                if len(text) >= LONG_SEG_BISECT_CHARS or attempt % 2 == 0:
                    # 用 split_for_retry 而不是 build_segments：后者只认句末标点，
                    # 对已经是一整句的段切不开（实测 93 字真实韩语 -> 仍 1 段），
                    # 会让"二分"静默退化成"整段换种子"。
                    parts = split_for_retry(text, min_chars=min_chars)
                else:
                    parts = [text]
                try:
                    arrs, sr2 = self._gen_one(
                        parts, mode, language, speaker, instruct,
                        ref_audio, ref_text, ref_key, base, temperature, top_p,
                        repetition_penalty, do_sample)
                    cand = np.concatenate(arrs) if len(arrs) > 1 else arrs[0]
                    if len(cand) == 0:
                        continue
                    nc = _cps(cand, text)
                    log.info("段 %d 第 %d 次重试 -> %.2f 字/秒", idx, attempt, nc)
                    dev = abs(nc - ref_cps)
                    if dev < best_dev:
                        best, best_dev = cand, dev
                except BaseException as e:  # noqa: BLE001
                    if is_oom(e) and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    log.warning("段 %d 第 %d 次重生成失败: %s", idx, attempt, e)
                    break
            if best is not orig:
                audios[idx] = best
                fixed += 1
                log.info("段 %d 最终采用重生成版: %.2f -> %.2f 字/秒",
                         idx, _cps(orig, text), _cps(best, text))
        return fixed, abnormal
