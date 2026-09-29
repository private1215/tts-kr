# -*- coding: utf-8 -*-
"""韩语长文本分段 + 韵律拼接。

长音频(8分钟级)不能一次性 AR 生成，否则韵律漂移、后段重复。
这里把文本切成句读级片段，每段可带独立语气指令，生成后再按真人停顿拼接。
"""
import re
import random
import numpy as np

# 句末标点：中英韩混排场景
SENT_END = re.compile(r'(?<=[.!?。！？…])\s*')
# 句中停顿点
CLAUSE_END = re.compile(r'(?<=[,，、;；:：—])\s*')

# 韩语常见需要口语化展开的模式（数字/单位/缩写）
KO_NUMBER_UNIT = [
    (re.compile(r'(\d+)\s*%'), lambda m: f"{m.group(1)}퍼센트"),
    (re.compile(r'(\d+)\s*℃'), lambda m: f"{m.group(1)}도"),
    (re.compile(r'(\d+)\s*km', re.I), lambda m: f"{m.group(1)}킬로미터"),
    (re.compile(r'(\d+)\s*kg', re.I), lambda m: f"{m.group(1)}킬로그램"),
    (re.compile(r'(\d+)\s*cm', re.I), lambda m: f"{m.group(1)}센티미터"),
    (re.compile(r'(\d+)\s*mm', re.I), lambda m: f"{m.group(1)}밀리미터"),
    (re.compile(r'(\d+)\s*원'), lambda m: f"{m.group(1)}원"),
]


def normalize_korean(text: str) -> str:
    """文本规整：把数字+单位、缩写展开成自然口语读法，减少机械感。"""
    t = text.strip()
    for pat, rep in KO_NUMBER_UNIT:
        t = pat.sub(rep, t)
    # 多余空白
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t


def split_sentences(text: str) -> list:
    """按句末标点切句，保留标点本身。"""
    parts = SENT_END.split(text)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def merge_segments(sentences: list, max_chars: int = 160, min_chars: int = None) -> list:
    """把过短的句子合并到 max_chars 上限内，避免生成太碎导致韵律断裂。

    注意：合并短片段时必须同时守住上限，否则短句会被一路堆到相邻段上，
    段长彻底失控（实测 60 字的上限切出过 141 字的段，导致该段生成 45 秒、
    越读越慢）。max_chars 因此形同虚设。
    """
    if min_chars is None:
        min_chars = max(24, int(max_chars * 0.45))
    # 合并短片段时允许的溢出上限
    cap = int(max_chars * 1.4)

    segs = []
    buf = ""
    for s in sentences:
        if not buf:
            buf = s
            continue
        if len(buf) + len(s) <= max_chars:
            buf = buf + " " + s
        else:
            segs.append(buf)
            buf = s
    if buf:
        segs.append(buf)

    # 把过短的片段并到相邻片段，但不得让目标段超过 cap
    merged = []
    for s in segs:
        if merged and len(s) < min_chars and len(merged[-1]) + len(s) + 1 <= cap:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


def build_segments(text: str, max_chars: int = 160) -> list:
    """完整分段流程：规整 -> 切句 -> 合并。返回片段列表。"""
    t = normalize_korean(text)
    # 空行视为段落分隔，段落之间停顿更长
    paragraphs = re.split(r'\n\s*\n', t)
    segments = []
    for pi, para in enumerate(paragraphs):
        sents = split_sentences(para)
        merged = merge_segments(sents, max_chars=max_chars)
        for si, seg in enumerate(merged):
            segments.append({
                "text": seg,
                "para": pi,
                "idx": si,
                "is_para_end": (si == len(merged) - 1),
            })
    return segments


def split_for_retry(text: str, min_chars: int = 6) -> list:
    """把一段文本切成两半，供语速体检失败时重试使用。**保证切得开**。

    为什么不能直接用 build_segments：
      build_segments 只按**句末标点**切句（SENT_END），而体检的对象
      本身就是已经分好的句读级片段——再走一遍往往切不开。实测：
        93 字真实韩语句子 + max_chars=22 -> 仍返回 1 段
        带空格的 79 字文本 + max_chars=22 -> 仍返回 1 段
      结果是「二分重试」从上线起就没真正二分过，两次重试机会实际都是
      整段换种子。这正是段 08（81 字 / 30.52s）连救两次都没救回来的原因。

    四级降级，最后一定切得开：
      1. 句末标点（。！？…）  2. 句中停顿（，、；：—）
      3. 空格（韩语词边界）    4. 中点硬切（宁可在词中断，也要缩短上下文）

    硬切听起来会怪，但它是**兜底**：只有当重生成结果的语速更接近正常时
    才会被采用（调用方按 |cps - 参考| 择优），不会让成品变差。
    """
    t = text.strip()
    if len(t) < 2:
        return [t]
    mid = len(t) // 2

    def _cut_at(positions):
        """在给定候选切点中取离中点最近的那个，两半都要够长。"""
        if not positions:
            return None
        best = min(positions, key=lambda p: abs(p - mid))
        left, right = t[:best].strip(), t[best:].strip()
        if len(left) >= min_chars and len(right) >= min_chars:
            return [left, right]
        return None

    for pat in (SENT_END, CLAUSE_END):
        got = _cut_at([m.end() for m in pat.finditer(t)])
        if got:
            return got
    got = _cut_at([i + 1 for i, c in enumerate(t) if c.isspace()])
    if got:
        return got
    # 兜底：中点硬切
    return [t[:mid].strip(), t[mid:].strip()]


# ============================ 长文案自动分块 ============================

CHUNK_MAX_CHARS = 2600      # 单块字数上限（对齐 elevenlabs 习惯）
CHUNK_MIN_TAIL_RATIO = 0.30  # 尾块短于上限的这个比例时，改为均匀分块


def split_chunks(text: str, max_chars: int = CHUNK_MAX_CHARS,
                 min_tail_ratio: float = CHUNK_MIN_TAIL_RATIO) -> list:
    """把整段长文案切成若干「块」，每块独立生成一个音频文件。

    三条硬规则（员工原话：「不能固定按照字数，不能从语句中间截断，
    要区分好断句，避免前后衔接不上」）：

      1. 只在句末标点（. ! ? 。！？…）或段落边界（\\n\\n）处断开，
         **绝不切在句子中间**。段落边界优先级高于句末——段落是天然的语义单元。
      2. 2600 是**上限不是刻度**。累积到接近上限时，在上限之前最近的句末断开，
         块长在 2600 上下浮动是正常结果，不是 bug。
      3. 避免头重脚轻：最后一块若短于上限的 30%（<780 字），改为**均匀分块**，
         避免出现 2600+2600+100 这种分布。

    实测预期：
      4091 字 -> ~2600 + ~1491      （尾块 1491 > 780，按上限切）
      5300 字 -> ~1767 × 3          （尾块仅 100 < 780，触发均匀分块）

    注意与 build_segments 的分工：本函数是**块级**（产出独立文件），
    build_segments 是**段级**（块内按句切小段批量生成）。不要混用。
    """
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]

    # 单元列表：每项是 (句子文本, 所属段落序号)
    units = []
    for pi, para in enumerate(re.split(r'\n\s*\n', t)):
        for s in split_sentences(para):
            units.append((s, pi))
    if len(units) <= 1:
        # 全文没有句末标点（无换行无句号的粘贴文本），只能整块交给段级切分
        return [t]

    def _render(group):
        """把一组 (句, 段号) 还原成文本：段内空格连，段间空行分隔。"""
        out = []
        prev_pi = None
        for s, pi in group:
            if prev_pi is None:
                out.append(s)
            elif pi == prev_pi:
                out.append(" " + s)
            else:
                out.append("\n\n" + s)
            prev_pi = pi
        return "".join(out)

    def _greedy(limit):
        """贪心累积到 limit 就断，断点必然在句末。"""
        chunks, buf = [], []
        for s, pi in units:
            cand = buf + [(s, pi)]
            length = sum(len(x) for x, _ in cand) + len(cand) - 1
            if buf and length > limit:
                chunks.append(_render(buf))
                buf = [(s, pi)]
            else:
                buf = cand
        if buf:
            chunks.append(_render(buf))
        return [c for c in chunks if c.strip()]

    chunks = _greedy(max_chars)
    # 规则 3：尾块过短 -> 均匀分块重来一次
    if len(chunks) >= 2 and len(chunks[-1]) < max_chars * min_tail_ratio:
        n = len(chunks)
        target = int(len(t) / n) + 1
        balanced = _greedy(max(target, 1))
        # 均匀分块必须仍满足上限，否则说明单块就装不下，保持原切法
        if balanced and max(len(c) for c in balanced) <= max_chars:
            chunks = balanced
    return chunks


def make_silence(ms: float, sr: int = 24000) -> np.ndarray:
    """生成静音片段。"""
    n = int(sr * ms / 1000.0)
    return np.zeros(n, dtype=np.float32)


def pause_for(seg: dict, base_ms: float, para_ms: float, jitter: float,
              rng: random.Random) -> float:
    """根据片段在句/段中的位置决定后续停顿时长，带抖动。"""
    ms = para_ms if seg.get("is_para_end") else base_ms
    # 句末标点越长停得越久
    tail = seg["text"].rstrip()
    if tail.endswith(("?", "？")):
        ms *= 1.15
    elif tail.endswith(("!", "！")):
        ms *= 1.05
    elif tail.endswith(("…",)):
        ms *= 1.3
    if jitter > 0:
        ms *= 1.0 + rng.uniform(-jitter, jitter)
    return ms


def concat_segments(audios: list, segs: list, sr: int = 24000,
                    base_ms: float = 320.0, para_ms: float = 620.0,
                    jitter: float = 0.18, seed: int = None) -> np.ndarray:
    """把各段音频按真人停顿节奏拼接。

    audios: 与 segs 一一对应的 float32 numpy 数组
    """
    rng = random.Random(seed)
    parts = []
    for i, (a, seg) in enumerate(zip(audios, segs)):
        parts.append(a)
        if i < len(audios) - 1:
            parts.append(make_silence(pause_for(seg, base_ms, para_ms, jitter, rng), sr))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def apply_speed_jitter(audio: np.ndarray, factor: float) -> np.ndarray:
    """轻微变速，打破 AI 语音过于恒定的节奏感。factor 建议 0.97~1.03。

    注意：输出长度必须随 factor 变化（n/factor），否则 factor>1 时源索引越界
    被 clip 到末尾，尾部会变成重复最后样本的常数段（拖尾/咔哒）。
    """
    if abs(factor - 1.0) < 1e-4:
        return audio
    n = len(audio)
    n_out = max(int(round(n / factor)), 1)
    src_idx = np.arange(n_out, dtype=np.float32) * factor
    src_idx = np.clip(src_idx, 0, n - 1)
    return np.interp(src_idx, np.arange(n, dtype=np.float32), audio).astype(np.float32)


def _energy_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    return 20.0 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


def prosody_factors(n: int, seed=None) -> list:
    """为全文各段生成语速因子，模拟真人朗读的"呼吸曲线"。

    真人念长稿不会全程匀速：起句偏稳略慢、中段进入状态稍快、收尾自然放缓，
    再叠加一点点随机微扰。AI 味很大一部分来自"从头到尾一模一样的速度"。
    返回因子 >1 表示更快（时长更短），幅度控制在 ±3% 内，听得出来但不夸张。
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        pos = 0.5 if n <= 1 else i / (n - 1)
        if pos < 0.15:                                   # 起句：偏稳、略慢
            f = 0.975 + (pos / 0.15) * 0.040
        elif pos < 0.85:                                 # 中段：进入状态，稍快
            f = 1.015 + ((pos - 0.15) / 0.70) * 0.012
        else:                                            # 收尾：放缓收敛
            f = 1.027 - ((pos - 0.85) / 0.15) * 0.045
        f *= 1.0 + rng.uniform(-0.022, 0.022)            # 逐段微扰，打破机械感
        out.append(round(float(f), 4))
    return out


def apply_gain(audio: np.ndarray, db: float) -> np.ndarray:
    """轻微增益调整（dB）。用于段落间的音量起伏，真人不会全程一个响度。"""
    if abs(db) < 1e-3:
        return audio
    return (audio * (10.0 ** (db / 20.0))).astype(np.float32)


def trim_edges(audio: np.ndarray, sr: int = 24000, thresh_db: float = -45.0,
               frame_ms: float = 20.0, keep_ms: float = 120.0) -> np.ndarray:
    """削掉片段首尾多余静音，只保留 keep_ms 的自然余量。

    AR 模型常在片段首尾生成 0.5~1.5s 的空白，若再叠加拼接停顿，
    段与段之间会出现 2s 以上的死寂，听感非常拖沓。
    """
    if audio.size == 0:
        return audio
    fr = max(int(sr * frame_ms / 1000.0), 1)
    n = len(audio) // fr
    if n < 4:
        return audio
    e = np.array([_energy_db(audio[i * fr:(i + 1) * fr]) for i in range(n)])
    loud = np.where(e > thresh_db)[0]
    if loud.size == 0:
        return audio[:int(sr * keep_ms / 1000.0)]
    keep_frames = int(keep_ms / frame_ms)
    a = max(int(loud[0]) - keep_frames, 0)
    b = min(int(loud[-1]) + 1 + keep_frames, n)
    return audio[a * fr:b * fr]
