# -*- coding: utf-8 -*-
"""呼吸声合成（C 方案）—— 在段间停顿处插入吸气声。

## 为什么必须合成

2026-10-10 诊断：**案例原声里根本没有呼吸**。静音段的 `rel_db` 分布单峰连续
（−19.9 → −10.2 dB，没有「纯静音 / 呼吸」双峰），最响的静音段 `flat=0.002`、
`zcr=0.037` —— 是残留信号，不是宽带气流噪声。原因是案例本身由 ElevenLabs
生成，TTS 本来就不合成呼吸。

所以「呼吸自然」这条需求**从案例那里拿不到参照物**，只能主动合成。
用户拍板走 C 方案（主动合成，位置结合案例的停顿分布）。

## 依据

INTERSPEECH 2024（Yang/Koriyama/Saito, arXiv:2402.00288）：
- 呼吸声 = 有气流摩擦的**宽带噪声**：过零率高、频谱平坦、能量明显低于语音
- 用 duration + ZCR + VMS + NA-VMS 四个声学特征即可检出，IoU 0.836
- 主观实验：带呼吸的 TTS 自然度 **MOS 3.55 > 不带 3.34**

## 设计取舍

1. **只插在长停顿**：换气发生在长停顿。短停顿（逗号级 0.2~0.3s）插呼吸会
   显得喘不过气。默认要求停顿 ≥ 380 ms。
2. **放在停顿后半段**：真人节奏是「说完 → 停 → 吸气 → 接着说」，
   所以呼吸贴近下一句起点，而不是停顿正中。
3. **频率可控**：默认 ~10 次/分，用累计时长闭环控制，不会随文案长短漂。
4. **电平很轻**：默认 −22 dB 相对语音。宁可听不出也不要听成噪声
   （案例测得的静音段残留约 −11 dB，但我们不确定那是呼吸还是底噪，
    所以取更保守的值）。
5. **可一键关闭**：`TTS_BREATH_SYN=0`。

⚠️ 边界：这是**合成**的呼吸，不是从案例克隆的。它解决「停顿处死寂、
一口气念到底」的听感，但不保证与某个具体案例的呼吸习惯一致 ——
因为案例压根没有呼吸可供对齐。
"""
import os

import numpy as np

# ---- 可调参数（环境变量，改完需重启服务）----
ENABLED = os.environ.get("TTS_BREATH_SYN", "1").strip() not in ("0", "", "false", "False")
PER_MIN = float(os.environ.get("TTS_BREATH_PER_MIN", "10.0"))       # 目标呼吸频率
# 相对语音的电平。注意 voice_rms 取的是 **75 分位以上** 那部分（较响的），
# 所以换算到"相对整体有声 rms"要再低约 5 dB：−25 → 实际约 −20 dB，
# 落在真人呼吸的 −20~−25 dB 区间。宁可轻到听不出，也别听成底噪。
GAIN_DB = float(os.environ.get("TTS_BREATH_GAIN_DB", "-25.0"))
MIN_GAP_MS = float(os.environ.get("TTS_BREATH_MIN_GAP_MS", "380"))  # 只在这之上的停顿换气
MIN_SPACING_SEC = float(os.environ.get("TTS_BREATH_MIN_SPACING", "2.2"))  # 两次呼吸最小间隔
DUR_MS = float(os.environ.get("TTS_BREATH_DUR_MS", "220"))          # 呼吸时长
TAIL_PAD_MS = float(os.environ.get("TTS_BREATH_TAIL_PAD", "70"))    # 末尾留白，避免顶到下一句
F_LO = float(os.environ.get("TTS_BREATH_F_LO", "260.0"))            # 带通下限
F_HI = float(os.environ.get("TTS_BREATH_F_HI", "3200.0"))           # 带通上限


def _randn(rng, n: int) -> np.ndarray:
    """兼容两种 rng。

    🔴 踩过：调用方 `pipeline.concat_segments` 用的是标准库 `random.Random`
    （`random.Random(seed)`），它**没有** `standard_normal`，
    直接调会抛 AttributeError 让整个生成任务失败。
    """
    if hasattr(rng, "standard_normal"):          # numpy Generator
        return np.asarray(rng.standard_normal(n), dtype=np.float64)
    if hasattr(rng, "gauss"):                    # random.Random
        return np.fromiter((rng.gauss(0.0, 1.0) for _ in range(n)),
                           dtype=np.float64, count=n)
    return np.random.standard_normal(n)          # 兜底


def synth_inhale(n: int, sr: int, rng, f_lo=F_LO, f_hi=F_HI) -> np.ndarray:
    """合成一段吸气声：带通白噪声 + 快起慢落包络。

    吸气（inhale）比呼气更常出现在句间换气的听感里，且频谱更"粗"，
    用带通噪声模拟气流摩擦即可，不需要真实录音。
    """
    if n <= 8:
        return np.zeros(max(n, 0), dtype=np.float32)
    x = _randn(rng, n)
    # 频域带通（直接置零最简单可控，边缘用余弦过渡避免振铃）
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    m = np.zeros_like(fr)
    m[(fr >= f_lo) & (fr <= f_hi)] = 1.0
    w = max(int(0.02 * len(fr)), 1)
    m = np.convolve(m, np.ones(w) / w, mode="same")
    X = X * m
    x = np.fft.irfft(X, n)
    # 包络：15% 快速起音 + 指数衰减，末尾淡出到 0（避免被硬切产生咔哒）
    t = np.linspace(0.0, 1.0, n)
    a = 0.15
    env = np.where(t < a, t / a, np.exp(-(t - a) / (1.0 - a) * 2.2))
    env = env * np.clip((1.0 - t) / 0.12, 0.0, 1.0)      # 末尾 12% 淡出
    x = x * env
    pk = float(np.max(np.abs(x)))
    if pk > 1e-9:
        x = x / pk
    return x.astype(np.float32)


def make_breath_silence(ms: float, sr: int, rng, voice_rms: float,
                        gain_db: float = GAIN_DB, dur_ms: float = DUR_MS,
                        tail_pad_ms: float = TAIL_PAD_MS) -> np.ndarray:
    """生成一段「带呼吸的静音」。

    voice_rms：参考语音的 RMS，用来把 gain_db 换算成绝对幅度。
    呼吸放在停顿的后半段（贴近下一句起点）。
    """
    n = int(sr * ms / 1000.0)
    out = np.zeros(max(n, 0), dtype=np.float32)
    if n < 64 or voice_rms <= 0:
        return out
    dn = int(sr * min(dur_ms, max(ms - tail_pad_ms - 40.0, 60.0)) / 1000.0)
    pad = int(sr * tail_pad_ms / 1000.0)
    dn = min(dn, max(n - pad - 8, 32))
    start = max(n - pad - dn, 0)
    br = synth_inhale(dn, sr, rng)
    amp = float(voice_rms) * (10.0 ** (gain_db / 20.0))
    out[start:start + dn] = br * np.float32(amp)
    return out


def sil_gaps(x: np.ndarray, sr: int, min_sil: float = 0.18):
    """检测低能量区间。返回 [(start, end), ...]（采样点）。

    🔴 与 tools/prosody_target.sil_gaps 同口径（pct(db,25)+6），保证
    「案例目标分布」和「产物实测」两边量的是同一件事。
    """
    hop, win = int(0.01 * sr), int(0.03 * sr)
    n = max((len(x) - win) // hop, 1)
    rms = np.sqrt(np.array([np.mean(x[i * hop:i * hop + win] ** 2)
                            for i in range(n)]) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    sil = db < (np.percentile(db, 25) + 6.0)
    gaps, i = [], 0
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]:
                j += 1
            if (j - i) * hop / sr >= min_sil:
                gaps.append((i * hop, min(j * hop, len(x))))
            i = j
        else:
            i += 1
    return gaps


LAST_N = 0        # 上一次 inject 实际注入了几处（调试/回归用）


def inject(x: np.ndarray, sr: int, rng, per_min: float = PER_MIN,
           gain_db: float = GAIN_DB, min_gap_ms: float = MIN_GAP_MS,
           spacing: float = MIN_SPACING_SEC) -> np.ndarray:
    """在**已有**长停顿里叠加呼吸，不改变音频长度。

    🔴 为什么必须在 PBA 之后调用：
    第一版把呼吸插在 `pipeline.concat_segments`（段间拼接处），实测只有
    0.3 次/分 —— 因为 server.py 后段的 **PBA align + reshuffle 会重写停顿**，
    把刚插进去的呼吸整段覆盖掉。挪到 PBA 之后才真正生效。

    为什么用"叠加"而不是"替换"：不动长度就不会影响后面的响度归一、
    时长对齐、atempo 等任何一步，风险最小。
    """
    global LAST_N
    LAST_N = 0
    if not ENABLED or per_min <= 0:
        return x
    y = np.asarray(x, dtype=np.float32).copy()
    gaps = sil_gaps(y, sr)
    if not gaps:
        return y
    v = np.abs(y)
    voiced = v[v > np.percentile(v, 75)] if v.size else v
    voice_rms = float(np.sqrt(np.mean(voiced ** 2))) if voiced.size else 0.0
    if voice_rms <= 0:
        return y
    plan = Plan(per_min=per_min, min_gap_ms=min_gap_ms, spacing=spacing)
    picked = []
    for (a, b) in gaps:
        gap_ms = (b - a) / float(sr) * 1000.0
        # elapsed 用停顿起点近似（只用于频率闭环，精度够）
        plan.elapsed = a / float(sr)
        plan.since = (a / float(sr)) - (picked[-1][1] / float(sr) if picked else -1e9)
        if plan.want(gap_ms):
            picked.append((a, b))
            plan.count += 1
    for (a, b) in picked:
        gap_ms = (b - a) / float(sr) * 1000.0
        br = make_breath_silence(gap_ms, sr, rng, voice_rms, gain_db=gain_db)
        n = min(len(br), b - a)
        y[a:a + n] += br[:n]
    LAST_N = len(picked)
    return y


class Plan:
    """按累计时长闭环控制呼吸频率，避免随文案长短漂移。"""

    def __init__(self, per_min=PER_MIN, min_gap_ms=MIN_GAP_MS,
                 spacing=MIN_SPACING_SEC, enabled=ENABLED):
        self.per_min = max(per_min, 0.0)
        self.min_gap_ms = min_gap_ms
        self.spacing = max(spacing, 0.0)
        self.enabled = enabled and per_min > 0
        self.elapsed = 0.0     # 已拼接秒数
        self.since = 1e9       # 距上次呼吸的秒数
        self.count = 0

    def advance(self, seconds: float):
        self.elapsed += max(seconds, 0.0)
        self.since += max(seconds, 0.0)

    def want(self, gap_ms: float) -> bool:
        """这个停顿要不要插呼吸。"""
        if not self.enabled:
            return False
        if gap_ms < self.min_gap_ms:
            return False
        if self.since < self.spacing:
            return False
        # 闭环：按已用时长算"此刻应该已经呼吸了几次"，落后就补一次
        want_n = int(self.elapsed * self.per_min / 60.0)
        return self.count < want_n

    def done(self):
        self.count += 1
        self.since = 0.0
