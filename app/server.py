# -*- coding: utf-8 -*-
"""Qwen3-TTS 服务：FastAPI + 异步任务队列 + Web 界面 + 账号体系。

单 GPU 实例串行推理，多人通过队列共用，避免多实例显存溢出。
账号：管理员 admin 可创建员工账号、查看全部任务；员工只能看自己的。
"""
import os
import io
import csv
import re
import json
import time
import uuid
import hmac
import base64
import hashlib
import secrets
import asyncio
import logging
import subprocess
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response

from tts_engine import TTSEngine
from voice_trim import trim_reference
from voice_match import get_target
from pipeline import CHUNK_MAX_CHARS, split_chunks
import asr_check
import cosy_client
import vram
import engine_select
import pba
import envshape

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("server")

# 数据/模型根目录。默认 /mnt/tts（现网数据盘挂载点）。
# 换服务器时不改代码：设环境变量 TTS_BASE 即可整体迁移
# （systemd 里加 Environment=TTS_BASE=/新路径）。不设则与旧行为完全一致。
BASE = os.environ.get("TTS_BASE", "").strip() or "/mnt/tts"
OUTPUT_DIR = os.path.join(BASE, "output")
DATA_DIR = os.path.join(BASE, "data")
VOICE_DIR = os.path.join(DATA_DIR, "voices")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

USERS_FILE = os.path.join(DATA_DIR, "users.json")
JOBS_FILE = os.path.join(DATA_DIR, "jobs.json")
DEFAULT_ADMIN = "admin"
# 出厂口令**不再硬编码**：所有部署共用同一个默认口令是致命问题（尤其仓库含源码）。
# 新环境初始化时：
#   1. 环境变量 TTS_ADMIN_PASSWORD 有值 → 用它（适合自动化部署）
#   2. 没有 → 随机生成一个 12 位口令，打到服务日志里
# 两种情况都会给账号打上 must_change，首次登录后强制修改。
# 现网 users.json 已存在时，这段逻辑不会执行，不影响已有账号。
DEFAULT_PASSWORD = os.environ.get("TTS_ADMIN_PASSWORD", "").strip()
SESSION_TTL = 7 * 24 * 3600
MIN_PW_LEN = 6

# ---------------------------------------------------------------------------
# 参考音频（声音克隆）硬性约束
#
# 克隆走 ICL 模式：参考音频会被 codec 编成 12.5 帧/秒的 token，连同文字稿
# 一起作为 prompt 前缀，并且 **batch 内每个样本各背一份**（voice_clone_prompt
# = [prompt] * n）。所以参考音频越长，显存按 batch 倍数放大。
#
# 实测（Qwen3-TTS-12Hz-1.7B-Base，服务端真机读取 config.json + tokenizer）：
#   329.33s + 全文稿 ≈ 5 420 prompt tokens → 593 MiB/样本 → batch8 需 4.63 GiB，
#   而模型常驻后仅剩 4.14 GiB ⇒ 必 OOM，只能降到 batch2，耗时翻 1.7~2.2 倍。
#   裁到 25s ≈ 430 tokens → 47 MiB/样本 → batch8 仅 0.37 GiB，吞吐零损失。
#
# 30 秒上限对应的最坏情况：375 codec + 约 185 字 × 0.634 ≈ 117 → 约 516 tokens
# → 56 MiB/样本 → batch8 仅 0.44 GiB，仍在安全区内，且不会拖慢生成。
#
# 但**不让员工自己剪**：员工只管上传整段录音 + 整段文字稿，超过 AUTO_TRIM_ABOVE
# 的由 voice_trim 自动挑最清晰的一段（约 TARGET 秒），并把文字稿按累计有声时长
# 对齐截出对应片段。分析一次的耗时在秒级，之后该音色就是现成的成品。
# ---------------------------------------------------------------------------
REF_AUDIO_MIN_SEC = 3.0           # 短于这个没法克隆
REF_AUDIO_MAX_SEC = 30.0          # 成品音色的硬上限（自动裁剪后必然满足）
REF_AUDIO_BEST_SEC = 15.0         # 给员工的建议长度（现在由代码自动达成）
REF_AUDIO_TRIM_TARGET = 15.0      # 自动截取的目标长度
REF_AUDIO_AUTO_TRIM_ABOVE = 30.0  # 超过这个长度才触发自动截取
REF_AUDIO_HARD_MAX_SEC = 900.0    # 超过 15 分钟直接拒（读进来太浪费内存）
# 克隆「保真增强」默认开关：把生成结果频谱对齐到音色原声，补回发闷的中高频。
# 关掉只会回到发闷的老输出，不影响其它环节。设 TTS_VOICE_MATCH=0 可整体关闭。
VOICE_MATCH_DEFAULT = os.environ.get("TTS_VOICE_MATCH", "1").strip().lower() \
    not in ("0", "false", "no", "off")

# 自动分块：整段文案按句子边界切成若干块，每块独立出一个 mp3。
# 上限对齐员工在 elevenlabs 时代的习惯（单条 ≤2600 字）；实际块长会在上限
# 上下浮动，因为断点必须落在句末——详见 pipeline.split_chunks。
CHUNK_MAX_CHARS = int(os.environ.get("TTS_CHUNK_CHARS", "2600") or 2600)
# ASR 内容回检不通过时最多重掷几次（0 = 只跑一次不重掷）
ASR_MAX_RETRY = int(os.environ.get("TTS_ASR_RETRY", "2") or 2)

# ---- 语速调节 ----------------------------------------------------------
# Qwen3-TTS 模型本身**没有** speed / length_scale 接口（已核对源码：
# qwen_tts/inference/qwen3_tts_model.py 里 grep speed|length_scale|duration|prosody
# 零命中），所以只能在后处理变速。
# 用 ffmpeg atempo（WSOLA 类算法，变速不变调）在**成品 wav**上做一次，
# 实测几乎无损（LTAS 误差 0.05dB，4-7k 能量保留 1.005），
# 远优于此前 np.interp 的线性插值（丢 4-7k 42% 能量）。
SPEED_MIN, SPEED_MAX = 0.5, 2.0          # atempo 单次支持的范围
MP3_BITRATE = os.environ.get("TTS_MP3_BITRATE", "192k")   # 旧值 128k，对齐 elevenlabs

# ---- 情感预设 → 引擎自动分流 --------------------------------------------
# 员工只选「情感」，引擎/指令/参数全部由后端决定。依据（同文本 776 字实测）：
#   Qwen3-TTS       CER 0.0542  —— 最准，但模型无 instruct，情感不可控
#   CosyVoice3      CER 0.0829  —— 略逊，但有 instruct，情感可调，且更快更省显存
# 所以「平稳/标准」走 Qwen 保准确度；要情绪的走 CosyVoice3。
EMOTION_PRESETS = {
    "standard": {"engine": "qwen",       "instruct": ""},
    "excited":  {"engine": "cosyvoice3", "instruct": "excited"},
    "warm":     {"engine": "cosyvoice3", "instruct": "warm"},
    "serious":  {"engine": "cosyvoice3", "instruct": "serious"},
}
DEFAULT_EMOTION = "standard"
# CosyVoice3 出片后仍走 ASR 内容回检；CER 超过这个阈值就自动回退 Qwen 重出，
# 保证「加了情绪也不能把字念错」。
COSY_FALLBACK_CER = float(os.environ.get("TTS_COSY_FALLBACK_CER", "0.15"))
# 停顿预算对齐（PBA）：把 AR 模型涌现的冗余停顿按比例压缩，再用 atempo 把
# 总时长拉回。α=0.8 是 2026-09-30 全篇（男 1999 字 / 女 1773 字）网格搜索
# 定下来的：男声句数 96→72（案例 70）、女声 42→24（案例 23），9 项韵律指标
# 里 8 项改善。设成 1.0 即完全关闭（等价于旧行为）。详见 app/pba.py 文件头。
PBA_ALPHA = float(os.environ.get("TTS_PBA_ALPHA", "0.8"))
# 低于这个字数不值得做（太短的片段本来就没有几句，压缩了反而显得赶）。
PBA_MIN_CHARS = int(os.environ.get("TTS_PBA_MIN_CHARS", "120"))
# ---- 停顿重排：修「卡顿」的另一半（等长停顿 = 机械感）------------------
# α=0.8 把句数修对了，但把 69 个停顿压成了同一个长度（sd 0.066 vs 案例 0.109、
# ≥0.45s 的长停顿 2 个 vs 案例 27 个），听起来像节拍器。reshuffle 按对数正态
# 做保序分位映射，把停顿拉回「有长有短」，目标均值由句均时长回归给出
# （mu = 0.463 - 0.0143×句均时长），所以只修男声、女声基本不动。
# 详见 app/pba.py 的 reshuffle() 文件头。
PBA_RESHAPE = os.environ.get("TTS_PBA_RESHAPE", "1") != "0"
PBA_CV = float(os.environ.get("TTS_PBA_CV", "0.28"))

# ---- 包络整形：修「咬字含糊（舌头没展开）」--------------------------------
# 2026-09-30 员工反馈「听着跟舌头没展开似的」，用调制谱定位到真因：
#     8~16Hz 调制能量（音节内瞬态 = 咬字清晰度）
#         案例男 0.1328  我们 0.0562  **-57%**
#         案例女 0.1099  我们 0.0967  -12%
#   声码器把辅音爆破/送气的瞬态抹平了，字就"糊"在一起。这跟频谱无关，
#   所以前面那套 EQ 完全管不到——必须到**时间维的包络**上去修。
#
# 强度是**自适应**的（二分求最小 sharp 使 8~16Hz 达标）：男女声差距很大
# （男声差 57%、女声只差 12%），写死一个强度必然有一个性别过度。
#   实测：男 sharp 0.72 -> 8~16Hz 0.0562→0.1272，n_sent 67→70（案例 70）
#         女 sharp 0.16 -> 8~16Hz 0.0967→0.1165，n_sent 22 不变（案例 23）
#
# ⚠️ `ENV_COMP`（压 1~4Hz 顿挫）默认 0：它能把男声 1~4Hz 从 0.55 降到 0.43，
#    但代价是 n_sent 掉到 63。先把最要紧的"咬字"修了，顿挫留待试听后再定。
ENV_SHAPE = os.environ.get("TTS_ENV_SHAPE", "1") != "0"
# ⚠️ 0.15 是**过补偿**值，不是案例值（案例男 0.1328 / 女 0.1099）。
# 原因：整形在 mp3 之前，而 lame 会把 8~16Hz 削掉一部分
#   （实测：整形后原始 0.2531 -> 成品 0.1769；0.1232 -> 0.0813）。
#   案例之所以 mp3 零损失（wav 0.1328 / mp3 0.1327），是因为它的瞬态是
#   **真实的频谱突变**，lame 会切短块保护；我们改的是全带包络，骗不过它，
#   只能靠过补偿把 mp3 削完剩下的部分抬到案例水平。
#
# ⚠️⚠️ 0.20 定得太大，r9 全篇实测**过冲**（调制深度绝对值，dB）：
#       案例男 -50.86 / r9 男 -47.71（+3.15dB）  案例女 -48.14 / r9 女 -48.07
#   按两点线性外推，0.15 时男 ≈ -49.85（+1.0dB）、女 ≈ -49.76（-1.6dB），
#   男女各偏 1.5dB 且方向相反 —— 这是单一全局目标能拿到的最好折中：
#   男女声的「原始瞬态量」和「案例目标量」都不同，写死一个值必然一个性别过。
#   宁可偏脆也别偏糊（员工原话是「舌头没展开」），所以取 +1/-1.6 而不是 -3/+3。
ENV_TARGET = float(os.environ.get("TTS_ENV_TARGET", "0.15"))
ENV_COMP = float(os.environ.get("TTS_ENV_COMP", "0.0"))
ENV_MIN_CHARS = int(os.environ.get("TTS_ENV_MIN_CHARS", "120"))

# ---- 音色对齐：修「空旷感 / 薄 / 贴脸」--------------------------------
# 2026-09-30 全篇实测，我们比案例原声（ElevenLabs 44.1kHz）薄且响：
#     100~300Hz 占比   男 13.75% vs 25.44%（-46%）  女 9.61% vs 14.45%（-33%）
#     1000~3000Hz 占比 男 28.32% vs 20.87%（+36%）  女 17.58% vs 11.57%（+52%）
#     整体 LUFS        男 -20.2 vs -25.0            女 -20.0 vs -23.0
# 低频（胸腔感）少一半、中高频多三到五成、整体还响 3~4.8dB ——
# 三者合起来就是员工说的「空旷感」：薄、飘、贴脸、没有实体感。
#
# 下面是 18 段图形 EQ（superequalizer），中心频率依次为
# 65/92/131/185/262/370/523/740/1047/1480/2093/2960/4186/5920/8372/11840/
# 16744/20000 Hz。⚠️ 取值是**线性倍数**不是 dB（ffmpeg 的定义如此）。
# 由 1/3 倍频程功率差拟合而来，只保留「男声和女声同向」的那部分（系统性偏差）；
# 11840Hz 以上一律归 1 —— 我们 24kHz 输出在 12kHz 就截止了，
# 案例 44.1k mp3 在那里的能量只是编码噪声，照搬会变成沙沙声。
# 效果（修正后 vs 案例）：男 100-300Hz 21.72 vs 25.44、1-3k 21.65 vs 20.87、
# LUFS -23.4 vs -25.0；女 15.20 vs 14.45、14.40 vs 11.57、-23.5 vs -23.0。
TONE_EQ_FILTER = os.environ.get(
    "TTS_TONE_EQ_FILTER",
    "superequalizer=1b=1.585:2b=1.778:3b=1.638:4b=1.428:5b=1.393:6b=1.283:"
    "7b=1.116:8b=1.012:9b=0.977:10b=0.977:11b=0.955:12b=1.000:13b=1.059:"
    "14b=1.143:15b=1.259:16b=1.0:17b=1.0:18b=1.0").strip()
TONE_EQ = os.environ.get("TTS_TONE_EQ", "1") != "0"

# ---- 高频空气感：补 11~13.5 kHz 的"亮" ----------------------------------
# ⚠️ 这是第二轮才查出来的，和上面的低频补偿是两件事。
#
# ⚠️⚠️ 先说一个**差点让我们调错 30 dB 的测量陷阱**（2026-09-30）：
#   早先拿 24 kHz 的成品 mp3 去量，11.8 kHz 是 -126 dB，看起来是"砖墙"。
#   但那是 **lame 在 24 kHz 输出时把 11.2 kHz 以上一刀切掉**造成的假象！
#   用 `TTS_AIR=0` 直接出 44.1 kHz 成品一量才看到真实值：
#     女 11.9k -44.8 / 12.6k -51.3   男 11.9k -58.6 / 12.6k -59.4
#   → 真实缺口只有 10~25 dB（不是 90 dB）。**凡是要量 11 kHz 以上，
#     必须先出 44.1 kHz 成品，绝不能拿 24 kHz 的文件量。**
#
# 真实缺口（相对案例，dB）：8~11 kHz 差 1~5；11.9k 差 10~24；
# 13.3k 差 24~36；14 kHz 以上差 57~67（那里案例是真实气声，我们从零
# 合成 60 dB 只会变成沙沙声，**放弃**，接受这一段短）。
#
# 参数来源：`/tmp/tune3.py` 在**线上真实原始音频**上做的三轮网格搜索，
# 评分 = 11~13.5k 误差 + 2×14k 以上超标（沙沙声）+ 1×7~11k 超标（齿音）。
# 胜出组合：女声误差 1.9 dB、男声 8.5 dB，且两性别齿音/沙沙都 ≈ 0。
#
# ⚠️ 为什么不能用 `firequalizer`（FIR 任意曲线）：
#    曲线太陡 → 停顿底噪 +1 dB、峰值 +0.5 dB，而我们的停顿普遍只有
#    0.26~0.29 s、紧贴 0.25 s 的判句阈值，女声 n_sent 从 22 掉到 19。
# ⚠️ `crystalizer` 更不能用：实测 n_sent 直接崩到 10~13。
# ⚠️ 顺序不能反：必须**先重采样到 44.1 kHz 再进后面的滤波器**。
AIR_EXCITER = os.environ.get(
    "TTS_AIR_EXCITER", "aexciter=amount=5:freq=11200:ceil=13000").strip()
AIR_EQ = os.environ.get(
    "TTS_AIR_EQ",
    "equalizer=f=10000:t=h:w=3200:g=-6,"
    "equalizer=f=12300:t=h:w=2500:g=8,"
    "equalizer=f=13300:t=h:w=2000:g=8").strip()
AIR = os.environ.get("TTS_AIR", "1") != "0"
# 出片采样率。必须 ≥44.1 kHz，否则 12 kHz 以上的空气感无处安放
# （24 kHz 的奈奎斯特就是 12 kHz）。案例原声就是 44.1 kHz。
OUT_SR = int(os.environ.get("TTS_OUT_SR", "44100"))

# 整体响度拉到案例水平（案例 LUFS -25.0 / -23.0，我们原样 -20.2 / -20.0）。
# -4.5 里含 0.5 dB 是抵掉 aexciter/firequalizer 带来的能量增加。
# 设 0 即不调整。注意这会让人觉得"变小声了"，但案例就是这个响度。
OUT_GAIN_DB = float(os.environ.get("TTS_OUT_GAIN_DB", "-4.5"))
# ---- 分性别响度偏移：修男声「像吵架」--------------------------------------
# 2026-09-30 员工听后感：「男声前半段有点跟吵架似的，女声的还算好点」。
# 逐 20s 窗测有声段 RMS（/tmp/win.py），案例男声是**全程都比我们小**：
#
#     有声段响度    案例男 -9.8 dB    我们 r10 -7.5 dB  （+2.3 dB，偏吵）
#     有声段响度    案例女 -8.5 dB    我们 r10 -8.9 dB  （-0.4 dB，正常）
#     2~8k 锐度     案例男 0.1216     我们 0.1602（+32%）   ← 男女都偏，不是主因
#     attack/s      案例男 7.93       我们 8.33（+5%）
#
# 锐度男女都高 25~32% 但只有男声被投诉 → **主因是响度，不是音色**。
# 而且案例男声的响度在 -8~-11 之间起伏，我们恒定在 -6~-8 ——「全程满音量」
# 本身就听着像在嚷。
#
# ⚠️ 这是两点拟合（男 -24.9 / 女 -22.9 LUFS），不是定理；用 f0 判性别，
#    判不出来就用 0 dB。想关掉设 TTS_GAIN_GENDER=0。
GAIN_GENDER = os.environ.get("TTS_GAIN_GENDER", "1") != "0"
GAIN_MALE_DB = float(os.environ.get("TTS_GAIN_MALE_DB", "-1.7"))
GAIN_FEMALE_DB = float(os.environ.get("TTS_GAIN_FEMALE_DB", "0.5"))
F0_SPLIT_HZ = float(os.environ.get("TTS_F0_SPLIT_HZ", "220"))

# ---- 块尾分段停顿 ----------------------------------------------------------
# 案例（ElevenLabs）的段落之间是 0.45~0.66 s 的真换气；我们两块 mp3 拼起来
# 中间只有编码残留的 0.1~0.2 s 数字零，员工听起来就是"没有分段"。
# 给非最后一块补到 600 ms（与 pipeline.para_ms=620 同一量级），置 0 可关。
CHUNK_TAIL_PAUSE_MS = float(os.environ.get("TTS_CHUNK_TAIL_PAUSE_MS", "600"))
CHUNK_TAIL_SIL_DB = float(os.environ.get("TTS_CHUNK_TAIL_SIL_DB", "-45"))


def est_f0_hz(y, sr, sec=20.0, fmin=60.0, fmax=400.0):
    """估基频中位数，只用来判男女。判不出返回 None。

    ⚠️ 别用「纯 numpy 自相关」那版：实测女声会系统性读低
       （案例女 pyin 269Hz / 自相关 221Hz，我们女声 287/209 —— 209 已经
       跌破 210 的男女分界，会把女声误判成男声砍掉 1.7 dB）。
       改用与 `pros.py` 同口径的 `librosa.pyin`，只跑中间 20 秒，约 1~2 s。
    """
    try:
        import librosa
        y = np.asarray(y, dtype=np.float32).flatten()
        if y.size < 16000:
            return None
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 \
            else y
        n = len(y16)
        a = int(n * 0.25)
        b = min(n, a + int(sec * 16000))
        seg = y16[a:b]
        if len(seg) < 16000:
            seg = y16[:min(n, int(sec * 16000))]
        f0 = librosa.pyin(seg, fmin=fmin, fmax=fmax, sr=16000,
                          frame_length=1024)[0]
        v = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        if len(v) < 20:
            return None
        return float(np.median(v))
    except Exception as e:
        log.warning("est_f0_hz 失败（%s）", repr(e)[:120])
        return None


# 新增音色时是否自动跑「引擎 + 情感」选型，把最优组合绑到这个音色上。
# 之后员工选默认情感（standard）生成时不再写死走 Qwen，而是走该音色的最优组合。
# 置 0 可退回上面的静态映射，便于 A/B 对照。
# ⚠️ 默认关闭。原因（2026-09-29 实测）：选型要连着跑 Qwen + CosyVoice3 两个模型，
# 而 Qwen 常驻 6.89 GiB、CosyVoice 还要 3.2 GiB，12G 卡放不下 —— 两边互相 OOM。
# 在做出「GPU 串行锁 + 选型前先卸载对方模型」之前，不能开着它跑生产，
# 否则员工录入音色时会挤爆正在生成的任务。
AUTO_SELECT_ON_CREATE = os.environ.get("TTS_AUTO_SELECT_ON_CREATE", "0") != "0"

# ---- 音色级引擎：由管理员在录入音色时手动指定 ---------------------------

# ⚠️ 2026-09-29 取消自动选型。原打分 score = -10×CER + 句内动态，

# Qwen 的 CER 极低（0.0047）导致它恒赢，两个内置音色都被判成 qwen/standard，

# 测了等于没测。改为录入时人工指定，行为可预期、可追溯。

DEFAULT_ENGINE = os.environ.get("TTS_DEFAULT_ENGINE", "qwen")

VALID_ENGINES = ("qwen", "cosyvoice3")

MODEL_DIRS = {
    "custom_voice": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    "base": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-Base"),
}
MODEL_DIRS = {k: v for k, v in MODEL_DIRS.items() if os.path.isdir(v)}

# 文件名里不能出现的字符（Windows 也拒绝这套），统一去掉
_UNSAFE_CHARS = re.compile(r'[/\\:*?"<>|\r\n\t]')


def _safe_name(s: str, limit: int = 40) -> str:
    """把标题/文案片段整理成可用作文件名与目录名的字符串。"""
    s = _UNSAFE_CHARS.sub("", (s or "").strip()).strip(". ").strip()
    return s[:limit] or "output"


def _unique_dir(name: str) -> str:
    """在 OUTPUT_DIR 下取一个不重名的目录名（重名则追加 -2、-3 …）。"""
    cand, i = name, 1
    while os.path.exists(os.path.join(OUTPUT_DIR, cand)):
        i += 1
        cand = f"{name}-{i}"
    return cand

app = FastAPI(title="Qwen3-TTS 语音生成")
engine = TTSEngine(MODEL_DIRS, default_variant="custom_voice")

jobs = {}
job_queue: asyncio.Queue = asyncio.Queue()
# 经验值：每段的生成耗时（秒），任务完成后滑动平均更新，用于首批未完成时预估 ETA
SEG_SEC = 6.0
# 端到端 RTF(生成耗时/音频时长)滑动平均，用于首批未出结果时预估总时长。
# 实测长音频 ~0.37，短文案/克隆 ~1.0-1.4，取 0.8 起步。
AVG_RTF = 0.8
# 音频时长估算用的朗读速度（字/秒），实测各语言 4.3~4.9
CHARS_PER_SEC = 4.5
VOICES_FILE = os.path.join(VOICE_DIR, "voices.json")
SESSIONS = {}   # token -> {"user": str, "exp": float}


# ============================ 账号体系 ============================

def hash_pw(pw: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, 120000)
    return base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def verify_pw(pw: str, stored: str) -> bool:
    try:
        s, h = stored.split("$", 1)
        dk = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), base64.b64decode(s), 120000)
        return hmac.compare_digest(base64.b64encode(dk).decode(), h)
    except Exception:
        return False


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.error("读取 %s 失败: %s", path, e)
    return default


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


users = load_json(USERS_FILE, None)
if not isinstance(users, dict) or not users:
    _init_pw = DEFAULT_PASSWORD or secrets.token_urlsafe(9)[:12]
    _src = "环境变量 TTS_ADMIN_PASSWORD" if DEFAULT_PASSWORD else "随机生成"
    users = {
        DEFAULT_ADMIN: {
            "role": "admin",
            "pwd": hash_pw(_init_pw),
            "created_at": time.time(),
            "last_login": None,
            "must_change": True,
        }
    }
    save_json(USERS_FILE, users)
    # 用 warning 级别：这是部署后必须处理的一件事，不该被 INFO 淹没
    log.warning("已初始化默认管理员账号 admin / %s（口令来源：%s）。"
                "请立即登录并修改密码；本行是随机口令的唯一记录，修改后即可丢弃日志。",
                _init_pw, _src)


def persist_jobs():
    slim = {}
    for jid, j in list(jobs.items())[:500]:
        slim[jid] = {k: j.get(k) for k in (
            "id", "title", "owner", "mode", "language", "speaker", "voice",
            "instruct", "status", "created_at", "started_at", "finished_at",
            "progress", "stats", "wav", "mp3", "error", "preview",
            "max_chars", "batch_size", "enhanced", "speed", "subtalker_temperature",
            "emotion", "engine", "auto_applied", "effective_emotion",
            # 分块输出：files 是每块的 mp3 清单，不落盘的话重启后
            # 员工就再也下载不到已生成的音频了
            "files", "chunks", "chunks_total", "chunk", "folder", "seed")}
        # 正文留着，失败时可一键重试
        slim[jid]["text"] = j.get("text") or ""
    save_json(JOBS_FILE, slim)


# ---------------------------------------------------------------------------
# 服务重启时的任务恢复
#
# 旧行为：把 running/queued 一律标成 error「服务重启，任务中断」。
# 问题是几千字的长文案要跑好几分钟（4750 字克隆约 6~7 分钟），一次重启就白等，
# 用户还得手动点重试。现在改成启动后自动重新入队续跑。
#
# 两个保护：
#   1. resumed 计数上限，防止「反复重启 → 反复重跑」打满队列
#   2. 原文丢失（text 为空）的任务无法续跑，只能判死
# ---------------------------------------------------------------------------
RESUME_MAX_TIMES = 3        # 同一任务最多自动续跑几次
PENDING_RESUME: list = []   # restore_jobs 收集，startup 时入队


def restore_jobs():
    data = load_json(JOBS_FILE, {})
    for jid, j in data.items():
        if not j.get("id"):
            j["id"] = jid
        if j.get("status") in ("running", "queued"):
            times = int(j.get("resumed") or 0)
            if not (j.get("text") or "").strip():
                j["status"] = "error"
                j["error"] = "服务重启，任务中断且原文已丢失，无法自动续跑，请重新提交"
            elif times >= RESUME_MAX_TIMES:
                j["status"] = "error"
                j["error"] = (f"服务重启中断，已自动续跑 {times} 次仍未完成。"
                              f"请确认服务稳定后手动重试")
            else:
                j["status"] = "queued"
                j["progress"] = {"done": 0, "total": 0}
                j["started_at"] = None
                j["finished_at"] = None
                j.pop("error", None)
                j.pop("stats", None)
                j["resumed"] = times + 1
                PENDING_RESUME.append(j)
        j.setdefault("owner", "admin")
        j.setdefault("title", "")
        jobs[jid] = j
    if jobs:
        log.info("恢复历史任务 %d 条，其中 %d 条待续跑", len(jobs), len(PENDING_RESUME))


restore_jobs()


def current_user(request: Request) -> Optional[dict]:
    """从 Cookie / Authorization 头解析登录态。"""
    tok = request.cookies.get("tts_session")
    if not tok:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            tok = auth[7:].strip()
    if not tok:
        return None
    s = SESSIONS.get(tok)
    if not s:
        return None
    if s["exp"] < time.time():
        SESSIONS.pop(tok, None)
        return None
    u = users.get(s["user"])
    if not u:
        return None
    return {"username": s["user"], "role": u.get("role", "user")}


def require(request: Request) -> dict:
    u = current_user(request)
    if not u:
        raise HTTPException(401, "未登录")
    return u


def require_admin(request: Request) -> dict:
    u = require(request)
    if u["role"] != "admin":
        raise HTTPException(403, "需要管理员权限")
    return u


def new_session(username: str) -> str:
    tok = secrets.token_urlsafe(32)
    SESSIONS[tok] = {"user": username, "exp": time.time() + SESSION_TTL}
    return tok


# ============================ 音频读写 ============================

def load_voices():
    if os.path.exists(VOICES_FILE):
        with open(VOICES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_voices(v):
    with open(VOICES_FILE, "w", encoding="utf-8") as f:
        json.dump(v, f, ensure_ascii=False, indent=2)


voices = load_voices()


def read_audio(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


# ============================ 任务执行 ============================

async def run_job(job):
    loop = asyncio.get_running_loop()
    job["status"] = "running"
    job["started_at"] = time.time()
    job["stage"] = "preparing"

    ref_audio = None
    ref_text = None
    ref_key = None
    ref_wav_path = None          # CosyVoice3 走 HTTP，需要的是路径而不是数组
    _match_target = None
    # 情感 → 引擎自动分流：员工只选情感，引擎与指令由这里决定
    _emotion = (job.get("emotion") or DEFAULT_EMOTION).strip()
    _preset = EMOTION_PRESETS.get(_emotion) or EMOTION_PRESETS[DEFAULT_EMOTION]
    _engine_name = (job.get("engine") or _preset["engine"]).strip()
    _cosy_instruct = _preset.get("instruct") or "neutral"
    job["engine"] = _engine_name
    job["emotion"] = _emotion
    log.info("任务 %s 情感=%s -> 引擎=%s", job["id"], _emotion, _engine_name)
    # 重启续跑的旧任务没有这个字段，补上服务端默认值，保证行为一致
    if "enhance" not in job:
        job["enhance"] = VOICE_MATCH_DEFAULT
    if job["mode"] == "clone":
        v = voices.get(job["voice"])
        if not v:
            raise RuntimeError(f"音色不存在: {job['voice']}")
        # 提交时已校验过，这里再兜一次：重试/补跑的旧任务同样要给出中文原因，
        # 否则只会看到模型抛的英文 ValueError，且要等到排队轮到才失败。
        ref_text = (v.get("ref_text") or "").strip() or None
        if not ref_text:
            raise RuntimeError(
                f"音色「{job['voice']}」缺少参考文字稿，无法克隆。"
                f"请重新录入该音色，并填写参考音频对应的逐字文字")
        if float(v.get("dur") or 0) > REF_AUDIO_MAX_SEC:
            raise RuntimeError(
                f"音色「{job['voice']}」的参考音频 {v.get('dur')}s 是未裁剪的旧音色"
                f"（超过 {REF_AUDIO_MAX_SEC:.0f}s，显存会暴涨）。"
                f"请用原音频重新录入一次，系统会自动截取最佳片段")
        ref_audio = read_audio(v["path"])
        ref_wav_path = v["path"]
        ref_key = job["voice"]
        # 引擎取「管理员录入该音色时手动指定的」engine 字段（voices.json）。
        # 员工选了非平稳情感时**强制**改走 CosyVoice3 —— Qwen 克隆没有 instruct，
        # 只有 CosyVoice3 带得动情感，否则员工点了「激昂」却听不出任何变化。
        _ve = (v.get("engine") or "").strip()
        if _ve in VALID_ENGINES:
            _engine_name = _ve
        if _emotion != DEFAULT_EMOTION and _engine_name != "cosyvoice3":
            _engine_name = "cosyvoice3"
        _cosy_instruct = _preset.get("instruct") or "neutral"
        job["engine"] = _engine_name
        log.info("任务 %s 音色引擎=%s 情感=%s -> 最终引擎=%s",
                 job["id"], _ve or "未指定", _emotion, _engine_name)
        # 保真增强：取该音色源录音的长时谱作为目标（详见 app/voice_match.py）
        if job.get("enhance"):
            try:
                _match_target = get_target(job["voice"], v)
            except Exception as e:
                log.warning("保真增强目标曲线计算失败，本次跳过: %s", e)
                _match_target = None
            if _match_target is None:
                log.warning("音色「%s」找不到可用的原声素材，本次不做保真增强",
                            job["voice"])

    # ---- 自动分块：整段文案按句子边界切成若干块，每块一个独立音频文件 ----
    # 员工不用手动分段（elevenlabs 时代是手动截成两半的），代码在这里代劳。
    # 块间不拼接，避免为了拼一个 769 秒的大文件白白多花时间。
    chunks = split_chunks(job["text"], max_chars=CHUNK_MAX_CHARS)
    n_chunk = max(len(chunks), 1)
    jid = job["id"]
    base_name = _safe_name((job.get("title") or "").strip()
                           or (job["text"] or "").strip()[:6])
    folder_name = _unique_dir(base_name)
    folder_path = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    job["chunks_total"] = n_chunk

    def _mk_progress(ci):
        def _progress(done, total):
            job["chunk"] = {"index": ci + 1, "total": n_chunk,
                            "seg_done": done, "seg_total": total}
            # 全局进度按块折算成千分位，避免每换一块进度条就归零
            frac = (ci + (done / total if total else 0)) / n_chunk
            job["progress"] = {"done": int(frac * 1000), "total": 1000}
            if total and done >= total and ci == n_chunk - 1:
                job["stage"] = "merging"
            elif total:
                job["stage"] = "generating"
        return _progress

    def _synth_cosy(text_ci):
        """CosyVoice3 分支：走独立服务的 HTTP 接口。

        只有它有 instruct（情感指令），Qwen3-TTS Base 克隆模式没有这个入参。
        """
        audio, sr, meta = cosy_client.synth(
            text=text_ci[1],
            ref_wav=ref_wav_path,
            ref_text=ref_text or "",
            instruct=_cosy_instruct,
            speed=float(job.get("speed") or 1.0),
            seed=int(text_ci[2]),
            # 进度回传：没有它，走 CosyVoice3 的任务前端会一直停在 0%
            on_progress=_mk_progress(text_ci[0]),
        )
        if audio is None:
            raise RuntimeError(f"CosyVoice3 生成失败: {meta.get('error')}")
        # 已按模型原生 speed 变速，后续不要再叠 atempo
        job["_cosy_speed_applied"] = True
        st = {
            "segments": meta.get("chunks") or 1,
            "chars": meta.get("chars") or len(text_ci[1]),
            "gen_seconds": meta.get("gen_seconds") or 0.0,
            "audio_seconds": meta.get("audio_seconds") or 0.0,
            "abnormal_segments": 0, "slow_fixed": 0, "voice_matched": 0,
            "segments_detail": [],
            "plan": {"engine": "cosyvoice3", "emotion": _emotion,
                     "instruct": _cosy_instruct},
        }
        return audio, sr, st

    def _synth(text_ci):
        """单次块生成，返回 (audio, sr, stats)。"""
        if _engine_name == "cosyvoice3" and ref_wav_path:
            return _synth_cosy(text_ci)
        return engine.synth_long(
            text=text_ci[1],
            mode=job["mode"],
            language=job["language"],
            speaker=job.get("speaker"),
            instruct=job.get("instruct") or "",
            ref_audio=ref_audio,
            ref_text=ref_text,
            ref_key=ref_key,
            max_chars=job.get("max_chars", 60),
            batch_size=job.get("batch_size", 16),
            seed=text_ci[2],
            temperature=job.get("temperature"),
            top_p=job.get("top_p"),
            repetition_penalty=job.get("repetition_penalty"),
            subtalker_temperature=job.get("subtalker_temperature"),
            voice_match=_match_target,
            progress=_mk_progress(text_ci[0]),
        )

    files = []
    agg = {"segments": 0, "chars": 0, "gen_seconds": 0.0, "audio_seconds": 0.0,
           "abnormal_segments": 0, "slow_fixed": 0, "voice_matched": 0,
           "asr_skipped": 0}
    detail_all = []
    chunk_info = []
    base_seed = int(job.get("seed") or 42)
    # 各块的 plan 内容一致，取最后一块的即可用于参数留痕
    last_plan = {}

    # ---- 显存串行：走 CosyVoice3 之前必须先把 Qwen 请出显存 ----
    # 两个引擎不能同时驻留（12G 卡放不下 Qwen ~8 GiB + CosyVoice3 3.2 GiB）。
    # 少了这一步，CosyVoice3 在加载阶段直接 CUDA OOM → HTTP 500 → 整个任务失败
    # （2026-09-29 实测失败两单：9d95858c / 2a8e4e31）。详见 app/vram.py。
    job["_used_cosy"] = False
    # 换引擎/加载模型这段时间前端一直显示「正在分段」，容易让人以为卡住了。
    # 标成 loading，等第一段产出后 _mk_progress 会自动改成 generating。
    job["stage"] = "loading"
    if _engine_name == "cosyvoice3" and ref_wav_path:
        vram.switch_to("cosyvoice3", f"任务 {jid} ")
        job["_used_cosy"] = True

    for ci, chunk_text in enumerate(chunks):
        best = None
        for attempt in range(ASR_MAX_RETRY + 1):
            seed = base_seed + attempt * 7919
            try:
                audio, sr, st = await loop.run_in_executor(
                    None, _synth, (ci, chunk_text, seed))
            except Exception as _e:
                # CosyVoice3 失败（绝大多数是显存不够导致加载 OOM）→ 整任务降级 Qwen。
                # 员工拿到手的永远是成品，不会因换引擎而失败；降级原因留在 job 上可查。
                if _engine_name != "cosyvoice3":
                    raise
                log.warning("任务 %s 第 %d/%d 块 CosyVoice3 失败（%s），"
                            "整任务降级为 Qwen 重出", jid, ci + 1, n_chunk,
                            repr(_e)[:120])
                _engine_name = "qwen"
                job["engine"] = "qwen"
                job["fallback_from"] = "cosyvoice3"
                job["fallback_reason"] = repr(_e)[:200]
                job.pop("_cosy_speed_applied", None)
                job["_used_cosy"] = False
                vram.switch_to("qwen", f"任务 {jid} ")
                audio, sr, st = await loop.run_in_executor(
                    None, _synth, (ci, chunk_text, seed))
            # 内容回检：语速体检抓不到「说着说着又笑又说奇怪的话」，
            # 只能用 ASR 转写比对原文。详见 app/asr_check.py。
            tmp_wav = os.path.join(folder_path, f".asr_{jid}_{ci}.wav")
            sf.write(tmp_wav, audio, sr)
            try:
                res = asr_check.check_chunk(tmp_wav, chunk_text,
                                            language=job["language"])
            finally:
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass
            cand = (audio, sr, st, res)
            if res.get("ok") or best is None:
                best = cand
            elif (res.get("cer") or 1.0) < (best[3].get("cer") or 1.0):
                best = cand
            log.info("任务 %s 第 %d/%d 块 第 %d 次: CER=%.4f %s",
                     jid, ci + 1, n_chunk, attempt + 1,
                     res.get("cer") or 0.0,
                     "通过" if res.get("ok") else "不通过->重掷")
            if res.get("ok"):
                break
        audio, sr, st, res = best
        # 质量闸门：CosyVoice3 带了情绪，但念错太多就必须退回 Qwen 重出一遍。
        # 员工拿到手的永远是「念得对」的成品，情绪不能以牺牲准确度为代价。
        if (_engine_name == "cosyvoice3"
                and (res.get("cer") or 0.0) > COSY_FALLBACK_CER):
            log.warning("任务 %s 第 %d 块 CosyVoice3 CER=%.4f 超过 %.2f，"
                        "自动回退 Qwen 重出", jid, ci + 1,
                        res.get("cer") or 0.0, COSY_FALLBACK_CER)
            _engine_name = "qwen"
            job["engine"] = "qwen"
            job["fallback_from"] = "cosyvoice3"
            job.pop("_cosy_speed_applied", None)
            # Qwen 此时已被卸掉（串行），要让位回来才能重出
            vram.switch_to("qwen", f"任务 {jid} ")
            audio, sr, st = await loop.run_in_executor(
                None, _synth, (ci, chunk_text, base_seed))
            tmp_wav = os.path.join(folder_path, f".asr_{jid}_{ci}.wav")
            sf.write(tmp_wav, audio, sr)
            try:
                res = asr_check.check_chunk(tmp_wav, chunk_text,
                                            language=job["language"])
            finally:
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass
            vram.switch_to("cosyvoice3", f"任务 {jid} ")
            _engine_name = "cosyvoice3"      # 下一块仍按原情感走
            job["engine"] = "cosyvoice3"
        st = st or {}
        if st.get("plan"):
            last_plan = st["plan"]
        agg["segments"] += st.get("segments") or 0
        agg["chars"] += st.get("chars") or 0
        agg["gen_seconds"] += st.get("gen_seconds") or 0.0
        agg["audio_seconds"] += st.get("audio_seconds") or 0.0
        agg["abnormal_segments"] += st.get("abnormal_segments") or 0
        agg["slow_fixed"] += st.get("slow_fixed") or 0
        agg["voice_matched"] += st.get("voice_matched") or 0
        if res.get("skipped"):
            agg["asr_skipped"] += 1
        detail_all.extend(st.get("segments_detail") or [])

        # 停顿预算对齐（PBA）：只切静音，不动任何有声样本，无音质损失。
        # 丢掉的静音时间在下面用 atempo 拉回，总时长与发音速率一起回到
        # 案例原声的水平。放在 ASR 回检之后——CER 要量的是模型的原始输出。
        _pba_ratio = 1.0
        if PBA_ALPHA < 0.999 and len(chunk_text) >= PBA_MIN_CHARS:
            audio, _pba_ratio = pba.align(audio, sr, PBA_ALPHA)
        # 停顿重排（在压缩之后）：压缩只管句数，这里把停顿长度重新分布，
        # 改掉的时长同样由下面的 atempo 拉回。
        if PBA_RESHAPE and len(chunk_text) >= PBA_MIN_CHARS:
            try:
                audio, _rs_ratio = pba.reshuffle(audio, sr, PBA_CV)
                _pba_ratio *= _rs_ratio
            except Exception as _e:
                log.warning("任务 %s 第 %d 块 停顿重排失败（跳过）: %r",
                            jid, ci + 1, _e)

        # 包络整形：补音节瞬态（修「舌头没展开」）。放在 PBA 之后——
        # PBA 靠 VAD 判静音，先整形会改变 VAD 结果。不动频谱、不动时长。
        if ENV_SHAPE and len(chunk_text) >= ENV_MIN_CHARS:
            try:
                audio, _es_sharp, _es_before, _es_after = envshape.fit(
                    audio, sr, ENV_TARGET, ENV_COMP)
                if _es_sharp:
                    log.info("任务 %s 第 %d 块 包络整形 sharp=%.3f "
                             "8~16Hz %.4f -> %.4f（目标 %.4f）",
                             jid, ci + 1, _es_sharp, _es_before,
                             _es_after, ENV_TARGET)
            except Exception as _e:
                log.warning("任务 %s 第 %d 块 包络整形失败（跳过）: %r",
                            jid, ci + 1, _e)

        # CosyVoice3 已在模型内部按 speed 变速（梅尔谱插值 + 声码器重合成），
        # 这里不能再叠一次 atempo，否则会变两次速。
        _speed = 1.0 if job.get("_cosy_speed_applied") else float(
            job.get("speed") or 1.0)
        # PBA 削掉的静音时间靠这一档 atempo 补回来：最终时长 = 原时长 / speed，
        # 与不开 PBA 时一致，员工侧的时长预估和统计都不受影响。
        _atempo = _speed * _pba_ratio
        # ---- 块尾分段停顿：修「听着没有分段」--------------------------------
        # split_chunks 只在句末/段末断开，但两块是**两个独立 mp3**，拼起来时
        # 接缝处只有 mp3 编码留下的约 0.1~0.2 s 数字零（实测块1 尾部从 -33dB
        # 直接掉到 -165dB），远短于段落级停顿 → 听感上"两段粘成一条"。
        # 这里给**非最后一块**补一段静音，补到 CHUNK_TAIL_PAUSE_MS 为止
        # （先量已有尾部静音，不够才补），员工把几个 mp3 顺序拼起来就是
        # 自然的换气/分段。
        if CHUNK_TAIL_PAUSE_MS > 0 and ci < len(chunks) - 1:
            try:
                _fr = max(int(sr * 0.02), 1)
                _n = len(audio) // _fr
                if _n >= 4:
                    _e = np.sqrt((audio[:_n * _fr].reshape(_n, _fr) ** 2)
                                 .mean(axis=1))
                    _db = 20 * np.log10(_e + 1e-9)
                    _tail = 0
                    for _v in _db[::-1]:
                        if _v < CHUNK_TAIL_SIL_DB:
                            _tail += 1
                        else:
                            break
                    _have = _tail * _fr / float(sr) * 1000.0
                    _need = CHUNK_TAIL_PAUSE_MS - _have
                    if _need > 20:
                        # atempo 会把这段一起拉长，先除回去
                        _pad = np.zeros(int(sr * _need / 1000.0 / max(
                            _atempo, 0.25)), dtype=audio.dtype)
                        audio = np.concatenate([audio, _pad])
                        log.info("任务 %s 第 %d 块 补分段停顿 %.0fms"
                                 "（原有尾部静音 %.0fms）", jid, ci + 1,
                                 _need, _have)
            except Exception as _e:
                log.warning("任务 %s 第 %d 块 补分段停顿失败（跳过）: %r",
                            jid, ci + 1, _e)
        # 分性别响度偏移：整段只判一次（同一个音色不会有男女混着的情况），
        # 判不出来就退回 0 dB。
        if "_gain_off" not in job:
            _off = 0.0
            if GAIN_GENDER:
                try:
                    _f0 = est_f0_hz(audio, sr)
                except Exception:
                    _f0 = None
                if _f0 is None:
                    log.info("任务 %s 性别判定失败（f0 测不出），响度偏移 0 dB",
                             jid)
                else:
                    _male = _f0 < F0_SPLIT_HZ
                    _off = GAIN_MALE_DB if _male else GAIN_FEMALE_DB
                    log.info("任务 %s 性别判定 f0=%.0fHz -> %s，响度偏移 %+.1f dB",
                             jid, _f0, "男" if _male else "女", _off)
            job["_gain_off"] = _off
        _gain_db = OUT_GAIN_DB + float(job.get("_gain_off") or 0.0)
        mp3_name = f"{base_name}-{ci + 1}.mp3"
        mp3_abs = os.path.join(folder_path, mp3_name)
        job["stage"] = "encoding"
        tmp_wav = os.path.join(folder_path, f".enc_{jid}_{ci}.wav")
        try:
            sf.write(tmp_wav, audio, sr)
            # 语速：模型无 speed 接口（见文件头 SPEED_MIN 注释），用 atempo 在
            # 未压缩的 wav 上做一次——只经过一次 mp3 编码，且 atempo 变速不变调。
            _cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_wav]
            _af = []
            if abs(_atempo - 1.0) > 1e-3:
                _af.append(f"atempo={_atempo:.4f}")
            # 音色对齐放在变速之后：atempo 只改时长不改频谱，顺序无副作用，
            # 但 EQ 一定要在 volume 之前，否则先衰减再提增益会白损失动态余量。
            if TONE_EQ and TONE_EQ_FILTER:
                _af.append(TONE_EQ_FILTER)
            # 高频空气感：先重采样到 44.1k（否则 12k 以上放不下），
            # 再生成谐波 + FIR 精确整形。
            if AIR:
                _af.append(f"aresample={OUT_SR}")
                if AIR_EXCITER:
                    _af.append(AIR_EXCITER)
                if AIR_EQ:
                    _af.append(AIR_EQ)
            if abs(_gain_db) > 1e-3:
                _af.append(f"volume={_gain_db:.2f}dB")
            if _af:
                _cmd += ["-filter:a", ",".join(_af)]
            _cmd += ["-codec:a", "libmp3lame", "-b:a", MP3_BITRATE,
                     "-ar", str(OUT_SR), mp3_abs]
            # 滤波链里多了 FIR 卷积，给足时间；超时不能让整单失败。
            subprocess.run(_cmd, check=True, timeout=600)
        except Exception as e:
            log.warning("mp3 转码失败 %s: %s", mp3_abs, e)
        finally:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass
        # 变速后实际时长 = 原时长 / speed，展示给员工和统计都要按这个值
        _dur = (st.get("audio_seconds") or 0.0) / (_speed or 1.0)
        rel = f"{folder_name}/{mp3_name}"
        files.append({"name": mp3_name, "path": rel,
                      "seconds": round(_dur, 2),
                      "chars": st.get("chars") or 0,
                      "cer": res.get("cer") if not res.get("skipped") else None})
        chunk_info.append({"index": ci + 1, "chars": len(chunk_text),
                           "seconds": round(_dur, 2),
                           "cer": res.get("cer") if not res.get("skipped") else None,
                           "asr_skipped": bool(res.get("skipped"))})
        job["files"] = files
        job["chunks"] = chunk_info
        persist_jobs()

    # 覆盖度自检：分段时空白/换行会被规整掉，正常落在 90%~100%。
    # 掉到 85% 以下说明有文本没被送进模型，必须在日志里暴露出来。
    _src_len = max(len(job.get("text") or ""), 1)
    # ASR 全部跳过 = 这一单**根本没做内容校验**（乱音/跑飞完全裸奔）。
    # 不能静默：2026-09-29 就是这么悄悄失效了很久，日志里只留一行 warning 被人忽略，
    # 而 jobs.json 里 cer 全是 null 又被误读成"CER=0，内容没问题"。
    if agg["asr_skipped"] >= n_chunk:
        log.warning("⚠️ 任务 %s: ASR 内容回检 %d/%d 块全部被跳过，成品未做内容校验。"
                    "请检查 faster-whisper 可用性（常见原因：模型缓存不在服务运行用户的 "
                    "HOME 下，或语言码不被接受，见 app/asr_check.py）",
                    jid, agg["asr_skipped"], n_chunk)

    _cov = agg["chars"] / _src_len
    if _cov < 0.85:
        log.warning("覆盖度异常 job=%s: 原文 %d 字，实际只生成 %d 字（%.1f%%），段数 %s",
                    jid, _src_len, agg["chars"], _cov * 100, agg["segments"])

    agg["rtf"] = round(agg["gen_seconds"] / agg["audio_seconds"], 4) \
        if agg["audio_seconds"] else 0.0
    agg["cps"] = round(agg["chars"] / agg["audio_seconds"], 2) \
        if agg["audio_seconds"] else 0.0
    agg["segments_detail"] = detail_all
    agg["plan"] = last_plan
    agg["chunks"] = n_chunk
    stats = agg
    # 兼容老前端/统计：mp3 指向第一个文件
    job["mp3"] = files[0]["path"] if files else None
    job["stats"] = stats
    job["enhanced"] = bool(agg["voice_matched"])
    # 自适应生效值回写：前端提交的 max_chars/batch_size 可能不安全，
    # 这里记录引擎实际采用的值，便于在历史任务里核对。
    _plan = (stats or {}).get("plan") or {}
    if _plan:
        job["max_chars"] = _plan.get("max_chars")
        job["batch_size"] = _plan.get("batch_size")
    job["status"] = "done"
    job["stage"] = "done"
    job["finished_at"] = time.time()
    # 用实测速度更新每段耗时经验值，供后续任务预估 ETA
    global SEG_SEC, AVG_RTF
    seg = stats.get("segments") or 0
    if seg > 0:
        per = stats.get("gen_seconds", 0) / seg
        SEG_SEC = round(0.7 * SEG_SEC + 0.3 * per, 2)
    rtf = stats.get("rtf")
    if rtf and 0.05 < rtf < 10:
        AVG_RTF = round(0.6 * AVG_RTF + 0.4 * rtf, 3)
    persist_jobs()
    log.info("任务 %s 完成: 用户=%s 分段=%s 音频=%.1fs 耗时=%.1fs RTF=%.3f",
             jid, job.get("owner"), stats.get("segments"), stats.get("audio_seconds", 0),
             stats.get("gen_seconds", 0), stats.get("rtf", 0))
    if _plan:
        log.info("任务 %s 参数: max_chars=%s batch=%s 批次=%s 单批tokens=%s "
                 "峰值显存=%sGiB | 前端提交 %s/%s | 净余量 %sGiB",
                 jid, _plan.get("max_chars"), _plan.get("batch_size"),
                 _plan.get("batches"), _plan.get("batch_tokens"),
                 _plan.get("peak_alloc_gib"), _plan.get("requested_max_chars"),
                 _plan.get("requested_batch_size"), _plan.get("headroom_gib"))


async def worker():
    while True:
        job = await job_queue.get()
        try:
            await run_job(job)
        except Exception as e:
            log.exception("任务失败")
            job["status"] = "error"
            job["error"] = str(e)
            job["finished_at"] = time.time()
            persist_jobs()
        finally:
            # 双引擎显存串行收尾：本任务用过 CosyVoice3 就把 Qwen 恢复常驻，
            # 否则下个 Qwen 任务要等懒加载，Cosy 也会白占 3.2 GiB 直到空闲超时。
            if job.get("_used_cosy"):
                try:
                    _loop = asyncio.get_running_loop()
                    await _loop.run_in_executor(None, vram.restore_qwen)
                except Exception as _e:
                    log.warning("收尾恢复 Qwen 失败: %s", repr(_e)[:160])
            job_queue.task_done()


@app.on_event("startup")
async def startup():
    asyncio.create_task(worker())
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, engine.warmup)
        log.info("模型预热完成: %s", engine.variant)
    except Exception as e:
        log.error("模型预热失败: %s", e)
    # 预热后再入队，避免和模型加载抢显存；按提交时间排序，先来的先跑
    if PENDING_RESUME:
        PENDING_RESUME.sort(key=lambda x: x.get("created_at") or 0)
        for j in PENDING_RESUME:
            await job_queue.put(j)
        log.info("重启续跑：已重新入队 %d 个被中断的任务 %s",
                 len(PENDING_RESUME), [j["id"] for j in PENDING_RESUME])
        persist_jobs()


# ============================ 认证 API ============================

@app.post("/api/login")
async def login(username: str = Form(...), password: str = Form(...)):
    u = users.get(username.strip())
    if not u or not verify_pw(password, u["pwd"]):
        raise HTTPException(401, "账号或密码错误")
    u["last_login"] = time.time()
    save_json(USERS_FILE, users)
    tok = new_session(username)
    resp = JSONResponse({
        "username": username,
        "role": u.get("role", "user"),
        "must_change": bool(u.get("must_change")),
    })
    resp.set_cookie("tts_session", tok, max_age=SESSION_TTL, httponly=True,
                    samesite="lax", path="/")
    return resp


@app.post("/api/logout")
async def logout(request: Request):
    tok = request.cookies.get("tts_session")
    if tok:
        SESSIONS.pop(tok, None)
    r = JSONResponse({"ok": True})
    r.delete_cookie("tts_session", path="/")
    return r


@app.get("/api/me")
async def me(request: Request):
    u = current_user(request)
    if not u:
        raise HTTPException(401, "未登录")
    rec = users.get(u["username"], {})
    return {
        "username": u["username"],
        "role": u["role"],
        "must_change": bool(rec.get("must_change")),
        "created_at": rec.get("created_at"),
    }


@app.post("/api/password")
async def change_password(request: Request,
                          old_password: str = Form(...),
                          new_password: str = Form(...)):
    u = require(request)
    rec = users[u["username"]]
    if not verify_pw(old_password, rec["pwd"]):
        raise HTTPException(400, "旧密码不正确")
    if len(new_password) < MIN_PW_LEN:
        raise HTTPException(400, f"新密码至少 {MIN_PW_LEN} 位")
    rec["pwd"] = hash_pw(new_password)
    rec["must_change"] = False
    save_json(USERS_FILE, users)
    # 改密后踢掉该用户的所有会话，强制重新登录
    for tok, s in list(SESSIONS.items()):
        if s["user"] == u["username"]:
            SESSIONS.pop(tok, None)
    return {"ok": True}


@app.get("/api/users")
async def list_users(request: Request):
    """管理员看全员产出：累计条数 + 今日条数。"""
    require_admin(request)
    today = time.strftime("%Y-%m-%d")
    cnt, today_cnt = {}, {}
    for j in jobs.values():
        if j.get("status") != "done":
            continue
        o = j.get("owner", "?")
        cnt[o] = cnt.get(o, 0) + 1
        d = time.strftime("%Y-%m-%d", time.localtime(j.get("created_at", 0)))
        if d == today:
            today_cnt[o] = today_cnt.get(o, 0) + 1
    return [{
        "username": n,
        "role": r.get("role", "user"),
        "created_at": r.get("created_at"),
        "last_login": r.get("last_login"),
        "jobs": cnt.get(n, 0),
        "today": today_cnt.get(n, 0),
    } for n, r in users.items()]


@app.post("/api/users")
async def create_user(request: Request,
                      username: str = Form(...),
                      password: str = Form(...),
                      role: str = Form("user")):
    require_admin(request)
    name = username.strip()
    if not name or len(name) > 32:
        raise HTTPException(400, "用户名无效")
    if name in users:
        raise HTTPException(400, "账号已存在")
    if role not in ("admin", "user"):
        raise HTTPException(400, "角色无效")
    if len(password) < MIN_PW_LEN:
        raise HTTPException(400, f"密码至少 {MIN_PW_LEN} 位")
    users[name] = {"role": role, "pwd": hash_pw(password),
                   "created_at": time.time(), "last_login": None, "must_change": False}
    save_json(USERS_FILE, users)
    log.info("管理员创建新账号 %s (%s)", name, role)
    return {"ok": True, "username": name}


@app.post("/api/users/{name}/password")
async def reset_password(request: Request, name: str, new_password: str = Form(...)):
    require_admin(request)
    if name not in users:
        raise HTTPException(404, "账号不存在")
    if len(new_password) < MIN_PW_LEN:
        raise HTTPException(400, f"密码至少 {MIN_PW_LEN} 位")
    users[name]["pwd"] = hash_pw(new_password)
    users[name]["must_change"] = True
    save_json(USERS_FILE, users)
    for tok, s in list(SESSIONS.items()):
        if s["user"] == name:
            SESSIONS.pop(tok, None)
    return {"ok": True}


@app.delete("/api/users/{name}")
async def delete_user(request: Request, name: str):
    u = require_admin(request)
    if name == u["username"]:
        raise HTTPException(400, "不能删除自己")
    if name not in users:
        raise HTTPException(404, "账号不存在")
    del users[name]
    save_json(USERS_FILE, users)
    for tok, s in list(SESSIONS.items()):
        if s["user"] == name:
            SESSIONS.pop(tok, None)
    return {"ok": True}


# ============================ 页面与业务 API ============================

@app.get("/", response_class=HTMLResponse)
async def index():
    p = os.path.join(os.path.dirname(__file__), "ui", "index.html")
    with open(p, "r", encoding="utf-8") as f:
        # 禁用缓存：改前端后刷新就能生效，不用清浏览器缓存
        return HTMLResponse(f.read(), headers={
            "Cache-Control": "no-store, must-revalidate",
            "Pragma": "no-cache",
        })


# 内置音色 → 所属语言。用于界面按"输出语言"筛选音色，
# 避免出现"韩语音色念中文"这类别扭组合。
SPEAKER_LANG = {
    "aiden": "english",
    "dylan": "chinese",
    "eric": "chinese",
    "ono_anna": "japanese",
    "ryan": "english",
    "serena": "chinese",
    "sohee": "korean",
    "uncle_fu": "chinese",
    "vivian": "chinese",
}
# 每种语言的推荐音色（切语言时自动选中）。
# 注：9 个音色实测都支持全部 10 种语言，这里只是"哪种音色念这个语言最自然"的推荐，
# 不是限制——界面会完整展示全部音色供自由选择。
SPEAKER_DEFAULT = {
    "auto": "sohee",
    "chinese": "serena",
    "english": "aiden",
    "japanese": "ono_anna",
    "korean": "sohee",
    "french": "serena",
    "german": "uncle_fu",
    "italian": "vivian",
    "portuguese": "ryan",
    "russian": "dylan",
    "spanish": "aiden",
}


@app.get("/api/meta")
async def meta(request: Request):
    require(request)
    spk = engine.speakers if engine.model else []
    # 兜底：official 9 个发音人来自 CustomVoice 模型的 talker_config.spk_id，
    # 若因为模型未加载 / 该字段读取为空而拿不到，界面上的「预设音色」会整个
    # 消失（实测出现过 speakers=[] 的情况），员工会以为音色被删了。
    # 这里用元数据里写死的 SPEAKER_LANG 键兜住，保证 9 个始终可用。
    if not spk:
        spk = list(SPEAKER_LANG.keys())
        log.warning("模型未返回发音人列表，已用内置兜底列表（%d 个）", len(spk))
    return {
        "languages": engine.languages if engine.model else [],
        "speakers": spk,
        # 音色的母语（仅用于排序与推荐，不做过滤）
        "speaker_langs": {s: SPEAKER_LANG.get(s, "") for s in spk},
        "speaker_default": SPEAKER_DEFAULT,
        # 各音色支持的语言：实测全部音色均支持全部语言
        "all_speakers_all_langs": True,
        "voices": list(voices.keys()),
        # 音色是否可用于克隆：ICL 模式必须有文字稿，且参考音频不能超长
        "voices_ok": {
            k: bool((v.get("ref_text") or "").strip())
            and float(v.get("dur") or 0) <= REF_AUDIO_MAX_SEC
            for k, v in voices.items()},
        "variants": list(MODEL_DIRS.keys()),
        "loaded": engine.variant,
        "voice_match_default": VOICE_MATCH_DEFAULT,
        # 语速档位：模型无 speed 接口，靠 ffmpeg atempo 后处理（变速不变调）
        "speeds": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        # 2026-09-30 全量实测（男 1999 字 / 女 1773 字，与案例原声逐项对比）：
        # 默认 1.0 生成的语速比原声快 11.5~11.9%，降到 0.9 后偏差缩到 +0.4% / +0.7%。
        "speed_default": 0.9,
        "speed_min": SPEED_MIN, "speed_max": SPEED_MAX,
        # 员工只需要选「情感」；引擎/指令/参数由后端自动分流
        "emotions": [
            {"key": "standard", "engine": "qwen"},
            {"key": "excited", "engine": "cosyvoice3"},
            {"key": "warm", "engine": "cosyvoice3"},
            {"key": "serious", "engine": "cosyvoice3"},
        ],
        "emotion_default": DEFAULT_EMOTION,
    }


@app.post("/api/jobs")
async def create_job(request: Request,
                     title: str = Form(""),
                     text: str = Form(...),
                     mode: str = Form("custom"),
                     language: str = Form("Korean"),
                     speaker: str = Form(""),
                     instruct: str = Form(""),
                     voice: str = Form(""),
                     max_chars: int = Form(60),
                     batch_size: int = Form(16),
                     seed: int = Form(42),
                     speed: float = Form(1.0),
                     emotion: str = Form(DEFAULT_EMOTION),
                     temperature: Optional[float] = Form(None),
                     subtalker_temperature: Optional[float] = Form(None),
                     enhance: Optional[bool] = Form(None)):
    u = require(request)
    if temperature is not None and not (0.3 <= temperature <= 1.5):
        raise HTTPException(400, "temperature 需在 0.3~1.5 之间")
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        speed = 1.0
    if not (SPEED_MIN <= speed <= SPEED_MAX):
        raise HTTPException(400, f"语速需在 {SPEED_MIN}~{SPEED_MAX} 之间")
    speed = round(speed, 2)
    emotion = (emotion or DEFAULT_EMOTION).strip()
    if emotion not in EMOTION_PRESETS:
        raise HTTPException(400, f"情感需在 {list(EMOTION_PRESETS)} 之内")
    if not text.strip():
        raise HTTPException(400, "文本为空")
    if mode == "clone":
        if not voice:
            raise HTTPException(400, "克隆模式需要选择已录入的音色")
        _v = voices.get(voice)
        if not _v:
            raise HTTPException(400, f"音色不存在：{voice}")
        if not (_v.get("ref_text") or "").strip():
            raise HTTPException(
                400, f"音色「{voice}」缺少参考文字稿，无法克隆。"
                     f"请重新录入该音色，并填写参考音频对应的逐字文字")
        if float(_v.get("dur") or 0) > REF_AUDIO_MAX_SEC:
            raise HTTPException(
                400, f"音色「{voice}」的参考音频 {_v.get('dur')}s 是未裁剪的旧音色"
                     f"（超过 {REF_AUDIO_MAX_SEC:.0f}s，显存会暴涨、生成明显变慢）。"
                     f"请用原音频重新录入一次，系统会自动截取最佳片段")

    title = (title or "").strip() or text.strip().replace("\n", " ")[:24]
    jid = uuid.uuid4().hex[:12]
    job = {
        "id": jid,
        "title": title[:80],
        "owner": u["username"],
        "text": text,
        "mode": mode,
        "language": language,
        "speaker": speaker or None,
        "instruct": instruct,
        "voice": voice,
        "max_chars": max_chars,
        "batch_size": batch_size,
        "seed": seed,
        "speed": speed,
        "emotion": emotion,     # 员工只选情感，引擎由 EMOTION_PRESETS 自动分流
        "engine": EMOTION_PRESETS[emotion]["engine"],
        "temperature": temperature,
        # 韵律细节采样温度（模型第二路采样）。默认不传=沿用官方 0.9；
        # 只在实测确认不放大跑飞后才放开，见 tts_engine.GEN_SUBTALKER_TEMPERATURE。
        "subtalker_temperature": subtalker_temperature,
        # 不传则跟随服务端默认开关；传 false 可产出老版本输出做 A/B 对照
        "enhance": VOICE_MATCH_DEFAULT if enhance is None else bool(enhance),
        "enhanced": False,
        "status": "queued",
        "started_at": None,
        "progress": {"done": 0, "total": 0},
        "created_at": time.time(),
    }
    jobs[jid] = job
    await job_queue.put(job)
    persist_jobs()
    return {"id": jid}


@app.get("/api/jobs")
async def list_jobs(request: Request, scope: str = "mine", q: str = "", owner: str = "",
                    status: str = "", page: int = 1, page_size: int = 10):
    """scope: mine=只看自己（默认）；all=看全部（仅管理员）。
    q=标题/正文关键词，owner=按作者过滤，status=按状态过滤。
    page/page_size=分页，返回 {items, total, page, pages, page_size}。"""
    u = require(request)
    now = time.time()
    items = sorted(jobs.values(), key=lambda j: j.get("created_at", 0), reverse=True)
    if scope == "all" and u["role"] == "admin":
        if owner:
            items = [j for j in items if j.get("owner") == owner]
    else:
        items = [j for j in items if j.get("owner") == u["username"]]

    q = (q or "").strip().lower()
    if q:
        items = [j for j in items
                 if q in (j.get("title") or "").lower()
                 or q in (j.get("text") or "").lower()]
    if status:
        items = [j for j in items if j["status"] == status]

    # 分页：先算总量再切片，避免一次性把上万条推给前端。
    # 变量名不能用 total —— 下面的循环里有 done/total 解包会把它覆盖掉。
    total_cnt = len(items)
    page_size = max(1, min(int(page_size or 10), 1000))
    pages = max(1, (total_cnt + page_size - 1) // page_size)
    page = max(1, min(int(page or 1), pages))
    page_items = items[(page - 1) * page_size: page * page_size]

    running_any = any(j["status"] == "running" for j in jobs.values())
    out = []
    for j in page_items:
        p = j.get("progress") or {}
        done, total = p.get("done", 0), p.get("total", 0)
        eta = None
        ahead = 0
        if j["status"] == "running":
            st = j.get("started_at") or j["created_at"]
            elapsed = max(now - st, 0.1)
            if total and 0 < done < total:
                eta = round(elapsed / done * (total - done), 1)
            else:
                # 还没出第一批结果（含 CosyVoice3 路径还没回报进度的情形）：
                # 用历史速度粗估 —— RTF + 文本长度，比"每段固定秒数"准得多。
                # 少了这个兜底，走 CosyVoice3 的任务会一直显示 0%（2026-09-29 实测）。
                txt = j.get("text") or ""
                est_audio = (len(txt) / CHARS_PER_SEC) if txt else (total * 8.0)
                eta = round(max(est_audio * AVG_RTF - elapsed, 1.0), 1) if est_audio else 0
        elif j["status"] == "queued":
            ahead = sum(1 for x in jobs.values()
                        if x["status"] == "queued" and x["created_at"] < j["created_at"])
            if running_any:
                ahead += 1
        out.append({
            "id": j["id"],
            "title": j.get("title") or "",
            "owner": j.get("owner"),
            "status": j["status"],
            "mode": j["mode"],
            "language": j["language"],
            "speaker": j.get("speaker"),
            "voice": j.get("voice"),
            "created_at": j["created_at"],
            "started_at": j.get("started_at"),
            "progress": p,
            "stage": j.get("stage"),
            "batch_size": j.get("batch_size", 16),
            "eta_seconds": eta,
            "elapsed_seconds": (round(now - (j.get("started_at") or j.get("created_at", now)), 1)
                                if j["status"] == "running" else None),
            "queue_ahead": ahead,
            # 服务重启后自动续跑的次数，>0 说明这条任务曾被中断过又接着跑
            "resumed": j.get("resumed", 0),
            "stats": j.get("stats"),
            "wav": j.get("wav"),
            "mp3": j.get("mp3"),
            # 分块输出：同一任务可能产出 2~3 个独立 mp3，都在同一个子目录里
            "files": j.get("files") or [],
            "chunks": j.get("chunks") or [],
            "error": j.get("error"),
            "preview": (j.get("text") or "")[:60],
        })
    return {"items": out, "total": total_cnt, "page": page,
            "pages": pages, "page_size": page_size}


@app.delete("/api/jobs/{jid}")
async def delete_job(request: Request, jid: str):
    u = require(request)
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "任务不存在")
    if u["role"] != "admin" and j.get("owner") != u["username"]:
        raise HTTPException(403, "无权删除")
    names = [f.get("path") for f in (j.get("files") or [])] \
        or [f for f in (j.get("wav"), j.get("mp3")) if f]
    for f in names:
        if not f:
            continue
        p = os.path.realpath(os.path.join(OUTPUT_DIR, f))
        root = os.path.realpath(OUTPUT_DIR)
        if not (p == root or p.startswith(root + os.sep)):
            continue
        try:
            os.remove(p)
        except OSError:
            pass
    # 分块输出的任务独占一个子目录，文件删完就把空目录也收掉
    if j.get("files"):
        sub = os.path.dirname(os.path.join(OUTPUT_DIR, names[0])) if names else None
        if sub and os.path.isdir(sub):
            try:
                os.rmdir(sub)
            except OSError:
                pass
    jobs.pop(jid, None)
    persist_jobs()
    return {"ok": True}


@app.post("/api/jobs/{jid}/retry")
async def retry_job(request: Request, jid: str):
    u = require(request)
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "任务不存在")
    if u["role"] != "admin" and j.get("owner") != u["username"]:
        raise HTTPException(403, "无权操作")
    if not j.get("text"):
        raise HTTPException(400, "原文案已丢失，无法重试，请重新提交")
    new = dict(j)
    new["id"] = uuid.uuid4().hex[:12]
    new["status"] = "queued"
    new["progress"] = {"done": 0, "total": 0}
    new["created_at"] = time.time()
    new.pop("wav", None)
    new.pop("mp3", None)
    new.pop("error", None)
    new.pop("stats", None)
    new["title"] = (j.get("title") or "")[:80]
    jobs[new["id"]] = new
    await job_queue.put(new)
    persist_jobs()
    return {"id": new["id"]}


@app.get("/api/stats")
async def stats(request: Request, days: int = 7):
    """按人统计产出。管理员看全部，员工只看自己。含今日条数与最近 N 天明细。"""
    u = require(request)
    today = time.strftime("%Y-%m-%d")
    rows = {}
    for j in jobs.values():
        if j.get("status") != "done":
            continue
        o = j.get("owner", "?")
        if u["role"] != "admin" and o != u["username"]:
            continue
        r = rows.setdefault(o, {"jobs": 0, "today": 0, "audio_seconds": 0.0,
                                "gen_seconds": 0.0, "by_day": {}})
        r["jobs"] += 1
        s = j.get("stats") or {}
        r["audio_seconds"] += float(s.get("audio_seconds", 0))
        r["gen_seconds"] += float(s.get("gen_seconds", 0))
        d = time.strftime("%Y-%m-%d", time.localtime(j.get("created_at", 0)))
        r["by_day"][d] = r["by_day"].get(d, 0) + 1
        if d == today:
            r["today"] += 1
    # by_day 按真实日期窗口过滤（days=0 表示不限）；
    # jobs / audio_seconds / gen_seconds 始终是全部累计，不随窗口变化。
    cutoff_date = ""
    if days and days > 0:
        cutoff_date = time.strftime("%Y-%m-%d", time.localtime(time.time() - days * 86400))
    for r in rows.values():
        bd = r["by_day"]
        if cutoff_date:
            bd = {k: v for k, v in bd.items() if k >= cutoff_date}
        r["by_day"] = dict(sorted(bd.items(), reverse=True))
        r["audio_seconds"] = round(r["audio_seconds"], 1)
        r["gen_seconds"] = round(r["gen_seconds"], 1)
    return rows


@app.get("/api/stats/export")
async def stats_export(request: Request, days: int = 0):
    """导出任务明细 CSV（可用 Excel 直接打开）。

    管理员导出全部，员工只能导出自己的。days=0 表示不限时间。
    """
    import urllib.parse
    u = require(request)
    cutoff = None
    if days and days > 0:
        cutoff = time.time() - days * 86400

    rows = []
    for j in jobs.values():
        if u["role"] != "admin" and j.get("owner") != u["username"]:
            continue
        ct = j.get("created_at", 0)
        if cutoff and ct < cutoff:
            continue
        s = j.get("stats") or {}
        rows.append({
            "任务ID": j.get("id", ""),
            "标题": j.get("title") or "",
            "生成人": j.get("owner") or "",
            "状态": j.get("status") or "",
            "模式": j.get("mode") or "",
            "语言": j.get("language") or "",
            "音色": j.get("speaker") or j.get("voice") or "",
            "创建时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ct)) if ct else "",
            "完成时间": (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(j["finished_at"]))
                        if j.get("finished_at") else ""),
            "音频时长(秒)": s.get("audio_seconds", ""),
            "生成耗时(秒)": s.get("gen_seconds", ""),
            "RTF": s.get("rtf", ""),
            "分段数": s.get("segments", ""),
            "字数": len(j.get("text") or ""),
            "错误信息": j.get("error") or "",
        })
    rows.sort(key=lambda r: r["创建时间"], reverse=True)

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else
                       ["任务ID", "标题", "生成人", "状态", "创建时间"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
    # BOM 让 Excel 正确识别 UTF-8
    data = "\ufeff" + buf.getvalue()
    fname = "tts-report-%s.csv" % time.strftime("%Y%m%d-%H%M%S")
    return Response(
        content=data.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition":
                 "attachment; filename=\"%s\"; filename*=UTF-8''%s"
                 % (fname, urllib.parse.quote(fname))},
    )


@app.get("/api/jobs/{jid}")
async def get_job(request: Request, jid: str):
    u = require(request)
    if jid not in jobs:
        raise HTTPException(404, "任务不存在")
    j = jobs[jid]
    if u["role"] != "admin" and j.get("owner") != u["username"]:
        raise HTTPException(403, "无权查看")
    return j


@app.post("/api/voices")
async def upload_voice(request: Request,
                       name: str = Form(...),
                       ref_text: str = Form(""),
                       engine: str = Form(DEFAULT_ENGINE),
                       file: UploadFile = File(...)):
    require(request)
    # 引擎由录入者（管理员）手动指定，写进 voices.json，之后这个音色的生成都走它。
    engine = (engine or DEFAULT_ENGINE).strip()
    if engine not in VALID_ENGINES:
        raise HTTPException(400, f"引擎需在 {list(VALID_ENGINES)} 之内")
    safe = "".join(c for c in name if c not in '/\\:*?"<>|').strip()
    if not safe:
        raise HTTPException(400, "音色名无效")
    # 克隆是 ICL 模式：模型内部对空文字稿直接抛 ValueError，必须拦在入口处，
    # 否则员工只会看到一句英文报错，且任务已经占掉队列才失败。
    ref_text = (ref_text or "").strip()
    if not ref_text:
        raise HTTPException(
            400, "参考音频对应的文字不能为空：克隆模式必须填写这段音频的完整逐字稿，"
                 "文字与音频不一致会直接拉低克隆相似度")
    raw = await file.read()
    tmp = os.path.join(VOICE_DIR, f"_tmp_{uuid.uuid4().hex[:8]}")
    with open(tmp, "wb") as f:
        f.write(raw)
    try:
        data, sr = sf.read(tmp, dtype="float32")
    except Exception as e:
        os.remove(tmp)
        raise HTTPException(400, f"无法读取音频: {e}")
    if data.ndim > 1:
        data = data.mean(axis=1)
    dur = len(data) / sr
    if dur < REF_AUDIO_MIN_SEC:
        os.remove(tmp)
        raise HTTPException(
            400, f"参考音频太短: {dur:.1f}s，至少需要 {REF_AUDIO_MIN_SEC:.0f} 秒")
    if dur > REF_AUDIO_HARD_MAX_SEC:
        os.remove(tmp)
        raise HTTPException(
            400, f"参考音频太长: {dur / 60:.1f} 分钟，上限 "
                 f"{REF_AUDIO_HARD_MAX_SEC / 60:.0f} 分钟。"
                 f"请先粗略裁剪到几分钟内再上传")

    # 原始音频留档：便于日后换策略重新分析，不必让员工再传一次
    src_dir = os.path.join(VOICE_DIR, "sources")
    src_path = ""
    try:
        os.makedirs(src_dir, exist_ok=True)
        src_path = os.path.join(src_dir, f"{safe}__full.wav")
        sf.write(src_path, data, sr)
    except Exception as e:
        log.warning("原始音频留档失败（不影响录入）: %s", e)
        src_path = ""

    # 自动截取：员工不需要懂剪辑。超过 30 秒的音频由代码挑最清晰的一段，
    # 文字稿按累计有声时长对齐截出对应片段。
    t0 = time.time()
    audio, sr2, used_text, info = trim_reference(
        data, sr, ref_text,
        target=REF_AUDIO_TRIM_TARGET, trim_above=REF_AUDIO_AUTO_TRIM_ABOVE)
    info["analyze_seconds"] = round(time.time() - t0, 2)
    out_dur = len(audio) / float(sr2)

    path = os.path.join(VOICE_DIR, f"{safe}.wav")
    sf.write(path, audio, sr2)
    os.remove(tmp)

    if not used_text.strip():
        # ref_text 为空会让模型提前收尾（表现为「只念了一半」），这里直接拦住，
        # 不让一个坏音色混进来。
        raise HTTPException(
            400, "自动截取后文字为空：文字稿与这段音频对不上（或文字稿为空）。"
                 "请确认粘贴的是这段录音的完整文字稿后重新录入")
    warn = ""
    if out_dur > 0:
        # 语速阈值收紧到 4.5~12 字/秒：文字明显多于/少于音频内容时，模型会从
        # prompt 里学到一个错的语速——实测 130 字配 10.5 秒（12.4 字/秒）时，
        # 生成语速崩到 2.44 字/秒（正常 6.15），耗时涨 5.6 倍、音频长 2.5 倍。
        # 下限由 2.5 抬到 4.5：文字远少于音频时长同样会教出一个错的慢语速，
        # 而生成端的体检只能事后补救，不如在录入时就拦住。
        rate = len(used_text) / out_dur
        if rate < 4.5 or rate > 12:
            warn = (f"文字稿与音频时长对不上（约 {rate:.1f} 字/秒，正常语速约 5~7），"
                    f"克隆出来的语速可能偏快或偏慢。建议核对文字稿后重新录入")

    voices[safe] = {
        "path": path,
        "ref_text": used_text,
        "sr": sr2,
        "dur": round(out_dur, 2),
        # 引擎：管理员录入时手动指定（qwen=最准平稳 / cosyvoice3=有情感）
        "engine": engine,
        # 留痕：出问题时能看出代码到底截了哪一段、文字对不对
        "src_path": src_path,
        "src_dur": info.get("src_dur"),
        "src_text_len": len(ref_text),
        # 完整原文稿：出问题时能直接比对「员工到底粘贴了什么」，
        # 不用再去翻 sources/ 里的音频猜。只存文本，体积可忽略。
        "src_ref_text": ref_text,
        "clip": None if not info.get("trimmed") else {
            "start": info.get("clip_start"),
            "end": info.get("clip_end"),
            "method": info.get("method"),
            "text_method": info.get("text_method"),
        },
    }
    save_voices(voices)
    log.info("录入音色 %s: 原 %.1fs -> 成品 %.2fs, ref_text %d 字, 引擎=%s, 分析 %.2fs, %s",
             safe, dur, out_dur, len(used_text), engine, info["analyze_seconds"], info)

    # ⚠️ 自动选型已取消（2026-09-29）：打分公式 score = -10×CER + 句内动态 让
    # CER 极低的 Qwen 恒赢，两个内置音色都被判成 qwen/standard，测了等于没测。
    # 改为录入时手动指定引擎（engine 字段）。
    # 手动端点 POST /api/voices/{name}/auto-select 保留，需要单独标定时仍可调用。
    auto_info = None

    return {
        "name": safe,
        "dur": round(out_dur, 2),
        "sr": sr2,
        "trimmed": bool(info.get("trimmed")),
        "src_dur": info.get("src_dur"),
        "clip_start": info.get("clip_start"),
        "clip_end": info.get("clip_end"),
        "ref_text": used_text,
        "warn": warn,
        "auto": auto_info,
    }


@app.post("/api/voices/{name}/engine")
async def set_voice_engine(request: Request, name: str, engine: str = Form(...)):
    """改某个音色默认用哪个引擎。**立即生效，不用重启服务**。

    2026-09-29 取消自动选型后，引擎完全由人工指定：录入时选一次，
    之后在这里随时改。voices 是内存里的 dict，写回 voices.json 后
    下一个任务就直接按新引擎走。

    注意：这只是「默认引擎」。员工生成时选了非平稳情感，**仍然强制走
    CosyVoice3** —— Qwen 克隆没有 instruct，给不了情绪。
    """
    require(request)
    v = voices.get(name)
    if not v:
        raise HTTPException(404, "音色不存在")
    engine = (engine or "").strip()
    if engine not in VALID_ENGINES:
        raise HTTPException(400, f"引擎需在 {list(VALID_ENGINES)} 之内")
    old = (v.get("engine") or "").strip() or "未指定"
    v["engine"] = engine
    save_voices(voices)
    log.info("音色 %s 引擎: %s -> %s", name, old, engine)
    return {"ok": True, "name": name, "engine": engine, "old": old}


@app.post("/api/voices/{name}/auto-select")
async def rerun_voice_auto_select(request: Request, name: str):
    """手动重跑某个音色的「引擎 + 情感」自动选型。

    用途：
      - 老音色（录入早于该功能）补跑一次，让它们也享受自适应路由
      - 上传时队列忙被跳过，事后补测
      - 换了模型/参数后重新标定
    """
    require(request)
    v = voices.get(name)
    if not v:
        raise HTTPException(404, "音色不存在")
    # 和录入路径同一道守卫：队列里有任务在跑时不能跑选型 ——
    # 选型要独占整卡（先卸 Qwen 跑 CosyVoice3，再卸 Cosy 恢复 Qwen），
    # 与正在生成的任务并行必 OOM。
    _busy = any(j.get("status") in ("running", "queued") for j in jobs.values())
    if _busy:
        raise HTTPException(409, "队列里有正在生成/排队的任务，选型需要独占显卡，请等队列空了再试")
    try:
        loop = asyncio.get_running_loop()
        t0 = time.time()
        auto = await loop.run_in_executor(
            None, engine_select.select_for_voice, name, v)
    except Exception as e:
        raise HTTPException(500, f"选型失败: {repr(e)[:200]}")
    if not auto:
        raise HTTPException(500, "选型无可用结果（参考音频缺失或全部候选失败）")
    auto["elapsed_total"] = round(time.time() - t0, 1)
    return {"ok": True, "name": name, "auto": auto}


@app.get("/api/voices")
async def list_voices(request: Request):
    require(request)
    return {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in voices.items()}


@app.get("/api/voices/{name}/audio")
async def voice_audio(request: Request, name: str):
    """试听已录入音色的参考音频（成品片段，不是原始长音频）。"""
    require(request)
    v = voices.get(name)
    if not v or not os.path.isfile(v.get("path", "")):
        raise HTTPException(404, "音色不存在")
    return FileResponse(v["path"], media_type="audio/wav",
                        filename=f"{name}.wav")


@app.delete("/api/voices/{name}")
async def delete_voice(request: Request, name: str):
    """删除音色：成品片段 + sources/ 里的原音频一起清掉。

    原音频动辄几十 MB，只删成品会留下孤儿文件（现网就堆积过），
    而且音色名可以被复用，残留的原音频还会被下一次同名录入覆盖。
    """
    require(request)
    if name not in voices:
        raise HTTPException(404, "音色不存在")
    v = voices[name]
    if v.get("builtin"):
        raise HTTPException(400, "内置音色不可删除")
    # 内置音色是团队共用模板，误删会让所有人立刻无法生成，后端再拦一道
    # （前端已禁用按钮，但接口可被直接调用）
    if v.get("builtin"):
        raise HTTPException(400, "内置音色不可删除")

    # 原音频路径：优先用元数据里的，没有就按约定路径兜底（早期录入可能没写 src_path）
    src = v.get("src_path") or os.path.join(VOICE_DIR, "sources", f"{name}__full.wav")
    removed = []
    for p in (v.get("path") or "", src):
        # 白名单校验：只删 VOICE_DIR 内的文件，防路径穿越误删
        if not p or not os.path.isfile(p):
            continue
        if os.path.realpath(p).startswith(os.path.realpath(VOICE_DIR) + os.sep):
            try:
                os.remove(p)
                removed.append(os.path.basename(p))
            except OSError as e:
                log.warning("删除音色 %s 的文件失败 %s: %s", name, p, e)
        else:
            log.warning("拒绝删除 VOICE_DIR 之外的路径: %s", p)
    del voices[name]
    save_voices(voices)
    log.info("删除音色 %s，清理文件: %s", name, removed or "无")
    return {"ok": True, "removed": removed}


@app.get("/api/download/{fn:path}")
async def download(request: Request, fn: str):
    require(request)
    # 分块输出后成品放在「标题同名子目录」里，fn 会带斜杠，所以这里用
    # path 转换器。必须校验解析后的真实路径仍在 OUTPUT_DIR 内，防目录穿越。
    path = os.path.realpath(os.path.join(OUTPUT_DIR, fn))
    root = os.path.realpath(OUTPUT_DIR)
    if not (path == root or path.startswith(root + os.sep)):
        raise HTTPException(400, "非法路径")
    if not os.path.exists(path):
        raise HTTPException(404, "文件不存在")
    u = current_user(request)
    # 权限：管理员可下载全部，员工只能下载自己的
    if u and u["role"] != "admin":
        owner = _owner_of_file(fn)
        if owner and owner != u["username"]:
            raise HTTPException(403, "无权下载")
    nice = os.path.basename(fn)
    for j in jobs.values():
        if _job_has_file(j, fn):
            t = (j.get("title") or "").strip()
            if t:
                safe_t = _safe_name(t)
                ext = "mp3" if fn.endswith(".mp3") else "wav"
                nice = f"{safe_t}-{j['id']}.{ext}"
            break
    media = "audio/mpeg" if fn.endswith(".mp3") else "audio/wav"
    return FileResponse(path, media_type=media, filename=nice)


def _job_has_file(j: dict, fn: str) -> bool:
    """该任务是否持有这个文件（兼容老任务的 wav/mp3 单文件字段）。"""
    if j.get("wav") == fn or j.get("mp3") == fn:
        return True
    return any(f.get("path") == fn for f in (j.get("files") or []))


def _owner_of_file(fn: str):
    return next((j.get("owner") for j in jobs.values() if _job_has_file(j, fn)), None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
