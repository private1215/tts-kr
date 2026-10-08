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
import random
import asyncio
import logging
import shutil
import subprocess
import threading
from typing import Optional, List

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response

from tts_engine import TTSEngine
from voice_trim import trim_reference
from voice_match import get_target
from pipeline import (CHUNK_MAX_CHARS, split_chunks, chunk_limit_chars,
                      split_by_seconds, chunk_cps, DEFAULT_CPS,
                      CHUNK_MAX_SECONDS)
import asr_check
import cosy_client
import vram
import engine_select
import pba
import envshape
import dyn
import transient           # 瞬态整形：抬 CF / 峰值（r21，已证伪默认关）
import roomtone           # 房间底噪：修「停顿是数字真零」造成的瞬断与电流声（r22）
import peakfit            # 峰值定点：把真峰值软削到案例的水平（r23，已证伪关）
import slowdyn            # 慢尺度能量轮廓：补段落级轻重，治 AI 味（r25）
import prosody_report
try:
    import proso          # 韵律 A/E 的 F0 重映射（依赖 parselmouth/Praat）
except Exception as _e:   # 没装就优雅降级，不影响主链路
    proso = None
    log.warning("proso 未启用（parselmouth 不可用）: %r", _e)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("server")

# 数据/模型根目录。默认 /mnt/tts（现网数据盘挂载点）。
# 换服务器时不改代码：设环境变量 TTS_BASE 即可整体迁移
# （systemd 里加 Environment=TTS_BASE=/新路径）。不设则与旧行为完全一致。
BASE = os.environ.get("TTS_BASE", "").strip() or "/mnt/tts"
OUTPUT_DIR = os.path.join(BASE, "output")
DATA_DIR = os.path.join(BASE, "data")
VOICE_DIR = os.path.join(DATA_DIR, "voices")
# 案例素材库：与 voices 严格分开。
#   voices = 克隆源（~15s 短片段，喂给模型当 prompt，是「输入」）
#   cases  = 对标目标（几分钟整段成品 + 原文，是「标准答案」）
# 混在一起会让人分不清哪个是输入、哪个是判卷标准，所以独立成库。
CASE_DIR = os.path.join(DATA_DIR, "cases")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CASE_DIR, exist_ok=True)

USERS_FILE = os.path.join(DATA_DIR, "users.json")
JOBS_FILE = os.path.join(DATA_DIR, "jobs.json")
CASES_FILE = os.path.join(DATA_DIR, "cases.json")
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
# ⚠️ r15 起分性别：α=0.8 下男声 pauses 171.5 对案例 170（+0.9%，刚好），
#    但女声 189 对 154（**+22.7%**）——女声的冗余停顿明显更多，同一档 α
#    压不动。女声收到 0.74：压得更狠 → 更多短停顿掉到 VAD 阈值以下被合并，
#    剩下的再由 reshuffle 拉到案例均值，于是「停顿更少、每个更长」。
PBA_ALPHA_MALE = float(os.environ.get("TTS_PBA_ALPHA_MALE", "0.8"))
# ⚠️ 0.74 那档**被数据证伪了，改回 0.8**。实测两档对比（女声）：
#    α=0.8  pauses +21.4% / +13.0%   F 0.820 / 0.677
#    α=0.74 pauses +28.6% / +14.9%   F 0.807 / 0.696 / **0.535**
#    收 α 既没减少停顿数，还引入了更大的 F 波动（最低掉到 −32%）。
#    结论：女声停顿多是**模型行为**，压 α 解决不了，得在 VAD 合并层面想办法。
#    这两个常量保留，方便下一轮继续试。
PBA_ALPHA_FEMALE = float(os.environ.get("TTS_PBA_ALPHA_FEMALE", "0.8"))
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
# 停顿的**形状**照抄案例实测经验分布（r15）。关掉则回落旧的对数正态。
# 为什么需要：2026-10-06 实测女声 我们 CV 0.476 > 案例 0.374，但熵 F 0.675
# < 案例 0.834 —— 差的不是"变化量"是"分布形状"（我们前两桶 73% vs 案例 46%）。
# 小 CV 的对数正态恰恰会把停顿挤成一坨，调 CV 是南辕北辙。
PAUSE_FROM_CASES = os.environ.get("TTS_PAUSE_FROM_CASES", "1") != "0"
# ---- 停顿条数预算（r22b）：静音占比是唯一五例同向的系统性缺陷 ----------
# 2026-10-07 五案例「成品 − 案例」带符号偏差，只有三项方向一致：
#     sil +7.8 ~ +17.1%（均值 +12.5%） / C +2.3 ~ +9.5% / dur -0.7 ~ -4.1%
# 拆开 sil 看，均值一样、多的是**条数**：案例 1a6c20 是 162 处 / 90.7s
# （均值 0.560s），我们是 178 处 / 94.8s（均值 0.533s），按字数归一后
# 多 11.5% 个停顿。α 是给每条停顿乘系数、mu 是给每条定均值，**条数多了
# 总量就是多**，两个旋钮都压不动（r15 已实测：α 收到 0.74 反而更糟）。
# 所以给条数一个预算：案例的「停顿数 / 字数」密度 × 本篇字数，超出的
# 取最短的压到 FLOOR_MS（60ms，低于 min_sil=0.25s 检不出来）。
# 1.0 = 全量对齐案例密度；0.6 = 只走六成（保守）；0 = 关。
PAUSE_BUDGET = float(os.environ.get("TTS_PAUSE_BUDGET", "1.0"))
# ---- 时长对齐（r22b）：成品时长按案例语速给，不再沿用模型的原生长度 --
# 我们五例成品**全部**比案例短（−0.7% ~ −4.1%，均值 −2.3%）：PBA 的 atempo
# 只是把「被 α 压掉的静音」拉回，从没对准过案例的语速。这里按
# 「案例库 Σ字数 / Σ时长」反推目标时长，用同一档 atempo 实现——
# atempo 是整体等比缩放，静音占比 / F / A / E 全部不变，只动 dur 一项。
CPS_ALIGN = os.environ.get("TTS_CPS_ALIGN", "1") != "0"
# atempo 限幅：极端文本（超短/超长）不至于把语速拉到荒唐的程度
CPS_ATEMPO_LO = float(os.environ.get("TTS_CPS_ATEMPO_LO", "0.80"))
CPS_ATEMPO_HI = float(os.environ.get("TTS_CPS_ATEMPO_HI", "1.25"))

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

# ---- 能量动态扩展：补「轻重音」（韵律六维里的 C）--------------------------
# 2026-10-05 案例库逐段实测（同文案、同口径）：
#     案例男 seg0/seg1 C=14.09/14.77 dB   我们 chunk1/2 12.11/12.77   低 14~15%
#     案例女 seg0/seg1 C=14.85/15.37      我们 chunk1/2 14.46/13.95   低 3~9%
# C = RMS 包络 p10~p90 的宽度，听感上就是「有没有轻重音」。我们偏低意味着
# 整段都差不多响 —— 这正是员工说的「像读课本 / 没感情」。
# 男声尤其糟，而且它同时就是 A（句内音高动态）偏高的那一个：
# 音高夸张 + 力度平 = 员工原话「男声前半段跟吵架似的」。
#
# 做法是给慢包络乘一条平滑增益曲线（中置的向上/向下扩展器），
# 不动频谱、不动时长、不过声码器，所以没有音质风险；
# 静音帧统一夹紧，句数 / 停顿熵不会被误伤（那是 PBA 管的）。
# 详见 app/dyn.py 文件头。设 TTS_DYN=0 关闭。
DYN_EXPAND = os.environ.get("TTS_DYN", "1") != "0"
# 🔴 r21b：**压 dyn 来修 C 这条路走不通，已回退**。
#    实证：把目标从 14.4/15.1 压到 12.7/13.6（放开 k<1 压缩）后
#        ✅ C 能量动态 系统性 +8.2% → **0.0%**（带符号），这正是想要的
#        ❌ 但**峰值全面崩**：标杆女 −4.20 → **−5.10**、리노y → 36.7% 偏差
#        ❌ 音色 LTAS 男声反而恶化（5.2%→10.4%、9.8%→14.0%）
#    根因：压缩慢包络 = 把响的那部分真压下去，而**峰值本来就长在响帧里**；
#          C 与 CF 在这个模块里是**此消彼长**，不能两头都要。
#    教训：**别为了让一个指标好看而牺牲另一个更重的因果关系**。
#    C 系统性 +8% 是已知、可接受、且**不值得用 CF 去换**的偏差。
DYN_C_MALE = float(os.environ.get("TTS_DYN_C_MALE", "14.4"))
DYN_C_FEMALE = float(os.environ.get("TTS_DYN_C_FEMALE", "15.1"))
DYN_C_DEFAULT = float(os.environ.get("TTS_DYN_C", "14.7"))
# ---- r25 慢尺度能量轮廓（治 AI 味的头号病因）------------------------------
# 目标值来自 24 条案例普查（tools/caseslow.py），分性别存在 slowdyn.py 里。
# 默认开：五例同向、且与案例零重叠，是诊断出来的系统性缺陷不是噪声。
SLOWDYN = os.environ.get("TTS_SLOWDYN", "1") != "0"
SLOWDYN_MIN_CHARS = int(os.environ.get("TTS_SLOWDYN_MIN_CHARS", "120"))
# ---- r25 全局 f0 基线对齐 --------------------------------------------------
# 句级 f0 中位我们比案例低 4.5~9.9%（男 181 vs 196.2、女 265.3 vs 280.2）。
# 目标用**案例库 Praat 口径**的中位（proso_med_targets），当前值也用
# proso.med_of（Praat 口径）实测 —— 两端同源，不做跨口径换算（陷阱 #4）。
PROSO_MED = os.environ.get("TTS_PROSO_MED", "1") != "0"
PROSO_MED_FALLBACK_MALE = float(os.environ.get("TTS_PROSO_MED_MALE", "196.2"))
PROSO_MED_FALLBACK_FEMALE = float(
    os.environ.get("TTS_PROSO_MED_FEMALE", "280.2"))
PROSO_MED_FALLBACK_DEFAULT = float(
    os.environ.get("TTS_PROSO_MED_DEFAULT", "196.2"))

# 响度闭环死区（dB）：编完成品 mp3 实测 LUFS，偏差超过它才补增益重编。
# 🔴 r25b 实测 0.5 太松：slowdyn / f0 基线平移会给成品引入 0.3~0.5 dB 的
#    LUFS 偏差，正好落在死区里被放过 → lufs 项失分 3.3~8.3（权重 .06）。
#    收紧到 0.15，代价只是多一次编码（几秒），且不兜圈子（最多重编一次）。
LOUD_TOL = float(os.environ.get("TTS_LOUD_TOL", "0.15"))

# ---- 瞬态整形：修「峰值因数 CF 比案例低」（r21 新增）----------------------
# 唯一能抬峰值而不被响度归一 1:1 吃掉的办法（compand 向上扩展已证伪）。
# 分性别是必须的：女声 CF 差 1.80 dB、男声只差 0.60 dB，差 3 倍，
# 根因是女声 F0 高 → PSOLA 周期短 → 重叠相加平均得更狠 → 波形更平。
# 标定见 app/transient.py 文件头（标杆女扫描：0.2 档 CF 20.50 = 案例 20.50）。
# 🔴 r21 最终结论：**默认关**。离线标定（/tmp/tshaper.py）在标杆女上
#    amount=0.20 能把 CF 从 18.70 抬到 20.50（= 案例），看着是 +1.8 dB 的大胜。
#    但连跑三轮线上五案例都没兑现：
#        r21  整形@链前（错位置）  标杆女峰值 −4.20 → **−5.10**
#        r21b 只保留整形          标杆女峰值 −4.20 → **−5.10**
#        r21c 整形改到 EQ/空气感后 标杆女峰值 −4.20 → **−5.00**
#    根因：离线标定量的是**整形后的 wav**，而线上成品是 wav → **libmp3lame** →
#    mp3（验收还要再解码到 24k）。最后那一道编码把峰值基本重新定了，
#    我们在 wav 上抬起来的 0.8 dB 到 mp3 里只剩 0.1 dB。
#    → 沉淀为测量陷阱 #18：**抬峰值类的改动必须在「成品口径」上标定**，
#      在中间 wav 上量出来的收益不能外推。
#    模块保留（TTS_TRANSIENT=1 可开），但不在默认链路里。
TRANSIENT = os.environ.get("TTS_TRANSIENT", "0") != "0"
TRANSIENT_MALE = float(os.environ.get("TTS_TRANSIENT_MALE", "0.06"))
TRANSIENT_FEMALE = float(os.environ.get("TTS_TRANSIENT_FEMALE", "0.20"))
TRANSIENT_DEFAULT = float(os.environ.get("TTS_TRANSIENT_AMOUNT", "0.12"))
TRANSIENT_MIN_CHARS = int(os.environ.get("TTS_TRANSIENT_MIN_CHARS", "120"))
DYN_MIN_CHARS = int(os.environ.get("TTS_DYN_MIN_CHARS", "120"))

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
# ⚠️ r15 起**分性别**：男女各一套。
#    原来是共用的，只取「男女同向的保守值」，结果女声 1480Hz 的 3.26 dB
#    缺口只补了 0.72 dB（EQ 残差 1.7 dB 下不去）。而「同向」这个约束
#    本来只是「一套 EQ 服务两个性别」的妥协——分性别后就不需要了，
#    方向相反的频段（65/262/370/740…）是男女音色本身的差异，不是误差。
# 求法：/tmp/eqsolve.py 迭代 3 轮（LTAS 按总能量归一，是零和游戏，
#    一次性补偿只兑现约 60%，必须迭代），补偿量 clip 到 [0.85, 1.6]。
#    女声 1.74 -> 0.66 dB，男声 1.7 -> 0.50 dB（185Hz 主动放弃，见下）。
TONE_EQ_FILTER_MALE = os.environ.get(
    "TTS_TONE_EQ_FILTER_MALE",
    # ↑↑ 2026-10-07 r20：**多案例联合拟合**（/tmp/eqsolve2.py）。
    #    旧值只拿标杆男 1 例拟合，换到 yohan 3 例就有 0.56~0.61 dB 偏差。
    #    新目标谱 = 4 个男声案例的平均 LTAS，误差谱 = 4 个成品的平均 LTAS，
    #    迭代 3 轮求一组对**所有案例同时最优**的 EQ：
    #        平均 0.555 -> 0.403 -> **0.345 dB**，且逐例**全都一起变好**
    #        （0.58/0.61/0.56/0.58 -> 0.36/0.36/0.35/0.35），没有互相牺牲。
    #    离线试算（只叠补偿倍数，别叠完整新 EQ —— 成品已打过一次 EQ，再打 = 双倍）：
    #        标杆男 音色 0.49 -> **0.36 dB**、C 4.7% -> 2.9%、总分 93.13% -> **94.60%**。
    #    补偿倍数 0.594/1.045/0.946/1.0/1.024/0.990/1.015/0.920/0.999/
    #             1.043/0.974/0.954/1.003/1.072/1.123。185Hz 仍主动放弃。
    # ↑↑ 2026-10-08 r25e：**响应矩阵 + 最小二乘**重新拟合（tools/eqfit.py）。
    #    r20 那版是「逐段独立补偿」，在 r25d 上实测**发散**（1.073→1.328→
    #    2.229→5.026 dB）—— 🔑 LTAS 按总能量归一是零和的：实测给 740Hz 单独
    #    +6dB，它自己只涨 4.28 dB，**其余 14 段全部跌 1.05 dB**；65Hz 甚至
    #    完全无响应。所以必须**先测响应矩阵 R（含串扰与归一化），再联立解**。
    #    起点 1.064 dB → **0.406 dB**。
    #    ⚠️ **低频 3 段（65/92/131Hz）一个都不动**（r25f，见 FEMALE 下方注释）：
    #       抬低频会连带把停顿里的底噪抬起来 → 验收 VAD 是相对阈值 p25+6dB
    #       → 判句/停顿检出漂移 → D 亏 0.43、sil 亏 0.16（r25e 实测）。
    #       而且那几列响应本来就不可靠（65Hz 完全无响应、92Hz 很弱），
    #       硬补只会引入误差 —— 实测不动时 tone 反而更好（6.1 vs 8.3）。
    #    ⚠️ 新值 = 旧值 × 补偿倍数，不是替换（成品已打过一次 EQ）。
    "superequalizer=1b=0.800:2b=1.524:3b=0.817:4b=1.138:5b=1.178:6b=1.059:7b=1.243:"
    "8b=1.081:9b=1.331:10b=1.193:11b=1.097:12b=0.951:13b=1.237:"
    "14b=1.048:15b=1.202:16b=1.000:17b=1.000:18b=1.000").strip()
TONE_EQ_FILTER_FEMALE = os.environ.get(
    "TTS_TONE_EQ_FILTER_FEMALE",
    # ↑↑ 2026-10-06 r18 再迭代 3 轮（旧值 × 补偿倍数 1.085/1.075/1.070/1.0/
    #    1.041/1.021/1.124/1.084/1.005/1.456/0.952/1.052/1.120/1.070/1.193）。
    #    女声音色偏差 0.696 -> 0.476 dB（EQ 同源口径，185Hz 仍主动放弃）。
    # ↑↑ 2026-10-08 r25e：同样用 tools/eqfit.py 的响应矩阵最小二乘重拟合。
    #    女声旧值还是 r18 那版（2026-10-06），之后再没动过，起点最差
    #    （**1.606 dB**，是五例里失分最大的一例）。
    #    🔴 r25e 第一版（低频也补）上线后 tone 确实从 19.9 降到 8.76，但
    #       **D 从 4.68 涨到 10.08、sil 从 1.90 涨到 4.14**，净值只有 +0.27。
    #       r25f 改为低频 3 段不动后，tone 1.606 -> **0.854 dB**。
    "superequalizer=1b=2.410:2b=1.460:3b=0.931:4b=1.138:5b=0.907:6b=1.570:7b=1.501:"
    "8b=1.575:9b=1.791:10b=1.786:11b=1.467:12b=1.640:13b=1.354:"
    "14b=1.292:15b=1.357:16b=1.000:17b=1.000:18b=1.000").strip()
# 兼容旧环境变量：没分性别时用男声那套（保持单性别场景行为不变）
TONE_EQ_FILTER = os.environ.get(
    "TTS_TONE_EQ_FILTER", TONE_EQ_FILTER_MALE).strip()
# ↑↑ 2026-10-06 迭代过一轮（r13 之后）。上面注释里那组旧值是 2026-09-30 口径下
#    fitted 的，实测**方向反了**：成品低频已比案例多 4 dB，旧值还在加 4~5 dB。
#    新值 = 旧值 × 补偿，补偿量来自长时平均谱（LTAS）量出的男女同向偏差，
#    方法见 /tmp/ltas.py 与 /tmp/eqcheck.py，数据见 docs/r13 §7。
#    离线试算（对现有成品叠加补偿，等价于从头用新 EQ）：
#      平均绝对偏差(<11kHz)  男 1.97 -> 0.59 dB   女 2.81 -> 1.72 dB
#    残差（下一轮再迭代，用同样方法）：1047Hz 仍缺 ~1.8 dB、4186 缺 1.6；
#      女声 185Hz 差 6.5 dB 是「男女同向取保守值」策略主动放弃的
#      —— 那里是基频/第一共振峰区，强对齐会改掉音色身份，不能动。
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

# ---- 成品响度归一化：把整篇拉到案例的响度（2026-10-06 r14）--------------
# 实测（EBU R128 积分响度，ffmpeg ebur128）：
#     案例男 -24.3 LUFS    我们 r13 -31.5   → 差 7.0 dB
#     案例女 -22.9 LUFS    我们 r13 -27.9   → 差 4.3 dB
# 7 dB 是最容易被耳朵抓住的差别（"不像同一个节目"），比频谱差更要命。
#
# 为什么不用静态常数补：上面的 -4.5 / -1.7 / +0.5 只对**当前这两个音色**
# 成立。换参考音频、换性别、换文案，合成出来的电平就变了，常数立刻失效，
# 而团队会持续录入新音色，不能每次手工重标一遍。
# 所以改成**逐块实测再反算增益**：先量未压缩 wav 的 LUFS，反推要补多少 dB。
#
# LOUD_CHAIN_DB 是后面「音色 EQ + 空气感」链路自身带来的响度变化
# （实测：新 EQ 男 +0.2 / 女 +0.7 dB）。我们量的是链路**前**的 wav，
# 不减掉它会系统性偏响。
# LOUD_LIMITER 是兜底限幅：案例真峰值男 -1.7 / 女 -2.4 dBFS，
# 照着这个卡（0.822 ≈ -1.7 dBFS），正常只在最热的几个瞬态上动 ~1 dB。
LOUD_NORM = os.environ.get("TTS_LOUD_NORM", "1") != "0"
# 兜底值 = 2026-10-06 实测的两份 ElevenLabs 案例（只有两点，不是定理）
LOUD_TARGET_MALE = float(os.environ.get("TTS_LOUD_TARGET_MALE", "-24.3"))
LOUD_TARGET_FEMALE = float(os.environ.get("TTS_LOUD_TARGET_FEMALE", "-22.9"))
LOUD_TARGET_DEFAULT = float(os.environ.get("TTS_LOUD_TARGET", "-23.6"))
# ⚠️ 2026-10-06 连改两次才对（都记录在案，别再踩）：
#   初版 0.5 —— 拿「整篇 LUFS」减「单块 LUFS」反推的，两边口径不一致，
#     结果系统性偏响 2.5~3.1 dB（成品男 -21.8 / 女 -19.8，目标是 -24.3 / -22.9）。
#   二版 3.0 —— 按单块重算，但仍差 1.6 dB。
#   现版 4.6 —— 单块实测反推：I_pre -25.9、增益 -1.4、首编成品 -22.70。
# 它只是**初值**：下面还有一道闭环复检兜底，填错了也不会让成品跑飞，
# 只是多花一次重编码。所以别再手工精调这个数，改链路时让它自愈。
LOUD_CHAIN_DB = float(os.environ.get("TTS_LOUD_CHAIN_DB", "4.6"))
# 0.79 ≈ -2.05 dBFS。卡到这个数的理由：
#   案例峰值因数（峰值 - LUFS）= -1.7 - (-24.3) = 22.6 dB。
#   限幅设 -1.7 时成品峰值冲到 -0.2（lame 过冲 ~1.5 dB，会削波）；
#   设 -3.2 时成品 -3.3，峰值因数只有 21.0，比案例憋 1.6 dB。
#   -2.05 是折中：过冲后约 -2.0，峰值因数 ≈ 22.3，与案例差 0.3 dB 且不削波。
LOUD_LIMITER = os.environ.get(
    "TTS_LOUD_LIMITER",
    "alimiter=limit=0.79:attack=5:release=50").strip()
# ⚠️ r15 起**分性别限幅**。
#    起因是女声 seed42 峰值冲到 −0.80 dBFS（案例 −2.40），只按这一次
#    测就想把女声压到 0.66 是错的——seed777 同配置却是 −3.80，
#    **峰值单次波动达 3 dB**，中位数 −2.30 本来就已经对准案例 −2.40。
#    （又一次踩 §6-7：别为单次波动调参。）
#    0.74 那一版也错了：改完两个种子都是 −4.30，中位比案例低 1.9 dB。
#    最终**回到 0.79**（与男声一致）——r15d 在 0.79 下的中位 −2.30
#    本来就对准案例 −2.40（差 0.1 dB），−0.80 那个种子虽余量小但并未削波。
#    保留分性别这两个常量，是为了以后某一性别需要单独微调时不用再改结构。
LOUD_LIMITER_MALE = os.environ.get(
    "TTS_LOUD_LIMITER_MALE",
    "alimiter=limit=0.79:attack=5:release=50").strip()
LOUD_LIMITER_FEMALE = os.environ.get(
    "TTS_LOUD_LIMITER_FEMALE",
    "alimiter=limit=0.79:attack=5:release=50").strip()

# ---- 块首尾静音：对齐案例的换气量 ----------------------------------------
# ⚠️ 2026-10-05 用案例库重测，推翻了原先「案例段落间 0.45~0.66 s」的说法
#    （那次是在别的素材上量的，且含底噪）。案例库两个样本逐段量首尾静音：
#        案例男 seg0 尾 0.21 s + seg1 头 0.04 s = 段间 0.25 s
#        案例女 seg0 尾 0.30 s + seg1 头 0.06 s = 段间 0.36 s
#    而我们（补到 600 ms 那版）：chunk1 尾 0.48 + chunk2 头 0.20 = 0.68 s，
#    是案例的两倍多 —— 拼接后听着「拖」。
# 所以改成：首静音裁到 60 ms、尾静音统一补/裁到 220 ms，
# 段间 = 0.22 + 0.06 = 0.28 s，正落进案例区间。
# 两个都设 0 即关闭。
# ---- 韵律 A / E 的 F0 重映射（app/proso.py，Praat PSOLA）-------------------
# 目标值是「案例音频用同一套 Praat F0 公式量出来的」，与 prosody_report 的
# librosa.pyin 口径数值不同（男 E 45.5 vs 34.6），但同源自洽，k=target/cur 成立。
# ⚠️ 默认关闭：PSOLA 是重合成，有音质代价（可能引入金属感/卡顿），
#    必须先跑一轮开着的全篇让耳朵验收，再决定要不要默认开。
PROSO = os.environ.get("TTS_PROSO", "0") != "0"
PROSO_MIN_CHARS = int(os.environ.get("TTS_PROSO_MIN_CHARS", "120"))
PROSO_A_MALE = float(os.environ.get("TTS_PROSO_A_MALE", "9.495"))
PROSO_E_MALE = float(os.environ.get("TTS_PROSO_E_MALE", "45.493"))
PROSO_A_FEMALE = float(os.environ.get("TTS_PROSO_A_FEMALE", "9.498"))
PROSO_E_FEMALE = float(os.environ.get("TTS_PROSO_E_FEMALE", "33.511"))
PROSO_A_DEFAULT = float(os.environ.get("TTS_PROSO_A_DEFAULT", "9.50"))
PROSO_E_DEFAULT = float(os.environ.get("TTS_PROSO_E_DEFAULT", "39.50"))

# ---- 🔴 目标值校准系数：Praat 达标 ≠ pyin 达标（2026-10-06 r14 实测）------
# 现象（job 08b47b0d 男 / e027e8af 女，整篇同文案）：
#     男 处理前 Praat E 40.07 --k_tail=1.153--> 目标 45.49，
#        但成品用 pyin 量只有 **29.848**，案例 pyin 是 **34.559**（−13.6%）
#     男 处理前 Praat A 9.64 --k=0.985--> 目标 9.49，成品 pyin 9.150 vs 案例 8.362（+9.4%）
# 即：**k 精确算到位了、Praat 口径也到目标了，pyin 口径却差一大截**。
# 根因见 MEMORY §6-5：两口径的换算系数在 PSOLA 重合成后失效
#   （案例 Praat/pyin 比：A 1.135、E 1.316；我们处理后：A 1.09、E 1.52）。
# 所以只能**按实测反推校准目标值**，不能用理论比值换算。
#
# 系数 = (案例 pyin / 成品 pyin) ^ γ，γ=0.85 是保守指数：
#   全量（γ=1）在单次波动 ±5% 的前提下容易过冲，而且 PSOLA 的 artifacts
#   与位移量成正比（Interspeech 2020 实测），男 E 的 k_tail 会从 1.15 涨到
#   1.31，拉太满有金属感风险。留 15% 余量，下一轮按复跑结果再收敛。
PROSO_CAL_A_MALE = float(os.environ.get("TTS_PROSO_CAL_A_MALE", "0.926"))
# ⚠️ r15 第二轮改回 1.0：1.133 是拿 r14（旧链路）的 E=29.848 反推的，
#    但 r15 把停顿加长到案例水平后，**链路本身就把 E 抬上去了**——
#    同样 CAL=1.133 下成品 E 中位 38.416 vs 案例 34.559，冲过零点 +11.2%。
#    用案例的口径比（成品 pyin / 目标 Praat = 34.559/45.49 = 0.760）反推，
#    新链路下 CAL=1.0 应给出 45.49×0.760 = 34.57 ≈ 案例值。
#    教训：**校准系数不能跨链路搬**，链路一动就要重新实测反推。
PROSO_CAL_E_MALE = float(os.environ.get("TTS_PROSO_CAL_E_MALE", "1.0"))
PROSO_CAL_A_FEMALE = float(os.environ.get("TTS_PROSO_CAL_A_FEMALE", "0.971"))
# 同 CAL_E_MALE：0.959 也是按旧链路反推的，r15c 实测女声 E 中位 31.578
# vs 案例 34.410（−8.2%）。用案例口径比（34.410/33.51 = 1.027）反推，
# CAL=1.0 时目标 33.51 应给出 ≈34.4。
PROSO_CAL_E_FEMALE = float(os.environ.get("TTS_PROSO_CAL_E_FEMALE", "1.0"))
PROSO_CAL_DEFAULT = float(os.environ.get("TTS_PROSO_CAL_DEFAULT", "1.0"))
CAL = {"male": (PROSO_CAL_A_MALE, PROSO_CAL_E_MALE),
       "female": (PROSO_CAL_A_FEMALE, PROSO_CAL_E_FEMALE)}
# 目标值改为从案例库动态聚合（2026-10-06）：上传的案例越多，目标值越稳。
# 上面那几个常量退化成「案例库里没有对应性别时」的兜底值。
# ⚠️ 必须用案例的 **Praat 口径** A/E（case["proso"]），不能用六维报告里的
#    pyin 口径 —— 两口径在 PSOLA 重合成后的换算系数会变（docs/r13 §5）。
PROSO_FROM_CASES = os.environ.get("TTS_PROSO_FROM_CASES", "1") != "0"
# 只把「标杆平台」的案例当目标。我们自己的输出（source=qwen）不能当自己的尺子，
# 否则会自我强化、越跑越偏。
PROSO_CASE_SOURCE = set(filter(None, os.environ.get(
    "TTS_PROSO_CASE_SOURCE", "elevenlabs").split(",")))

EDGE_HEAD_S = float(os.environ.get("TTS_EDGE_HEAD_S", "0.06"))
EDGE_TAIL_S = float(os.environ.get("TTS_EDGE_TAIL_S", "0.22"))
EDGE_SIL_DB = float(os.environ.get("TTS_EDGE_SIL_DB", "-45"))


def sil_edges(audio, sr, sil_db=EDGE_SIL_DB):
    """量首尾静音秒数（只用于诊断日志 / fit_edges 内部）。

    与 fit_edges 同一口径：20 ms 帧 RMS，低于 sil_db 算静音。
    第三个返回值是整段最响帧的 dB —— 用来判断「是不是整段都太安静，
    导致一个响帧都挑不出来」（那样 fit_edges 会直接放弃、静默不生效）。
    """
    a = np.asarray(audio, dtype=np.float64)
    fr = max(int(sr * 0.02), 1)
    n = len(a) // fr
    if n < 4:
        return None, None, None
    e = np.sqrt((a[:n * fr].reshape(n, fr) ** 2).mean(axis=1))
    db = 20 * np.log10(e + 1e-9)
    loud = np.where(db > sil_db)[0]
    dmax = float(db.max())
    if loud.size == 0:
        return None, None, dmax
    return (int(loud[0]) * fr / float(sr),
            (n - int(loud[-1]) - 1) * fr / float(sr), dmax)


def fit_edges(audio, sr, head_s=EDGE_HEAD_S, tail_s=EDGE_TAIL_S,
              sil_db=EDGE_SIL_DB):
    """把首尾静音修到指定秒数：长了裁掉，短了补上。

    只碰静音（低于 sil_db 的部分），不动任何有声样本，无音质影响。
    """
    if (head_s <= 0 and tail_s <= 0) or len(audio) < sr * 0.2:
        return audio
    fr = max(int(sr * 0.02), 1)
    n = len(audio) // fr
    if n < 4:
        return audio
    e = np.sqrt((audio[:n * fr].reshape(n, fr) ** 2).mean(axis=1))
    db = 20 * np.log10(e + 1e-9)
    loud = np.where(db > sil_db)[0]
    if loud.size == 0:
        return audio
    lead = int(loud[0]) * fr          # 首静音样本数
    trail = (n - int(loud[-1]) - 1) * fr
    if tail_s > 0:
        need = int(round(tail_s * sr)) - trail
        if need > 0:
            audio = np.concatenate([audio,
                                    np.zeros(need, dtype=audio.dtype)])
        elif need < 0:
            audio = audio[:len(audio) + need]   # need 为负，裁掉多余尾巴
    if head_s > 0:
        need = int(round(head_s * sr)) - lead
        if need > 0:
            audio = np.concatenate([np.zeros(need, dtype=audio.dtype), audio])
        elif need < 0:
            audio = audio[-need:]               # 裁掉多余头部静音
    return audio


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


def load_cases():
    return load_json(CASES_FILE, {})


def save_cases(c):
    save_json(CASES_FILE, c)


cases = load_cases()


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
    # ⚠️ 引擎不可用就**现在**退回 Qwen，不要等到 vram.switch_to 才发现：
    #    那时 Qwen 已经被卸出显存，失败后还得重新加载一次，白花一两分钟
    #    装卸时间，员工只看到「正在分段」卡住。2026-10-06 加。
    if _engine_name != "qwen" and not engine_available(_engine_name):
        log.warning("任务 %s 情感=%s 需要引擎 %s，但该引擎当前不可用 -> 直接用 qwen",
                    job["id"], _emotion, _engine_name)
        job["fallback_from"] = _engine_name
        job["fallback_reason"] = "engine_unavailable"
        _engine_name = "qwen"
        _cosy_instruct = ""
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
    #
    # ⭐ 上限不再是固定字数，而是「预估音频时长」——2026-10-05 案例拆解结论：
    #   ElevenLabs 按约 250 秒音频一块切（男声切在第 2157 字、女声第 1879 字，
    #   字数差 15%，换算成时长却都是 251~258 秒）。
    #   按字数切就会两性别切出完全不同的段长，我们第一块甚至到 352 秒，
    #   比案例长 40%，而长上下文正是语气漂移的温床。
    #   cps 用该音色的历史实测值（任务跑完自动写回 voices.json），无历史时用
    #   DEFAULT_CPS 起步，跑过一单就收敛。
    _vmeta = voices.get(job["voice"]) or {}
    _vcps = float(_vmeta.get("cps") or 0) or DEFAULT_CPS
    # 🔴 r22：分块用**锁定**的语速，不用实时 cps —— 否则 cps 每单漂移就会让
    #    同一篇文案的段数在 2/3 之间跳（ba24c0 实测 r21c 3 块 / r22 2 块）。
    _vcps_chunk, _need_lock = chunk_cps(_vmeta, _vcps)
    if _need_lock and job.get("voice") and _vmeta is not None:
        _vmeta["cps_lock"] = _vcps_chunk
        save_voices(voices)
    _spd = float(job.get("speed") or 1.0)
    _limit = chunk_limit_chars(_vcps_chunk, _spd)
    chunks = split_by_seconds(job["text"], cps=_vcps_chunk, speed=_spd)
    # 退化路径：整篇没有句末标点时 split_by_seconds 会返回整块，
    # 这时候才回退到字数贪心（它至少能在换行的地方断开）。
    if len(chunks) <= 1 and len(job["text"] or "") > _limit:
        chunks = split_chunks(job["text"], max_chars=_limit)
    n_chunk = max(len(chunks), 1)
    job["chunk_limit"] = _limit
    log.info("任务 %s 分块 %d 块（预估时长 %.0fs / 预算 %.0fs，音色 cps=%.2f "
             "speed=%s，字数上限 %d）",
             job["id"], n_chunk,
             len(job["text"] or "") / max(_vcps * _spd, 0.1),
             CHUNK_MAX_SECONDS, _vcps, _spd, _limit)
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

        # ---- 性别判定：整段只判一次（同一个音色不会有男女混着的情况）------
        # r15 起提前到 PBA 之前：停顿重排要按性别取案例的停顿样本，能量扩展
        # 的目标值也是分性别的（案例男 14.4 / 女 15.1）。
        # PBA/包络整形都不改有声样本，F0 判性别早判晚判结果一致。
        if "_gain_off" not in job:
            _off = 0.0
            job["_is_male"] = None
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
                    job["_is_male"] = bool(_male)
                    _off = GAIN_MALE_DB if _male else GAIN_FEMALE_DB
                    log.info("任务 %s 性别判定 f0=%.0fHz -> %s，响度偏移 %+.1f dB",
                             jid, _f0, "男" if _male else "女", _off)
            job["_gain_off"] = _off
        # 停顿样本只取一次并缓存到任务上：调 pause_targets 要遍历案例库，
        # 8 分钟的长篇有几十块，没必要每块都重算。
        _g = ("male" if job.get("_is_male") else
              "female" if job.get("_is_male") is False else None)
        if "_pause_samples" not in job:
            try:
                job["_pause_samples"] = pause_targets(_g) if _g else None
            except Exception as _e:
                log.warning("任务 %s 停顿样本取用失败: %r", jid, _e)
                job["_pause_samples"] = None
            if job["_pause_samples"]:
                log.info("任务 %s 停顿分布借案例经验分布：%d 个样本",
                         jid, len(job["_pause_samples"]))
        # 停顿**条数**密度与案例语速：与 _pause_samples 一样每单只算一次
        if "_pause_density" not in job:
            try:
                job["_pause_density"] = pause_density(_g) if _g else None
            except Exception as _e:
                log.warning("任务 %s 停顿密度取用失败: %r", jid, _e)
                job["_pause_density"] = None
        if "_cps_target" not in job:
            try:
                job["_cps_target"] = cps_target(_g) if _g else None
            except Exception as _e:
                log.warning("任务 %s 案例语速取用失败: %r", jid, _e)
                job["_cps_target"] = None
        if "_peak_target" not in job:
            try:
                job["_peak_target"] = peak_target(_g) if _g else None
            except Exception as _e:
                log.warning("任务 %s 案例峰值取用失败: %r", jid, _e)
                job["_peak_target"] = None

        # 停顿预算对齐（PBA）：只切静音，不动任何有声样本，无音质损失。
        # 丢掉的静音时间在下面用 atempo 拉回，总时长与发音速率一起回到
        # 案例原声的水平。放在 ASR 回检之后——CER 要量的是模型的原始输出。
        _pba_ratio = 1.0
        # 分性别 α：女声冗余停顿比男声多 22.7%，同一档压不动
        _alpha = (PBA_ALPHA_MALE if job.get("_is_male") else
                  PBA_ALPHA_FEMALE if job.get("_is_male") is False
                  else PBA_ALPHA)
        if _alpha < 0.999 and len(chunk_text) >= PBA_MIN_CHARS:
            audio, _pba_ratio = pba.align(audio, sr, _alpha)
        # 停顿重排（在压缩之后）：把停顿长度重新分布成案例的形状+均值。
        #
        # ⚠️ r15：reshuffle 加长的这部分**不进 _pba_ratio**，也就是不参加
        #    下面的 atempo 拉回。原因：拉回 = 把总时长压回原值 = 把有声段
        #    加速。而"静音占比只有案例的 6 成、听着赶"正是要靠加长静音来
        #    修的，再加速就完全抵消了（实测会把有声段从 4.21 字/s 拉到
        #    4.60，离案例更远）。只有 align 压缩掉的那部分需要拉回——
        #    不拉回的话整段会变慢，那是 r9 就验证过的老行为。
        #    代价：成品总时长会比原时长长（静音占比向案例看齐的必然结果，
        #    案例本身就比我们长 3.5%）。
        if PBA_RESHAPE and len(chunk_text) >= PBA_MIN_CHARS:
            try:
                # 停顿条数预算：把「我们比案例多出来的那批最短停顿」压到 60ms。
                # 削掉的那部分时间**不进 atempo 拉回**——拉回 = 把有声段加速，
                # 而 sil 偏高的根因正是静音太多，拉回就白削了（与 r15 对
                # reshuffle 加长部分的处理同一套道理）。
                _pb = None
                _dens = job.get("_pause_density")
                if PAUSE_BUDGET > 0 and _dens and _dens > 0:
                    _ntgt = _dens * len(chunk_text)
                    _n_now = None
                    try:
                        _iv = pba._intervals_report(audio, sr)
                        _n_now = max(len(_iv) - 1, 0)
                    except Exception:
                        _n_now = None
                    if _n_now and _n_now > _ntgt:
                        _pb = _n_now - PAUSE_BUDGET * (_n_now - _ntgt)
                audio, _rs_ratio = pba.reshuffle(
                    audio, sr, PBA_CV,
                    target_samples=job.get("_pause_samples"),
                    pause_budget=_pb)
                job.setdefault("_rs_ratio_log", []).append(round(_rs_ratio, 4))
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

        # ---- 慢尺度能量轮廓重建（r25，治「AI 味 / 没感情」）----------------
        # 🔴 必须放在能量扩展 **之前**：slowdyn 会把帧级 C 一起推高，而 C
        #    是评分项（权重 0.08）且本来就 +2.3~+9.5% 偏高。放前面让 dyn
        #    做最后校准。dyn 是绕全局 center 等比缩放，**等比缩放不改变
        #    慢变占比**，所以段落轮廓能完整保住，只有绝对幅度被压一点。
        # 🔴 头号病因，24 条案例普查坐实：句级能量慢变占比我们 0.18~0.28 vs
        #    案例 0.390(男)/0.411(女)；ac1 我们 −0.19~−0.32 vs 案例 +0.06~+0.10，
        #    零重叠。我们不是起伏不够，是起伏全堆在"句到句"、还逐句反相。
        # 增益只作用于有声帧、均值不变 → 静音占比/F/停顿/LUFS 都不动。
        if SLOWDYN and len(chunk_text) >= SLOWDYN_MIN_CHARS:
            _male_s = job.get("_is_male")
            try:
                _ssegs, _scuts, _shop = prosody_report.sent_windows(audio, sr)
                audio, _ks, _kf, _sdiag = slowdyn.apply(
                    audio, sr, segs=_ssegs, is_male=_male_s)
                if _ks > 1.002 or _ks < 0.998 or _kf > 1.002 or _kf < 0.998:
                    _sd = (_sdiag or {}).get("stats") or {}
                    log.info("任务 %s 第 %d 块 慢尺度轮廓 k_slow=%.3f "
                             "k_fast=%.3f  慢变占比 %.3f->目标 %.3f  "
                             "句级sd %.3f->目标 %.3f  增益 %+.2f~%+.2f dB",
                             jid, ci + 1, _ks, _kf,
                             _sd.get("share", -1),
                             slowdyn._slow_target(_male_s),
                             _sd.get("sd", -1),
                             slowdyn._sd_target(_male_s),
                             (_sdiag or {}).get("gain_min", 0),
                             (_sdiag or {}).get("gain_max", 0))
            except Exception as _e:
                log.warning("任务 %s 第 %d 块 慢尺度轮廓失败（跳过）: %r",
                            jid, ci + 1, _e)

        # ---- 能量动态扩展：补「轻重音」（韵律 C）----------------------------
        # 放在包络整形之后：整形补的是音节内瞬态（8~16Hz），这里补的是
        # 词/句级的轻重（慢包络），两者互不干扰，但先定"轻重"再定"咬字"
        # 更接近真实的发声链。静音整体搬移，句数/停顿熵不受影响。
        if DYN_EXPAND and len(chunk_text) >= DYN_MIN_CHARS:
            _male = job.get("_is_male")
            _tgt = (DYN_C_MALE if _male else
                    DYN_C_FEMALE if _male is False else DYN_C_DEFAULT)
            try:
                audio, _k, _c0, _c1 = dyn.expand(audio, sr, target=_tgt)
                if _k > 1.002:
                    log.info("任务 %s 第 %d 块 能量动态 k=%.3f  C %.2f -> %s"
                             "（目标 %.2f）", jid, ci + 1, _k, _c0,
                             ("%.2f" % _c1) if _c1 else "-", _tgt)
            except Exception as _e:
                log.warning("任务 %s 第 %d 块 能量动态扩展失败（跳过）: %r",
                            jid, ci + 1, _e)

        # CosyVoice3 已在模型内部按 speed 变速（梅尔谱插值 + 声码器重合成），
        # 这里不能再叠一次 atempo，否则会变两次速。
        _speed = 1.0 if job.get("_cosy_speed_applied") else float(
            job.get("speed") or 1.0)
        # PBA 削掉的静音时间靠这一档 atempo 补回来：最终时长 = 原时长 / speed，
        # 与不开 PBA 时一致，员工侧的时长预估和统计都不受影响。
        _atempo = _speed * _pba_ratio
        # ---- 时长对齐案例语速（r22b）--------------------------------------
        # 我们五例成品**全部**比案例短（−0.7% ~ −4.1%，均值 −2.3%）：上面
        # 这档 atempo 只负责把「被 α 压掉的静音」拉回，从来没对准过案例语速，
        # 于是模型原生多快、成品就多快。这里按案例库 Σ字数/Σ时长 反推目标
        # 时长，仍用同一档 atempo 实现 —— atempo 是整体等比缩放，静音占比、
        # F、A、E 全部不变，只动 dur 一项。
        if CPS_ALIGN and job.get("_cps_target"):
            try:
                _want = (len(chunk_text) / float(job["_cps_target"])
                         / max(_speed, 0.25))
                _cur = len(audio) / float(sr)
                if _want > 1.0 and _cur > 1.0:
                    _a2 = min(max(_cur / _want, CPS_ATEMPO_LO), CPS_ATEMPO_HI)
                    if abs(_a2 - _atempo) > 1e-3:
                        log.info("任务 %s 第 %d 块 时长对齐: %.1fs -> 目标 "
                                 "%.1fs（%.0f 字 / %.3f 字/s, speed %.2f）"
                                 " atempo %.4f -> %.4f",
                                 jid, ci + 1, _cur, _want, len(chunk_text),
                                 float(job["_cps_target"]), _speed,
                                 _atempo, _a2)
                    _atempo = _a2
            except Exception as _e:
                log.warning("任务 %s 第 %d 块 时长对齐失败（沿用原 atempo）: %r",
                            jid, ci + 1, _e)
        # ---- 首尾静音对齐案例的换气量 --------------------------------------
        # 案例逐段量出来：头 0.04~0.08 s、尾 0.16~0.30 s，两块拼起来间隙
        # 0.25~0.36 s。我们模型自带 0.16~0.20 s 头静音、最后一块几乎没有尾巴，
        # 非最后一块又被补到 0.48 s —— 都比案例长一截。
        # 这里统一裁/补到 60 ms 头 + 220 ms 尾，段间 0.28 s 落进案例区间。
        # 补的量要先除掉 atempo（它会把静音一起拉长）。
        # ---- 韵律 A / E 的 F0 重映射（Praat PSOLA）--------------------------
        # 放在能量扩展之后、首尾静音之前：PSOLA 只动浊音段，不动静音，
        # 所以不会破坏下面的首尾判定；反过来放则会被重合成轻微扰动首尾。
        if PROSO and proso is not None and len(chunk_text) >= PROSO_MIN_CHARS:
            _male = job.get("_is_male")
            # 目标值来自案例库聚合（proso_targets），案例不够时才回落常量
            _g = "male" if _male else "female" if _male is False else "unknown"
            _ta, _te = proso_targets(_g)
            try:
                _segs, _cuts, _hop = prosody_report.sent_windows(audio, sr)
                _ca, _ce = proso.ae_of(audio, sr, is_male=_male, segs=_segs)
                # ---- r25 全局 f0 基线 -------------------------------------
                # ⚠️ 这里用 **pyin 口径**测：PSOLA 施加的是**乘法位移**
                # （g = f × k_med），实际基频就是乘 k_med，与 PitchTier 和
                # pyin 的口径差无关 —— 所以只要 cur / target 同口径即可，
                # 比值天然跨口径不变（陷阱 #4 在这里不成立）。
                _cur_med = None
                if PROSO_MED:
                    try:
                        _mf = prosody_report.f0_of(audio, sr)
                        if _mf is not None and len(_mf) >= 32:
                            _cur_med = float(np.median(_mf))
                    except Exception:
                        _cur_med = None
                _tmed = proso_med_targets(_g)
                audio, _ka, _kt = proso.fit(
                    audio, sr, target_a=_ta, cur_a=_ca,
                    target_e=_te, cur_e=_ce, is_male=_male, segs=_segs,
                    target_med=_tmed, cur_med=_cur_med)
                if _ka > 1.002 or _ka < 0.998 or _kt > 1.002 or _kt < 0.998:
                    log.info("任务 %s 第 %d 块 韵律移植 A %.2f->目标 %.2f "
                             "(k=%.3f)  E %.2f->目标 %.2f (k_tail=%.3f)",
                             jid, ci + 1, _ca or -1, _ta, _ka,
                             _ce or -1, _te, _kt)
                if _cur_med and _tmed:
                    log.info("任务 %s 第 %d 块 f0基线 %.1f Hz -> 目标 %.1f Hz "
                             "(k_med=%.4f, %+.2f 半音)", jid, ci + 1,
                             _cur_med, _tmed,
                             min(max(_tmed / _cur_med, 0.94), 1.16),
                             12 * np.log2(min(max(_tmed / _cur_med, 0.94),
                                              1.16)))
            except Exception as _e:
                log.warning("任务 %s 第 %d 块 韵律移植失败（跳过）: %r",
                            jid, ci + 1, _e)

        # ---- 增益：优先按实测响度反算，量不出来才用静态常数 ----------------
        # ⚠️ 阈值必须按「成品电平」补偿：下面 ffmpeg 还会施加 _gain_db。
        # 不补偿的话，pre-encode 判为「响」的极弱起振段（-40 dB 量级）在成品里
        # 会掉到 -45 dB 以下，于是成品量出来的头静音还是 0.22 s —— 表现就是
        # 「日志说修好了、实测没变」。2026-10-05 排查就是这个坑。
        # 所以这里必须先定下 _gain_db，再去做首尾静音（顺序不能反）。
        _gain_db = OUT_GAIN_DB + float(job.get("_gain_off") or 0.0)
        _tgt_lufs = None
        if LOUD_NORM:
            _male_l = job.get("_is_male")
            _gl = "male" if _male_l else "female" if _male_l is False else "unknown"
            _tgt = loudness_targets(_gl)
            if _tgt is not None:
                _tgt_lufs = _tgt
                _mw = os.path.join(folder_path, f".loud_{jid}_{ci}.wav")
                try:
                    sf.write(_mw, audio, sr)
                    _il, _tp = measure_lufs(_mw)
                    if _il is not None and _il > -70:
                        _gain_db = _tgt - (_il + LOUD_CHAIN_DB)
                        log.info("任务 %s 第 %d 块 响度归一 %s: 实测 %.1f LUFS "
                                 "-> 目标 %.1f, 增益 %+.2f dB（常数原为 %+.2f）",
                                 jid, ci + 1, _gl, _il, _tgt, _gain_db,
                                 OUT_GAIN_DB + float(job.get("_gain_off") or 0.0))
                    else:
                        log.warning("任务 %s 第 %d 块 响度测量失败，回落常数",
                                    jid, ci + 1)
                except Exception as _e:
                    log.warning("任务 %s 第 %d 块 响度归一失败（回落常数）: %r",
                                jid, ci + 1, _e)
                finally:
                    try:
                        os.remove(_mw)
                    except OSError:
                        pass
        if EDGE_HEAD_S > 0 or EDGE_TAIL_S > 0:
            try:
                _h0, _t0, _d0 = sil_edges(audio, sr)
                audio = fit_edges(audio, sr,
                                  EDGE_HEAD_S / max(_atempo, 0.25),
                                  EDGE_TAIL_S / max(_atempo, 0.25),
                                  sil_db=EDGE_SIL_DB - _gain_db)
                _h1, _t1, _d1 = sil_edges(audio, sr)
                log.info("任务 %s 第 %d 块 首尾静音 head %.3f->%.3f "
                         "tail %.3f->%.3f（目标 %.3f/%.3f, atempo=%.3f）"
                         " | 峰值帧 %.1f->%.1f dB, sr=%d, n=%d",
                         jid, ci + 1, _h0 if _h0 is not None else -1,
                         _h1 if _h1 is not None else -1,
                         _t0 if _t0 is not None else -1,
                         _t1 if _t1 is not None else -1,
                         EDGE_HEAD_S / max(_atempo, 0.25),
                         EDGE_TAIL_S / max(_atempo, 0.25), _atempo,
                         _d0 if _d0 is not None else -999,
                         _d1 if _d1 is not None else -999,
                         int(sr), len(audio))
            except Exception as _e:
                log.warning("任务 %s 第 %d 块 首尾静音对齐失败（跳过）: %r",
                            jid, ci + 1, _e)
        mp3_name = f"{base_name}-{ci + 1}.mp3"
        mp3_abs = os.path.join(folder_path, mp3_name)
        job["stage"] = "encoding"
        tmp_wav = os.path.join(folder_path, f".enc_{jid}_{ci}.wav")
        # 瞬态整形用的中间产物（44.1k、已过 EQ 与空气感、**未**过 volume/限幅）
        pre_wav = os.path.join(folder_path, f".pre_{jid}_{ci}.wav")
        try:
            sf.write(tmp_wav, audio, sr)
            # 语速：模型无 speed 接口（见文件头 SPEED_MIN 注释），用 atempo 在
            # 未压缩的 wav 上做一次——只经过一次 mp3 编码，且 atempo 变速不变调。
            # 真正的命令在下面拼（源可能是 tmp_wav 也可能是 pre_wav）
            _af = []
            if abs(_atempo - 1.0) > 1e-3:
                _af.append(f"atempo={_atempo:.4f}")
            # 音色对齐放在变速之后：atempo 只改时长不改频谱，顺序无副作用，
            # 但 EQ 一定要在 volume 之前，否则先衰减再提增益会白损失动态余量。
            if TONE_EQ and TONE_EQ_FILTER:
                # r15 起分性别：男女频谱差异远大于各自与案例的偏差，
                # 共用一套只能取「同向保守值」，两边都补不到位。
                _eq = (TONE_EQ_FILTER_MALE if job.get("_is_male") else
                       TONE_EQ_FILTER_FEMALE
                       if job.get("_is_male") is False else TONE_EQ_FILTER)
                _af.append(_eq)
            # 高频空气感：先重采样到 44.1k（否则 12k 以上放不下），
            # 再生成谐波 + FIR 精确整形。
            if AIR:
                _af.append(f"aresample={OUT_SR}")
                if AIR_EXCITER:
                    _af.append(AIR_EXCITER)
                if AIR_EQ:
                    _af.append(AIR_EQ)
            # ---- 🔴 瞬态整形必须插在 EQ/空气感**之后**、volume/限幅**之前** ---
            # r21 第一版把它放在整条链之前（24k 的裸 wav 上），实测**反效果**：
            #   标杆女峰值 −4.20 → **−5.10**、리노y 偏差 0% → 36.7%。
            # 而离线标定（/tmp/tshaper.py）明明是 +1.8 dB —— 差别就在**位置**：
            #   离线是叠在**成品**上（EQ/空气感/响度归一全走完），
            #   线上却是叠在**进链之前**，后面 EQ 重采样 + aexciter + 响度闭环
            #   把刚抬起来的瞬态又抹回去了。
            # 所以这里把编码拆成两段：先出 44.1k 的 pre_wav，numpy 整形后再
            # 单独跑 volume + 限幅 + mp3。响度闭环只需重跑第二段，pre_wav 复用。
            _pre_needed = (TRANSIENT
                           and len(chunk_text) >= TRANSIENT_MIN_CHARS)
            if _pre_needed:
                _cmd_pre = ["ffmpeg", "-y", "-loglevel", "error",
                            "-i", tmp_wav, "-filter:a", ",".join(_af),
                            "-ar", str(OUT_SR), "-ac", "1", "-f", "wav",
                            pre_wav]
                subprocess.run(_cmd_pre, check=True, timeout=600)
                try:
                    _pa, _psr = sf.read(pre_wav, dtype="float32")
                    if getattr(_pa, "ndim", 1) > 1:
                        _pa = _pa.mean(axis=1)
                    _male = job.get("_is_male")
                    _amt = (TRANSIENT_MALE if _male else
                            TRANSIENT_FEMALE if _male is False else
                            TRANSIENT_DEFAULT)
                    _p0 = transient.peak_dbfs(_pa)
                    _pa = transient.shape(_pa, _psr, _amt)
                    sf.write(pre_wav, _pa, _psr)
                    log.info("任务 %s 第 %d 块 瞬态整形 amount=%.2f "
                             "峰值 %.2f -> %.2f dBFS（%d Hz）",
                             jid, ci + 1, _amt, _p0,
                             transient.peak_dbfs(_pa), _psr)
                except Exception as _e:
                    log.warning("任务 %s 第 %d 块 瞬态整形失败（跳过）: %r",
                                jid, ci + 1, _e)
                _src = pre_wav            # 第二段从整形后的 pre_wav 开始
                _af = []                  # EQ/空气感已跑过，别再跑一遍
            else:
                _src = tmp_wav
            # ---- 尾段：volume → 限幅 →（房间底噪）→ mp3 --------------------
            # 🔴 房间底噪必须插在**响度归一/限幅之后、libmp3lame 之前**：
            #   · 叠在归一之前 → 响度闭环会把多出来的能量整体缩回，等于白铺
            #     （还会把 LUFS 搞偏）；
            #   · 叠在 mp3 之后 → 低电平段的量化杂散已经烤进成品，修不掉
            #     （陷阱 #18：链尾 libmp3lame 会重新定一遍电平）。
            # ffmpeg 滤镜链里没有「我们要的粉红噪声」，所以尾段拆两步：
            #   先出一份已归一/已限幅的 float wav，numpy 叠完再单独编码。
            _af_base = list(_af)          # atempo / EQ / 空气感（瞬态路径下为空）
            _limiter = None
            if LOUD_NORM and LOUD_LIMITER:
                # 分性别：女声 EQ 补偿量大，峰值被推得比男声高 1.6 dB
                _limiter = (LOUD_LIMITER_MALE if job.get("_is_male") else
                            LOUD_LIMITER_FEMALE
                            if job.get("_is_male") is False else LOUD_LIMITER)
            _rt_db = None
            if roomtone.ENABLED:
                _rt_db = roomtone.level_for(
                    "male" if job.get("_is_male") else
                    "female" if job.get("_is_male") is False else "")
            _norm_wav = os.path.join(folder_path, f".norm_{jid}_{ci}.wav")
            _tail = ["-codec:a", "libmp3lame", "-b:a", MP3_BITRATE,
                     "-ar", str(OUT_SR), mp3_abs]

            def _encode_mp3(gain):
                """按给定增益重跑整条尾段。闭环复检复用它，避免两处拼命令。"""
                af = list(_af_base)
                if abs(gain) > 1e-3:
                    af.append(f"volume={gain:.2f}dB")
                if _limiter:
                    af.append(_limiter)
                # ⚠️ 源可能是 tmp_wav（不整形）也可能是 pre_wav（整形后），
                #    所以这里重新拼命令，别复用上面那条写死 tmp_wav 的 _cmd。
                c = ["ffmpeg", "-y", "-loglevel", "error", "-i", _src]
                if af:
                    c += ["-filter:a", ",".join(af)]
                if _rt_db is None:
                    # 没开底噪：一条命令走完，与 r21 及之前**完全一致**
                    subprocess.run(c + _tail, check=True, timeout=600)
                    return
                subprocess.run(c + ["-ac", "1", "-ar", str(OUT_SR),
                                    "-c:a", "pcm_f32le", _norm_wav],
                               check=True, timeout=600)
                _nx, _nsr = sf.read(_norm_wav, dtype="float32")
                if getattr(_nx, "ndim", 1) > 1:
                    _nx = _nx.mean(axis=1)
                _nx = roomtone.apply(_nx, _nsr, _rt_db)
                # 峰值定点放在底噪之后：底噪只有 -56/-50 dBFS，对峰值无影响，
                # 但整个尾段只做一次 wav 往返，顺序上放一起最省事。
                if peakfit.ENABLED and job.get("_peak_target") is not None:
                    _p0 = float(np.max(np.abs(_nx))) if len(_nx) else 0.0
                    _nx = peakfit.fit(_nx, job["_peak_target"])
                    _p1 = float(np.max(np.abs(_nx))) if len(_nx) else 0.0
                    if _p1 < _p0 - 1e-6:
                        log.info("任务 %s 第 %d 块 峰值定点: %.3f -> %.3f "
                                 "（目标 %.3f, %.2f dBFS）",
                                 jid, ci + 1, _p0, _p1,
                                 10.0 ** (job["_peak_target"] / 20.0),
                                 job["_peak_target"])
                sf.write(_norm_wav, _nx, _nsr, subtype="FLOAT")
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                                "-i", _norm_wav] + _tail,
                               check=True, timeout=600)

            _encode_mp3(_gain_db)

            # ---- 响度闭环复检：编完实测成品，偏了就用残差重编一次 --------
            # LOUD_CHAIN_DB 只是初值，链路一改（换 EQ、换空气感参数、换音色）
            # 它就不准。与其每次手工重标常数，不如编完再量一次成品 mp3：
            # 偏差超过 0.5 dB 就把残差补进增益重编。多一次编码（几秒），
            # 换来「改链路不用重标常数」。最多重编一次，不兜圈子。
            # （底噪只有 -56/-50 dBFS，对 LUFS 的贡献 < 0.003 dB，不会扰动闭环）
            if _tgt_lufs is not None:
                try:
                    _il2, _tp2 = measure_lufs(mp3_abs)
                    if _il2 is not None and abs(_il2 - _tgt_lufs) > LOUD_TOL:
                        _fix = _tgt_lufs - _il2
                        _gain_db += _fix
                        _encode_mp3(_gain_db)
                        log.info("任务 %s 第 %d 块 响度闭环复检: 成品 %.2f "
                                 "LUFS（目标 %.2f）→ 补 %+.2f dB 重编",
                                 jid, ci + 1, _il2, _tgt_lufs, _fix)
                    elif _il2 is not None:
                        log.info("任务 %s 第 %d 块 响度复检 OK: %.2f LUFS"
                                 "（目标 %.2f, 峰值 %s dBFS）",
                                 jid, ci + 1, _il2, _tgt_lufs, _tp2)
                except Exception as _e:
                    log.warning("任务 %s 第 %d 块 响度复检失败（保留首编）: %r",
                                jid, ci + 1, _e)
        except Exception as e:
            log.warning("mp3 转码失败 %s: %s", mp3_abs, e)
        finally:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass
            try:
                os.remove(pre_wav)
            except OSError:
                pass
            try:
                os.remove(_norm_wav)
            except (OSError, NameError, UnboundLocalError):
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
    # ---- 音色语速（cps）自校准 -------------------------------------------
    # 分块上限现在是「时长驱动」的（对齐 ElevenLabs 约 250 秒一块），
    # 字数→时长全靠 cps 换算。而 cps 每个音色都不一样（实测男 8.61 / 女 7.32，
    # 差 18%），写死一个数必然有一个性别切歪。
    # 所以每跑完一单就把实测值写回 voices.json（滑动平均，抗单篇波动）。
    #
    # ⚠️ r21 加了**死区**：cps 一动，同一篇文案的分块数就可能在 2 段 / 3 段之间
    #    跳（分块是按「预估时长 / 250 秒」切的，cps 换算出来的字数上限跟着变）。
    #    员工看到「上次 2 段、这次 3 段」会以为出了 bug。所以：
    #      · 相对变化 < 3% 就**不写回**（收敛后彻底静止，分块数稳定）
    #      · 真的要更新也放慢（0.8/0.2），别被单篇波动带跑
    _vc = job.get("voice")
    _new_cps = agg.get("cps") or 0.0
    if _vc and 4.0 < _new_cps < 14.0 and (agg.get("audio_seconds") or 0) > 60:
        _v = voices.get(_vc)
        if _v is not None:
            _old = float(_v.get("cps") or 0.0)
            _skip = False
            if _old and abs(_new_cps - _old) / _old < 0.03:
                _skip = True          # 死区内：不动，保证分块数稳定
            if not _skip:
                _v["cps"] = round(_new_cps if not _old
                                  else 0.8 * _old + 0.2 * _new_cps, 3)
                save_voices(voices)
                log.info("任务 %s 音色 %s 语速自校准 cps %.2f -> %.2f",
                         jid, _vc, _old or _new_cps, _v["cps"])
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


# ---- 引擎真实可用性（给「情感」下拉用）----------------------------------
# 为什么要这个：CosyVoice3 没部署时，选「激昂/亲切/严肃」会走到
# vram.switch_to("cosyvoice3") 失败、再回退 Qwen —— 任务**不报错**，
# 员工拿到的是和「平稳」逐字节一样的成品，还白等了一轮排队。
# 2026-10-06 实测三项 emotion 全部 fallback=cosyvoice3。
# 与其事后静默回退，不如在下拉里就标出来。
_ENGINE_AVAIL = {"ts": 0.0, "cosyvoice3": False}
_ENGINE_AVAIL_TTL = 60.0


def engine_available(name):
    """某引擎现在能不能真的用。带 60 秒缓存，避免每次 /api/meta 都去连端口。"""
    if name == "qwen":
        return True
    if name != "cosyvoice3":
        return False
    now = time.time()
    if now - _ENGINE_AVAIL["ts"] < _ENGINE_AVAIL_TTL:
        return _ENGINE_AVAIL["cosyvoice3"]
    ok = False
    try:
        ok = bool(cosy_client.health(timeout=3.0)[0])
    except Exception:
        ok = False
    _ENGINE_AVAIL["ts"] = now
    _ENGINE_AVAIL["cosyvoice3"] = ok
    return ok


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
        # 员工只需要选「情感」；引擎/指令/参数由后端自动分流。
        # ⚠️ available=false 的项前端会置灰：它的引擎当前连不上，选了也只是
        #    静默回退 Qwen（出和「平稳」一样的成品）。
        "emotions": [
            dict({"key": k, "engine": v["engine"]},
                 available=engine_available(v["engine"]))
            for k, v in EMOTION_PRESETS.items()
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
async def retry_job(request: Request, jid: str, new_seed: int = Form(1)):
    """重跑一个任务。

    ⚠️ new_seed=1（默认）会换一个随机种子。原来无条件复制旧 seed，
    于是「重试」出来的是**逐字节相同**的音频，等于没有这个按钮 ——
    而模型本身有随机性缺陷（官方 #239/#290），遇到某段别扭时，
    正确对策恰恰是**换种子再抽一次**。所以默认改成换种子。
    new_seed=0 保留原种子（想精确复现某一版时用）。
    """
    u = require(request)
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "任务不存在")
    if u["role"] != "admin" and j.get("owner") != u["username"]:
        raise HTTPException(403, "无权操作")
    if not j.get("text"):
        raise HTTPException(400, "原文案已丢失，无法重试，请重新提交")
    new = dict(j)
    if new_seed:
        new["seed"] = random.randint(1, 2 ** 31 - 1)
        new["retried_from"] = jid
        log.info("任务 %s 换种子重试: %s -> %s", jid, j.get("seed"), new["seed"])
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


# ============================ 案例素材库 ============================
#
# 存在的目的：把「对标 ElevenLabs 的 1:1 案例」沉淀成可复用的数据集，
# 而不是每次靠一两条音频拍脑袋。每份案例 = 原文 + 原声 + 自动算出的客观指标
# （时长 / 字数 / 语速 / CER / 韵律六维），攒够样本后才能横向找规律。
#
# 存储：/mnt/tts/data/cases/{cid}/
#   meta 见 cases.json      seg{N}.wav  24k 单声道归一化音频
#   seg{N}.txt  该段文本     source{N}{ext}  原始上传留档（重分析不用再传）
#   merged.wav  拼接整段（韵律分析按整段算，多段案例本来就是连着播的）

CASE_GENDERS = ("male", "female", "unknown")
# 来源平台：案例是「标准答案」，必须能追溯它是谁生成的，
# 否则后面做横向规律分析时，不同平台的样本会混在一起互相污染结论。
CASE_SOURCES = ("elevenlabs", "qwen", "cosyvoice3", "other", "unknown")
# 🔴 r19（2026-10-07）：默认**不再只认「标杆」tag**，改为同性别 + source 白名单全收。
#    ⚠️ 推翻 r15 的旧注释：r15 说「yohan 3 例是另一个音色」——那是凭 LUFS 差 0.7 dB
#    猜的，**从没做过说话人核验**。r19 用 CosyVoice3 自带 campplus 提嵌入实测：
#        标杆男 ↔ yohan 三例 = 0.935 / 0.943 / 0.924（同人阈值 0.75）
#        标杆女 ↔ 四个男声  = 0.309~0.328
#    → **4 个男声是同一个人**。所以把它们算进来不是「稀释」，而是同人多样本、
#      估计更稳（响度中位数 −25.0、停顿样本 162 → 688，后者本就是原设计的意图）。
#    → 也意味着这 4 例可以混在一起做微调（男声已有 ~34 分钟同人数据）。
#    想回到旧行为：设 TTS_CASE_BENCHMARK_TAG=标杆。
# ⚠️ 真正的排异依据是 **source 白名单 + 说话人身份**，不是 tag。
#    tag 是人工标注，员工新增案例很容易忘打，结果就是「加了案例却不生效」。
CASE_BENCHMARK_TAG = os.environ.get("TTS_CASE_BENCHMARK_TAG", "").strip()
# 韵律六维（pyin）很慢，几分钟音频要 1~2 分钟，默认后台跑，不卡上传请求。
# 设 TTS_CASE_PROSODY=0 可整体关掉。
CASE_PROSODY = os.environ.get("TTS_CASE_PROSODY", "1").strip().lower() not in (
    "0", "false", "no")


def _snap_sentence_end(t, pos, window=400):
    """把切点吸附到附近最近的句末标点之后，避免切在句子中间。"""
    lo, hi = max(0, pos - window), min(len(t), pos + window)
    best, bestd = pos, None
    for m in re.finditer(r"[.!?。！？…]", t):
        p = m.end()
        if lo <= p <= hi:
            d = abs(p - pos)
            if bestd is None or d < bestd:
                best, bestd = p, d
    return best


def _natkey(name):
    """自然排序键：让 2.mp3 排在 10.mp3 前面。

    ⚠️ 案例动辄十几段，命名常是裸数字（1.mp3 … 12.mp3）。
    直接用 sorted(key=filename) 是**字符串排序**，"10.mp3" < "2.mp3"，
    音频与文案会整组错位——不报错、不警告，只是静默把第 2 段音频配到
    第 10 段的文案上，后面所有「拆解规律」的结论全部作废。
    """
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", name or "")]


def _align_texts(texts, durs):
    """把文案对齐到各音频段。返回 ([(文本, (char_start, char_end))...], 方式)。

    为什么必须对齐，而不是「存整篇文案 + 每段用 ASR 补全」：
    ElevenLabs 的玩法是整篇文案丢进去，平台按 ~2600 字上限自动拆块、
    逐块生成。所以「1 篇文案 + N 段音频」是**平台分块后的产物**，
    不是素材没配对。文案哪一段对应哪段音频必须对齐出来，否则
      ① 拿不到准确文本（ASR 噪声 ~22%）
      ② 没法逆向研究平台的拆分边界策略——而这正是要拆解的东西
    对齐后每段都记 char_start/char_end，能回溯到整篇文案的位置。

    三种方式，按可靠性排序：
      per_file        上传时就按段给好了文本，直接用
      split_chunks    用与生成端同一口径的分块器（2600 上限、句末断开）拆，块数对得上
      duration_ratio  块数对不上，按音频时长占比切到最近句末
    """
    if not texts or not durs:
        return [], "none"
    # ① 逐段一一对应
    if len(texts) == len(durs):
        return [(t, None) for t in texts], "per_file"
    if len(texts) != 1:
        return [], "none"
    t = (texts[0] or "").strip()
    if not t:
        return [], "none"

    # ② 按时长占比切到最近句末。
    #    ⚠️ 不要拿 split_chunks（2600 字上限贪心）的结果当对齐依据：
    #    它的块长比与音频时长比常常对不上——实测 4045 字案例，
    #    split_chunks 切成约 64%/36%，而两段音频是 54%/46%。
    #    硬用字数块去套音频，文本范围根本不是音频念的那段，CER 会爆掉，
    #    还会误判成「文本与音频对不上」。
    #    时长占比才直接反映「这段音频念了多少字」。
    #    split_chunks 的预测边界另存进 chunk_pred，用来研究平台到底怎么切。
    total = float(sum(durs)) or 1.0
    t_len = len(t)
    ends, acc = [], 0
    for x in durs[:-1]:
        acc += x
        ends.append(_snap_sentence_end(t, int(round(t_len * acc / total))))
    out, prev = [], 0
    for e in ends + [t_len]:
        e = max(e, prev)
        out.append((t[prev:e].strip(), (prev, e)))
        prev = e
    return out, "duration_ratio"


def _case_dir(cid):
    return os.path.join(CASE_DIR, cid)


def measure_lufs(path):
    """量积分响度（LUFS）。走 ffmpeg ebur128，不额外引依赖。

    返回 (integrated_lufs, true_peak_dbfs)，失败返回 (None, None)。
    """
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
             "-filter:a", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True, timeout=900)
    except Exception:
        return None, None
    tail = (r.stderr or "")[-4000:]
    # ebur128 会先滚动输出、最后给一段 Summary，取最后一段才准
    seg = tail[tail.rfind("Summary"):] if "Summary" in tail else tail
    il, tp = None, None
    for ln in seg.split("\n"):
        if "I:" in ln and "LUFS" in ln:
            try:
                il = float(ln.split("I:")[1].split("LUFS")[0])
            except Exception:
                pass
        if "Peak:" in ln and "dBFS" in ln:
            try:
                tp = float(ln.split("Peak:")[1].split("dBFS")[0])
            except Exception:
                pass
    return il, tp


def loudness_targets(gender):
    """响度目标值：从案例库按性别取中位数，取不到回落到常量。

    与 proso_targets 同一套约定：只认 source 在白名单里的案例（默认
    elevenlabs），我们自己的产物不能当自己的尺子。
    """
    if not LOUD_NORM:
        return None
    vals = []
    for c in _bench_cases(gender):     # 共用标杆筛选，见 CASE_BENCHMARK_TAG
        v = c.get("lufs")
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            pass
    if not vals:
        if gender == "male":
            return LOUD_TARGET_MALE
        if gender == "female":
            return LOUD_TARGET_FEMALE
        return LOUD_TARGET_DEFAULT
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def proso_med_targets(gender):
    """r25：全局 f0 基线的目标值（Hz，pyin 口径）。

    数据来源 `tools/caseslow.py` 的 24 条普查结果（/mnt/tts/diag/caseslow.json），
    取同性别句级 f0 中位的**中位数**：男 196.2、女 280.2。
    文件不在就回落常量（数值就是普查出来的那两个）。
    """
    try:
        p = os.environ.get("TTS_CASESLOW_JSON",
                           "/mnt/tts/diag/caseslow.json")
        if os.path.exists(p):
            rows = json.load(open(p, encoding="utf-8"))
            v = [float(r["f0_med"]) for r in rows
                 if r.get("gender") == gender
                 and r.get("f0_med") is not None]
            if v:
                v.sort()
                n = len(v)
                return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0
    except Exception:
        pass
    if gender == "male":
        return PROSO_MED_FALLBACK_MALE
    if gender == "female":
        return PROSO_MED_FALLBACK_FEMALE
    return PROSO_MED_FALLBACK_DEFAULT


def proso_targets(gender):
    """PROSO 的目标 A/E：从案例库按性别取中位数，取不到才回落到常量。

    「多上传案例 → 规律更稳」就靠这里闭环：每份案例在后台拆解时会顺带算出
    Praat 口径的 A/E，这里按时取中位数当该性别的目标值。

    ⚠️ 只取 case["proso"]（Praat 口径，由 proso.ae_of 算），
       不取 case["prosody"]["A"]（pyin 口径）—— 后者换算到 Praat 会失效。
    """
    def fallback():
        ca, ce = CAL.get(gender, (PROSO_CAL_DEFAULT, PROSO_CAL_DEFAULT))
        if gender == "male":
            return PROSO_A_MALE * ca, PROSO_E_MALE * ce
        if gender == "female":
            return PROSO_A_FEMALE * ca, PROSO_E_FEMALE * ce
        return PROSO_A_DEFAULT * ca, PROSO_E_DEFAULT * ce

    if not PROSO_FROM_CASES:
        return fallback()
    vals = []
    for c in _bench_cases(gender):     # 共用标杆筛选，见 CASE_BENCHMARK_TAG
        p = c.get("proso") or {}
        if p.get("A") is None or p.get("E") is None:
            continue
        vals.append((float(p["A"]), float(p["E"])))
    if not vals:
        return fallback()
    vals.sort()
    n = len(vals)
    mid = n // 2
    if n % 2:
        a, e = vals[mid]
    else:
        a = (vals[mid - 1][0] + vals[mid][0]) / 2.0
        e = (vals[mid - 1][1] + vals[mid][1]) / 2.0
    # 最后乘校准系数：Praat 口径达标不等于 pyin 口径达标，详见 CAL 的注释。
    ca, ce = CAL.get(gender, (PROSO_CAL_DEFAULT, PROSO_CAL_DEFAULT))
    return a * ca, e * ce


def _bench_cases(gender):
    """标杆案例：优先只认打了「标杆」tag 的，否则回落同性别全部。

    ⚠️ 三个目标值函数（pause / proso / loudness）必须共用这一套筛选，
       否则「多上传案例 → 规律更稳」会变成「多上传案例 → 标杆被稀释」。
    """
    base = [c for c in cases.values()
            if c.get("gender") == gender
            and (not PROSO_CASE_SOURCE
                 or c.get("source") in PROSO_CASE_SOURCE)]
    if not CASE_BENCHMARK_TAG:
        return base
    tagged = [c for c in base
              if CASE_BENCHMARK_TAG in (c.get("tags") or "")]
    return tagged or base


def pause_targets(gender):
    """停顿分布的目标样本：把该性别所有合规案例的停顿样本拼起来。

    与 proso_targets / loudness_targets 同一套白名单约定。拼多份案例是为了
    样本量够——单份 100~200 个停顿，分 8 桶做分位映射时边界抖动大。

    ⚠️ 借的是**形状**不是均值：reshuffle 拿到后会整体缩放到它自己的 mu 回归
       值上（mu 由句均时长决定，我们句子比案例碎，均值不能照抄）。
    """
    if not PAUSE_FROM_CASES:
        return None
    out = []
    for c in _bench_cases(gender):     # 共用标杆筛选，见 CASE_BENCHMARK_TAG
        ps = c.get("pause_samples")
        if not ps:
            continue
        try:
            # 🔴 防御：只收 0.1~5 秒。出过一次事故——某个案例的
            #    pause_samples 因换算 bug 变成几百秒，拼进来后
            #    reshuffle 的目标 mu 变成 172 秒，成品被拉成 4.7 小时。
            #    分布形状本来就该落在「人能做出的停顿」范围内，
            #    超出的必然是脏数据，直接丢掉不要让整单报废。
            out.extend(v for v in (float(x) for x in ps)
                       if 0.1 <= v <= 5.0)
        except (TypeError, ValueError):
            pass
    return out or None


def pause_density(gender):
    """案例的「停顿条数 / 字数」密度（r22b）。

    为什么按字数而不是按时长：生成时文本是已知的，时长正是我们要求解的量。
    用时长做分母会变成循环论证（想要多长 → 决定几个停顿 → 决定多长）。

    口径与 `prosody_report.measure` 同源（案例入库时算的 `prosody.pauses`），
    和验收脚本的 `pauses` 是同一个数。
    """
    tot_p = tot_c = 0
    for c in _bench_cases(gender):
        p = (c.get("prosody") or {}).get("pauses")
        n = c.get("total_chars") or 0
        if p and n:
            tot_p += int(p)
            tot_c += int(n)
    return (tot_p / float(tot_c)) if tot_c else None


def peak_target(gender):
    """成品真峰值目标：按性别取标杆案例的中位数（r23）。

    为什么用中位数：评分是 |ΔdB| 的 L1 损失，中位数正是 L1 的最优估计。
    案例峰值分布很散（男 −0.9 ~ −3.1），众数/均值都不如中位数稳。
    """
    vals = []
    for c in _bench_cases(gender):
        v = c.get("peak_dbfs")
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            pass
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def cps_target(gender):
    """案例语速（字/秒）= Σ字数 / Σ时长（r22b）。

    用加权平均而不是各案例 cps 的中位数：长篇案例的语速更稳定，
    按字数加权后抗抖动更好（实测中位数比加权平均差 0.4 个点的 dur 失分）。
    """
    tot_c = 0.0
    tot_d = 0.0
    for c in _bench_cases(gender):
        n = c.get("total_chars") or 0
        d = c.get("total_dur") or 0
        if n and d:
            tot_c += float(n)
            tot_d += float(d)
    return (tot_c / tot_d) if tot_d else None


def _analyze_case(cid):
    """后台算韵律六维并回写 cases.json。失败只记日志，不影响案例本身。"""
    try:
        import prosody_report
        c = cases.get(cid)
        if not c:
            return
        merged = os.path.join(_case_dir(cid), "merged.wav")
        if not os.path.isfile(merged):
            return
        m = prosody_report.measure(merged, name=cid)
        c["prosody"] = {k: m.get(k) for k in
                        ("A", "B", "C", "D", "E", "F", "segs", "pauses")}
        c["prosody_done"] = True
        # Praat 口径的 A/E：PROSO 拿它当目标，必须同源。
        # 算一次存下来，生成时直接取，避免每次生成都跑一遍 Praat。
        if proso is not None:
            try:
                import soundfile as sf
                _x, _sr = sf.read(merged, dtype="float32")
                if getattr(_x, "ndim", 1) > 1:
                    _x = _x.mean(axis=1)
                _segs, _cuts, _hop = prosody_report.sent_windows(_x, _sr)
                _a, _e = proso.ae_of(_x, _sr,
                                     is_male=(c.get("gender") == "male"),
                                     segs=_segs)
                c["proso"] = {"A": round(float(_a), 3),
                              "E": round(float(_e), 3)}
                log.info("案例 %s PROSO 口径 A/E = %.3f / %.3f", cid, _a, _e)
            except Exception as e:
                    log.warning("案例 %s PROSO 口径分析失败: %s", cid, e)
        # 停顿样本：r15 起 reshuffle 直接借案例的经验分布（分布**形状**，
        # 不是均值）。只留 ≥0.28s 的——与 prosody_report 算 F 的口径同源，
        # 否则形状对不上。上限 400 个够了（再多只是同一分布的重复采样）。
        try:
            # ⚠️🔴 不要再乘 _hop：cuts 里已经是**样本索引**（sent_windows 内部
            #    存的就是 i*hop）。2026-10-06 这里写成 (j-i)*hop/sr，24k 下
            #    整体放大 480 倍，员工上传的新案例 pause_samples 变成
            #    max=470 秒，reshuffle 的目标 mu 跟着变成 172 秒，
            #    成品被拉成 **4.7 小时**（232.90s -> 16854.56s）。
            _runs = [(_j - _i) / float(_sr) for _i, _j in _cuts]
            _runs = sorted(r for r in _runs if r >= 0.28)
            if len(_runs) >= 8:
                c["pause_samples"] = [round(float(r), 4) for r in _runs[:400]]
                log.info("案例 %s 停顿样本 %d 个 均值 %.3fs 中位 %.3fs "
                         "最长 %.3fs", cid, len(_runs),
                         float(np.mean(_runs)), float(np.median(_runs)),
                         float(_runs[-1]))
        except Exception as e:
            log.warning("案例 %s 停顿样本提取失败: %s", cid, e)
        # 响度：成品要对标的就是这个数（loudness_targets 按时取中位数）。
        _il, _tp = measure_lufs(merged)
        if _il is not None:
            c["lufs"] = round(float(_il), 2)
            c["peak_dbfs"] = round(float(_tp), 2) if _tp is not None else None
            log.info("案例 %s 响度 = %.2f LUFS（峰值 %s dBFS）",
                     cid, _il, c["peak_dbfs"])
        save_cases(cases)
        log.info("案例 %s 韵律分析完成: %s", cid, c["prosody"])
    except Exception as e:
        log.warning("案例 %s 韵律分析失败: %s", cid, e)


@app.post("/api/cases")
async def upload_case(request: Request,
                      name: str = Form(...),
                      gender: str = Form("unknown"),
                      source: str = Form("unknown"),
                      tags: str = Form(""),
                      note: str = Form(""),
                      files: List[UploadFile] = File(...),
                      text_files: Optional[List[UploadFile]] = File(None)):
    """上传一份案例（可多段音频 + 对应文本）。

    文案怎么和音频对上（三种都支持）：
      · 传 N 个 txt  → 与音频按文件名**自然序**一一对应（1 在 2 前，2 在 10 前）
      · 传 1 个整篇  → 按各段音频的时长占比切到句末分给各段（ElevenLabs 的
                       产物本来就是「一篇文案被平台切成 N 块」，推荐这种）
      · 不传        → 每段自动 ASR 转写补上并标注 text_src=asr，方便人工校对

    source / tags 是给后续「拆解它是怎么生成的」用的：
    案例多来自 ElevenLabs，能标出 voice / model / speed 这类生成参数，
    后面按参数分组找规律才有依据（否则只能笼统说「案例比较好」）。
    """
    u = require(request)
    gender = (gender or "unknown").strip().lower()
    if gender not in CASE_GENDERS:
        raise HTTPException(400, f"gender 需在 {list(CASE_GENDERS)} 之内")
    source = (source or "unknown").strip().lower()
    if source not in CASE_SOURCES:
        raise HTTPException(400, f"source 需在 {list(CASE_SOURCES)} 之内")
    safe = "".join(ch for ch in (name or "") if ch not in '/\\:*?"<>|').strip()
    if not safe:
        raise HTTPException(400, "案例名无效")
    if not files:
        raise HTTPException(400, "至少上传一个音频文件")

    texts = []
    if text_files:
        for tf in sorted([f for f in text_files if f.filename],
                         key=lambda f: _natkey(f.filename)):
            raw = await tf.read()
            texts.append(raw.decode("utf-8", errors="ignore").strip())

    cid = uuid.uuid4().hex[:12]
    d = _case_dir(cid)
    os.makedirs(d, exist_ok=True)

    order = sorted([f for f in files if f.filename],
                   key=lambda f: _natkey(f.filename))

    # ---------- 第一遍：先处理音频，拿到每段时长 ----------
    # 必须先拿时长：文本对齐要按时长比例分配，而时长只有解码后才知道。
    audio_meta = []
    for i, up in enumerate(order):
        raw = await up.read()
        ext = os.path.splitext(up.filename or "")[1].lower() or ".mp3"
        src = os.path.join(d, f"source{i}{ext}")
        with open(src, "wb") as fh:
            fh.write(raw)
        try:
            data, sr = sf.read(src, dtype="float32")
        except Exception as e:
            # 单个文件坏了不该让整单失败：留档后跳过，其它段照常入库
            log.warning("案例 %s 第 %d 个音频无法读取 %s: %s", cid, i, up.filename, e)
            continue
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        # 统一 24k 单声道：与内部处理口径一致，后续分析不用再猜采样率
        if int(sr) != 24000:
            import librosa
            data = librosa.resample(data.astype(np.float64), orig_sr=sr,
                                    target_sr=24000).astype(np.float32)
            sr = 24000
        wav = os.path.join(d, f"seg{i}.wav")
        sf.write(wav, data, sr)
        audio_meta.append({"idx": i, "src": up.filename,
                           "audio": f"seg{i}.wav",
                           "dur": round(len(data) / float(sr), 3)})

    if not audio_meta:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "没有任何可用的音频文件")

    # ---------- 第二遍：把文案对齐到各音频段 ----------
    aligned, align_method = _align_texts(texts,
                                         [a["dur"] for a in audio_meta])
    # 只要上传的是「一篇整文案 + N 段音频」，就把整篇原文存下来。
    # 后续拆解要拿整篇去对段的字符区间；只留 seg 文本的话，对齐时被 strip 掉的
    # 空白/换行就永久丢了，也没法校验「两段拼回来是否等于原文」。
    full_text = texts[0].strip() if len(texts) == 1 else ""
    if full_text:
        with open(os.path.join(d, "full.txt"), "w", encoding="utf-8") as fh:
            fh.write(full_text)

    segs = []
    for i, am in enumerate(audio_meta):
        wav = os.path.join(d, am["audio"])
        dur = am["dur"]
        txt = aligned[i][0] if i < len(aligned) else ""
        rng = aligned[i][1] if i < len(aligned) else None
        text_src = "user" if txt else ""
        if not txt:
            try:
                txt = asr_check.transcribe(wav, language="Korean")
                text_src = "asr"
            except Exception as e:
                log.warning("案例 %s 第 %d 段 ASR 转写失败: %s", cid, i, e)
        if txt:
            with open(os.path.join(d, f"seg{i}.txt"), "w", encoding="utf-8") as fh:
                fh.write(txt)

        # CER 在这里的含义是「文本与音频对不对得上」，不是发音好坏：
        # 案例原声自己就有 ~0.2227 的噪声底，明显高于它才说明文本对不上。
        # ⚠️ 只对 text_src=user 的段算：文本若来自 ASR，等于拿转写比转写，
        #    结果必然是 0，反而会伪装成「文本完美匹配」误导后续分析
        #    （与「CER 恰好为 0 = 回检坏了」是同一个陷阱）。
        cer = None
        if txt and text_src == "user":
            try:
                cer = asr_check.check_chunk(wav, txt, "Korean").get("cer")
            except Exception:
                cer = None

        segs.append({"idx": i, "src": am["src"], "audio": am["audio"],
                     "text": f"seg{i}.txt" if txt else "",
                     "dur": dur, "chars": len(txt), "text_src": text_src,
                     "char_start": (rng or (None, None))[0],
                     "char_end": (rng or (None, None))[1],
                     "cps": round(len(txt) / dur, 3) if dur and txt else None,
                     "cer": cer})

    if not segs:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "没有任何可用的音频文件")

    try:
        parts = [sf.read(os.path.join(d, s["audio"]), dtype="float32")[0]
                 for s in segs]
        allx = np.concatenate(parts) if len(parts) > 1 else parts[0]
        sf.write(os.path.join(d, "merged.wav"), allx, 24000)
    except Exception as e:
        log.warning("案例 %s 拼接 merged 失败: %s", cid, e)

    # 顺带记下「按我们自己的口径会怎么切」，供后续对比平台的实际切法。
    # 这两者的差异本身就是「平台分块策略」的直接证据。
    chunk_pred = []
    _t = (texts[0] if len(texts) == 1 else "").strip()
    if _t and len(audio_meta) > 1:
        import pipeline
        pos = 0
        for c in pipeline.split_chunks(_t):
            s = _t.find(c[:60], pos)
            if s < 0:
                s = pos
            chunk_pred.append([s, s + len(c)])
            pos = s + len(c)

    total_dur = round(sum(s["dur"] for s in segs), 3)
    total_chars = sum(s["chars"] for s in segs)
    rec = {"id": cid, "name": safe, "gender": gender, "source": source,
           "tags": (tags or "").strip(), "note": (note or "").strip(),
           "has_full_text": bool(full_text), "align_method": align_method,
           "chunk_pred": chunk_pred,
           "owner": u["username"], "created_at": time.time(),
           "segments": segs, "total_dur": total_dur, "total_chars": total_chars,
           "cps": round(total_chars / total_dur, 3) if total_dur else None,
           "prosody": None, "prosody_done": False,
           "fb_count": 0, "fb_tags": {}, "fb_rating": None}
    cases[cid] = rec
    save_cases(cases)
    log.info("上传案例 %s(%s) name=%s 段数=%d 时长=%.1fs 字数=%d",
             cid, gender, safe, len(segs), total_dur, total_chars)

    if CASE_PROSODY:
        threading.Thread(target=_analyze_case, args=(cid,), daemon=True).start()
    return rec


@app.get("/api/cases")
async def list_cases(request: Request, gender: str = "", source: str = ""):
    require(request)
    items = list(cases.values())
    if gender.strip():
        items = [c for c in items if c.get("gender") == gender.strip().lower()]
    if source.strip():
        items = [c for c in items if c.get("source") == source.strip().lower()]
    items.sort(key=lambda c: c.get("created_at") or 0, reverse=True)
    # 段内文本可能很长，列表页只给统计值，详情再 GET /api/cases/{id}
    out = []
    for c in items:
        d = {k: v for k, v in c.items() if k != "segments"}
        d["seg_count"] = len(c.get("segments") or [])
        out.append(d)
    return out


@app.get("/api/cases/{cid}")
async def get_case(request: Request, cid: str):
    require(request)
    c = cases.get(cid)
    if not c:
        raise HTTPException(404, "案例不存在")
    return c


@app.get("/api/cases/{cid}/audio")
async def case_audio(request: Request, cid: str, seg: Optional[int] = None):
    """下载案例音频。不带 seg = 拼接整段；带 seg = 指定片段。"""
    require(request)
    c = cases.get(cid)
    if not c:
        raise HTTPException(404, "案例不存在")
    if seg is None:
        p = os.path.join(_case_dir(cid), "merged.wav")
    else:
        s = next((x for x in c.get("segments", []) if x.get("idx") == seg), None)
        if not s:
            raise HTTPException(404, "片段不存在")
        p = os.path.join(_case_dir(cid), s.get("audio", ""))
    if not os.path.isfile(p):
        raise HTTPException(404, "音频文件不存在")
    return FileResponse(p, media_type="audio/wav",
                        filename=f"{c.get('name', cid)}-{cid}.wav")


@app.get("/api/cases/{cid}/text")
async def case_text(request: Request, cid: str):
    """取案例文本。有整篇文案（full.txt）就优先给整篇，否则拼各段。"""
    require(request)
    c = cases.get(cid)
    if not c:
        raise HTTPException(404, "案例不存在")
    fp = os.path.join(_case_dir(cid), "full.txt")
    if os.path.isfile(fp):
        with open(fp, encoding="utf-8") as fh:
            return Response(content=fh.read(),
                            media_type="text/plain; charset=utf-8")
    parts = []
    for s in c.get("segments", []):
        p = os.path.join(_case_dir(cid), s.get("text", ""))
        if s.get("text") and os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                parts.append(fh.read())
    return Response(content="\n\n".join(parts),
                    media_type="text/plain; charset=utf-8")


# ======================== 案例反馈（员工试听后打分） ========================
# 为什么要有它：指标（韵律六维、CER）只能证明"差多少"，证明不了"差在哪、
# 值不值得修"。员工每天在听，他们一句话比我们跑十次实验准 ——
# 但"好听"这种反馈没法直接进代码，所以把它拆成**可勾选的维度**，
# 攒够之后按标签排序就知道该修哪一项。
#
# 存储：/mnt/tts/data/cases/{cid}/feedback.json（一条一条原始记录，不丢），
#       汇总（人数 / 各标签票数 / 平均分）回写 cases.json，列表页直接显示。
CASE_FEEDBACK_TAGS = (
    "pace",        # 语速节奏
    "intonation",  # 语调起伏
    "emotion",     # 情绪感染力
    "clarity",     # 咬字清晰度
    "pausing",     # 停顿换气
    "timbre",      # 音色质感
    "natural",     # 整体自然度
    "other",       # 其它（用备注说明）
)


def _fb_path(cid):
    return os.path.join(_case_dir(cid), "feedback.json")


def _load_fb(cid):
    p = _fb_path(cid)
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as fh:
                rows = json.load(fh)
            if isinstance(rows, list):
                return rows
        except Exception as e:
            log.warning("案例 %s 反馈读取失败: %s", cid, e)
    return []


def _save_fb(cid, rows):
    d = _case_dir(cid)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, "feedback.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, _fb_path(cid))     # 原子替换，避免并发写坏


def _fb_summarize(cid):
    """把原始反馈汇总回 cases.json（人数/票数/均分）。"""
    rows = _load_fb(cid)
    tags, rs = {}, []
    for r in rows:
        for t in (r.get("tags") or []):
            tags[t] = tags.get(t, 0) + 1
        if isinstance(r.get("rating"), (int, float)):
            rs.append(float(r["rating"]))
    c = cases[cid]
    c["fb_count"] = len(rows)
    c["fb_tags"] = tags
    c["fb_rating"] = round(sum(rs) / len(rs), 2) if rs else None


@app.post("/api/cases/{cid}/feedback")
async def add_case_feedback(request: Request, cid: str):
    """员工听完标杆案例后提交「喜欢哪些地方」。

    同一人对同一案例只保留最新一条（改主意了能覆盖），
    但**每条原始记录都留着**（未删），方便后面按人看变化。
    """
    u = require(request)
    if cid not in cases:
        raise HTTPException(404, "案例不存在")
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    tags = [t for t in (payload.get("tags") or []) if t in CASE_FEEDBACK_TAGS]
    rating = payload.get("rating")
    rating = int(rating) if isinstance(rating, (int, float)) and \
        1 <= float(rating) <= 5 else None
    note = (payload.get("note") or "").strip()[:500]
    seg = payload.get("seg")
    seg = int(seg) if isinstance(seg, (int, float)) else None
    if not tags and rating is None and not note:
        raise HTTPException(400, "至少选一个标签、打个分或写点备注")

    rows = _load_fb(cid)
    row = {"user": u["username"], "role": u.get("role"),
           "tags": tags, "rating": rating, "note": note, "seg": seg,
           "at": time.time()}
    rows.append(row)
    _save_fb(cid, rows)
    _fb_summarize(cid)
    save_cases(cases)
    log.info("案例 %s 收到 %s 的反馈 tags=%s rating=%s", cid,
             u["username"], tags, rating)
    return {"ok": True, "mine": row, "fb_count": cases[cid]["fb_count"],
            "fb_tags": cases[cid]["fb_tags"],
            "fb_rating": cases[cid]["fb_rating"]}


@app.get("/api/cases/{cid}/feedback")
async def get_case_feedback(request: Request, cid: str):
    """取反馈。员工只看得到自己的 + 汇总；管理员能看到每个人的备注。"""
    u = require(request)
    if cid not in cases:
        raise HTTPException(404, "案例不存在")
    rows = _load_fb(cid)
    mine = [r for r in rows if r.get("user") == u["username"]][-1:]
    c = cases[cid]
    out = {"mine": mine[0] if mine else None,
           "fb_count": c.get("fb_count") or 0,
           "fb_tags": c.get("fb_tags") or {},
           "fb_rating": c.get("fb_rating")}
    if u.get("role") == "admin":
        out["all"] = rows
    else:
        # 不暴露同事写的备注，只给"几个人投了什么"的统计
        out["all"] = [{"user": r.get("user"), "tags": r.get("tags"),
                       "rating": r.get("rating"), "seg": r.get("seg")}
                      for r in rows]
    return out


@app.post("/api/cases/{cid}/analyze")
async def reanalyze_case(request: Request, cid: str):
    """重算韵律六维（换了分析口径、或上次跑失败时用）。"""
    require(request)
    if cid not in cases:
        raise HTTPException(404, "案例不存在")
    threading.Thread(target=_analyze_case, args=(cid,), daemon=True).start()
    return {"ok": True, "msg": "已在后台重算，稍后刷新查看 prosody 字段"}


@app.post("/api/cases/{cid}/meta")
async def update_case_meta(request: Request, cid: str,
                           name: str = Form(""),
                           gender: str = Form(""),
                           source: str = Form(""),
                           tags: str = Form(""),
                           note: str = Form("")):
    """补改标注。

    单独开这个接口是因为：标注（尤其 ElevenLabs 的 voice/model/speed）
    往往是事后才打听到的，为改几个字重传几百 MB 音频不划算。
    只传要改的字段，留空表示不改。
    """
    u = require(request)
    c = cases.get(cid)
    if not c:
        raise HTTPException(404, "案例不存在")
    if u.get("role") != "admin" and c.get("owner") != u.get("username"):
        raise HTTPException(403, "无权修改他人上传的案例")
    if gender.strip():
        g = gender.strip().lower()
        if g not in CASE_GENDERS:
            raise HTTPException(400, f"gender 需在 {list(CASE_GENDERS)} 之内")
        c["gender"] = g
    if source.strip():
        s = source.strip().lower()
        if s not in CASE_SOURCES:
            raise HTTPException(400, f"source 需在 {list(CASE_SOURCES)} 之内")
        c["source"] = s
    for key, val in (("name", name), ("tags", tags), ("note", note)):
        if val.strip():
            c[key] = "".join(ch for ch in val.strip()
                             if ch not in '/\\:*?"<>|') if key == "name" else val.strip()
    save_cases(cases)
    return c


@app.delete("/api/cases/{cid}")
async def delete_case(request: Request, cid: str):
    u = require(request)
    c = cases.get(cid)
    if not c:
        raise HTTPException(404, "案例不存在")
    if u.get("role") != "admin" and c.get("owner") != u.get("username"):
        raise HTTPException(403, "无权删除他人上传的案例")
    shutil.rmtree(_case_dir(cid), ignore_errors=True)
    del cases[cid]
    save_cases(cases)
    log.info("删除案例 %s (%s)", cid, c.get("name"))
    return {"ok": True}


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
