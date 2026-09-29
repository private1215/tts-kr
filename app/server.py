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
    _match_target = None
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
        ref_key = job["voice"]
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

    def _synth(text_ci):
        """单次块生成，返回 (audio, sr, stats)。"""
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

    for ci, chunk_text in enumerate(chunks):
        best = None
        for attempt in range(ASR_MAX_RETRY + 1):
            seed = base_seed + attempt * 7919
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

        # 只输出 mp3：wav 36MB/块太占磁盘，员工实际用的就是 mp3
        _speed = float(job.get("speed") or 1.0)
        mp3_name = f"{base_name}-{ci + 1}.mp3"
        mp3_abs = os.path.join(folder_path, mp3_name)
        job["stage"] = "encoding"
        tmp_wav = os.path.join(folder_path, f".enc_{jid}_{ci}.wav")
        try:
            sf.write(tmp_wav, audio, sr)
            # 语速：模型无 speed 接口（见文件头 SPEED_MIN 注释），用 atempo 在
            # 未压缩的 wav 上做一次——只经过一次 mp3 编码，且 atempo 变速不变调。
            _cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_wav]
            if abs(_speed - 1.0) > 1e-3:
                _cmd += ["-filter:a", f"atempo={_speed:.3f}"]
            _cmd += ["-codec:a", "libmp3lame", "-b:a", MP3_BITRATE, mp3_abs]
            subprocess.run(_cmd, check=True, timeout=300)
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
        "speed_default": 1.0,
        "speed_min": SPEED_MIN, "speed_max": SPEED_MAX,
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
            elif total and done == 0:
                # 第一批还没出结果，用历史速度粗估
                # 用历史 RTF + 文本长度估算总时长，比"每段固定秒数"准得多
                txt = j.get("text") or ""
                est_audio = (len(txt) / CHARS_PER_SEC) if txt else total * 8.0
                eta = round(max(est_audio * AVG_RTF - elapsed, 1.0), 1)
            else:
                eta = 0
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
                       file: UploadFile = File(...)):
    require(request)
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
    log.info("录入音色 %s: 原 %.1fs -> 成品 %.2fs, ref_text %d 字, 分析 %.2fs, %s",
             safe, dur, out_dur, len(used_text), info["analyze_seconds"], info)
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
    }


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
