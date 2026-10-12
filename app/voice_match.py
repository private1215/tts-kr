# -*- coding: utf-8 -*-
"""克隆音色的「保真增强」：把生成结果的长时频谱对齐到该音色的原始录音。

为什么需要这一步（2026-09-28 实测，详见 docs/声音克隆机制与排障.md）：
Qwen3-TTS 克隆出来的语音相对**标杆音频**（原表述「真人原声」，2026-10-12 订正：
案例库音频全是 ElevenLabs 生成的，本文件里的「原声/源录音」一律指该标杆音频）
**系统性缺 2~4kHz 能量**。同一句话，
源录音该频段占 15.84%、克隆生成只有 4.03%；频谱质心 2216Hz → 1795Hz，听感发闷。
已经用两个对照实验排除掉常见的甩锅对象：

    源原声 → 过一遍 codec(encode→decode) : 2-4k 只掉 12%，质心只掉 2%
    源原声 → 44.1k 降采样到 24k          : 2-4k 完全不变

损失发生在自回归采样环节，模型配置和编解码都改不动，所以只能在波形后处理上补。
做法是把每段音频的长时平均谱(LTAS)对齐到音色源录音的 LTAS。

**必须逐段匹配，不能用整条一条曲线**：同一条任务内各段的偏离程度相差可达 4.7 倍
（实测 81 段里最差 3.06%、最好 14.38%），全局曲线会把本来就健康的段推过头
（实测会被推到 20.9%，远超源基准 12.65%）。
"""
import os
import threading

import numpy as np
import librosa
import soundfile as sf

TARGET_SR = 24000
N_FFT = 2048
HOP = 512
N_MELS = 128
# 增益限幅：超过这个值的补偿会把偶发的嘶声/气声一起放大，反而更假
CLAMP_DB = 6.0
# mel 轴平滑点数，避免窄带过冲造成金属味
SMOOTH = 9
# 太短的段统计量不可靠，直接跳过
MIN_SEC = 0.30
# 源录音最长分析时长：LTAS 取 5 分钟已经足够稳定，没必要读整段（可能十几分钟）
MAX_ANALYZE_SEC = 300.0

_LOCK = threading.Lock()
_CACHE = {}
_MEL = {}


def _mel(sr):
    fb = _MEL.get(sr)
    if fb is None:
        fb = (librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS,
                                  fmin=50, fmax=sr / 2),
              librosa.mel_frequencies(n_mels=N_MELS, fmin=50, fmax=sr / 2))
        _MEL[sr] = fb
    return fb


def ltas(audio, sr=TARGET_SR):
    """长时平均谱（mel 域 dB）。只统计较响的帧，避免静音把曲线拉平。"""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < int(sr * MIN_SEC):
        return None
    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP)) ** 2
    fb, _ = _mel(sr)
    M = fb @ S
    if M.shape[1] < 3:
        return None
    e = M.mean(axis=0)
    sel = e > np.percentile(e, 50)
    if int(sel.sum()) < 3:
        sel = np.ones(M.shape[1], dtype=bool)
    return 10 * np.log10(M[:, sel].mean(axis=1) + 1e-12)


def build_target(path):
    """读音色源录音 → 统一到 24k → 算 LTAS。任何异常都返回 None（降级不处理）。"""
    if not path or not os.path.exists(path):
        return None
    try:
        x, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size < sr * MIN_SEC:
        return None
    x = x[:int(min(x.size, MAX_ANALYZE_SEC * sr))]
    if sr != TARGET_SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR)
    return ltas(np.ascontiguousarray(x, dtype=np.float32))


def get_target(voice_key, meta):
    """取（并缓存）某个音色的目标曲线。

    优先用录入时的原始长录音 sources/*_full.wav（信息量最大），
    找不到才退回裁剪后的参考片段；两者都没有则返回 None，调用方跳过增强。
    缓存按 (音色, 路径, mtime, size) 失效，音色重录后自动重算。
    """
    for p in (meta.get("src_path"), meta.get("path")):
        if not p or not os.path.exists(p):
            continue
        try:
            st = os.stat(p)
        except OSError:
            continue
        key = (voice_key, p, int(st.st_mtime), int(st.st_size))
        with _LOCK:
            hit = _CACHE.get(key)
        if hit is not None:
            return hit
        t = build_target(p)
        if t is None:
            continue
        with _LOCK:
            _CACHE.clear()          # 只留当前这一个，避免多个音色常驻内存
            _CACHE[key] = t
        return t
    return None


def match_audio(audio, target, sr=TARGET_SR, clamp_db=CLAMP_DB):
    """单段谱匹配：按频率给 STFT 加增益后还原。"""
    if target is None or audio is None:
        return audio
    if len(audio) < int(sr * MIN_SEC):
        return audio
    cur = ltas(audio, sr)
    if cur is None:
        return audio
    gain = np.clip(target - cur, -clamp_db, clamp_db)
    gain = np.convolve(gain, np.ones(SMOOTH) / SMOOTH, mode="same")
    _, mel_f = _mel(sr)
    bin_f = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    gdb = np.interp(bin_f, mel_f, gain)
    gdb[bin_f < 90] = 0.0           # 90Hz 以下不动，避免带进隆隆声
    S = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP)
    y = librosa.istft(S * (10.0 ** (gdb / 20.0))[:, None],
                      hop_length=HOP, length=len(audio))
    return np.asarray(y, dtype=np.float32)


def match_segments(audios, target, sr=TARGET_SR, clamp_db=CLAMP_DB):
    """逐段谱匹配。任一段出错就原样保留——后处理绝不能毁掉整条音频。

    返回 (新的音频列表, 实际处理段数)
    """
    if target is None:
        return list(audios), 0
    out = []
    n = 0
    for a in audios:
        try:
            y = match_audio(a, target, sr, clamp_db)
        except Exception:
            y = a
        if y is None or len(y) == 0:
            y = a
        else:
            n += 1
        out.append(y)
    return out, n
