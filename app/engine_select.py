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

⚠️ 三个踩过的坑（改这里必看）：
  1) tts_engine.synth_long 的 ref_audio 必须是 **(audio, sr) 元组**，传裸数组会炸
  2) Qwen 必须复用 server.engine 全局实例，**不能 new TTSEngine()** ——
     那会把同一个模型再加载一份进显存（约 4.2 GiB）
  3) **显存串行，绝不并行**：12G 卡放不下 Qwen(常驻~4.2G/峰值6.9G) + CosyVoice3(3.2G)，
     实测两边同时驻留必 OOM(CUBLAS_STATUS_ALLOC_FAILED)。
     正解是「先这个、后那个」：跑某引擎候选前，把另一个引擎从显存彻底卸掉
     （Qwen 走 engine.unload()，CosyVoice3 走 cosy_client.unload()），
     目标引擎由自身懒加载机制自动起。候选按引擎分组连跑，整轮只在边界切换一次。
"""
import os
import time
import logging
import tempfile
import threading

import numpy as np
import soundfile as sf

log = logging.getLogger("tts")

# 全局互斥：同一时刻只允许一个选型在跑（两个选型并发会抢同一块显存）。
# 录入音色时已有「队列空才跑」守卫，这里是兜底双保险（手动触发端点等）。
_SELECT_LOCK = threading.Lock()

# 试听文本：一段中性的韩语股东播报体。
# ⚠️ 必须**明显长于**「情感指令 + 参考音频文字稿」（实测约 240 字），否则
#   CosyVoice3 会报 `too short than prompt text` 并直接失败 —— 这是踩过的坑。
PREVIEW_TEXT = (
    "안녕하세요. 주주 여러분, 반갑습니다. "
    "오늘 시장은 생각보다 훨씬 더 강한 흐름을 보여주고 있습니다. "
    "외인 창구에서는 순매도를 기록하고 있지만, 개인과 기관은 매집 흐름을 이어가고 있습니다. "
    "투매가 끝났다, 세력 매집이 끝났다, 이렇게 확정 지을 순 없습니다. "
    "수급 창을 한번 보실까요? 지금 구간에서는 절대 섣불리 판단하시면 안 됩니다. "
    "무엇보다 중요한 건 시나리오 매매의 뼈대를 세우는 겁니다. "
    "첫째, 이탈하면 손절입니다. 둘째, 뚫어주면 홀딩입니다. "
    "셋째, 거래량이 실리지 않으면 관망입니다. "
    "이 원칙은 어떤 종목에도 그대로 적용됩니다."
)

# CER 及格线：超过就认为念错太多，直接淘汰
CER_PASS = float(os.environ.get("TTS_SELECT_CER_PASS", "0.13"))
# 打分权重：CER 越低越好，音高动态越高越好
W_CER = float(os.environ.get("TTS_SELECT_W_CER", "10.0"))
W_DYN = float(os.environ.get("TTS_SELECT_W_DYN", "1.0"))


# 选型收尾后是否立即把 Qwen 重新加载回显存（恢复常驻稳态）。
# 1=立即重载（ health 保持 loaded、下个生成任务零加载等待，代价是选型多耗一次加载时间）
# 0=保持卸载，下次生成经 _ensure_variant 懒加载（选型返回更快）
TTS_SELECT_RELOAD_QWEN = os.environ.get("TTS_SELECT_RELOAD_QWEN", "1") == "1"


def _wait_free_vram(min_free_gib=6.0, timeout=90.0):
    """等整卡空闲显存真正达标再往下走。

    实现统一收敛到 `app/vram.py`（生成路径与选型路径共用一套），
    避免出现「一处改了另一处没改」的静默失效 —— 2026-09-29 的 P0 正是这么来的。
    """
    try:
        import vram
        return vram.wait_free_vram(min_free_gib, timeout)
    except Exception:
        return None


def _switch_engine(engine_name):
    """显存串行核心：跑某引擎前，把另一个引擎从显存彻底赶出去并**等显存真释放**。

    实现统一收敛到 `app/vram.py`。卸载都是幂等的，重复调用无副作用。
    """
    try:
        import vram
        return vram.switch_to(engine_name)
    except Exception as e:
        log.warning("切换引擎 -> %s 失败（可能 OOM）: %s", engine_name, repr(e)[:120])
        return None


def _finish_benchmark():
    """收尾：卸掉 CosyVoice3，按需把 Qwen 恢复到常驻稳态。"""
    try:
        import cosy_client
        cosy_client.unload()
    except Exception:
        pass
    if TTS_SELECT_RELOAD_QWEN:
        try:
            import server
            eng = getattr(server, "engine", None)
            if eng is not None:
                log.info("选型收尾：重新加载 Qwen 恢复常驻稳态")
                eng.warmup()
        except Exception as e:
            # 重载失败不致命：下次生成会懒加载兜底
            log.warning("选型收尾重载 Qwen 失败（下次生成将懒加载）: %s", repr(e)[:160])


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
        if sr != 24000:
            x = librosa.resample(x, orig_sr=sr, target_sr=24000)
            sr = 24000
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


def _qwen_engine():
    """复用 server 的全局引擎实例，避免重复加载模型占显存。"""
    import server
    eng = getattr(server, "engine", None)
    if eng is None:                      # 独立脚本调用时兜底
        from tts_engine import TTSEngine
        eng = TTSEngine()
    return eng


def _make_wav(engine_name, text, ref_wav, ref_text, instruct, language, wav):
    """跑一个候选，把结果写成 wav。返回错误信息字符串，成功返回 ''。"""
    if engine_name == "cosyvoice3":
        import cosy_client
        audio, sr, meta = cosy_client.synth(
            text=text, ref_wav=ref_wav, ref_text=ref_text or "",
            instruct=instruct or "neutral")
        if audio is None:
            return (meta.get("error") or "合成失败")[:160]
        sf.write(wav, audio, sr)
        return ""

    # Qwen：ref_audio 必须是 (audio, sr) 元组
    data, sr = sf.read(ref_wav, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    eng = _qwen_engine()
    audio, out_sr, _st = eng.synth_long(
        text=text, mode="clone", language=language, speaker=None,
        instruct="", ref_audio=(data, sr), ref_text=ref_text,
        max_chars=60, batch_size=8, seed=42)
    sf.write(wav, audio, out_sr)
    return ""


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

    # 候选按引擎分组连跑：同引擎的相邻执行，整轮只在引擎边界切换一次显存，
    # 符合「先这个、后那个」的串行原则（12G 卡放不下两个引擎并行）。
    _ENGINE_ORDER = {"qwen": 0, "cosyvoice3": 1}
    presets = sorted(presets, key=lambda p: _ENGINE_ORDER.get(p["engine"], 9))

    # 全局互斥：已有选型在跑就直接跳过（两个选型并发抢显存必 OOM）
    if not _SELECT_LOCK.acquire(blocking=False):
        log.warning("已有选型任务在跑，本次跳过")
        return []
    try:
        for p in presets:
            t0 = time.time()
            _switch_engine(p["engine"])
            wav = os.path.join(tmp_dir, f".sel_{p['key']}_{int(t0 * 1000)}.wav")
            try:
                err = _make_wav(p["engine"], text, ref_wav, ref_text,
                                p.get("instruct") or "", language, wav)
                if err:
                    log.warning("候选 %s 生成失败: %s", p["key"], err)
                    results.append(dict(p, cer=None, dyn=None, score=None, ok=False,
                                        error=err[:120]))
                    continue
            except Exception as e:
                log.warning("候选 %s 异常: %s", p["key"], repr(e)[:200])
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
    finally:
        try:
            _finish_benchmark()
        except Exception as e:
            log.warning("选型收尾异常（不影响已产出结果）: %s", repr(e)[:160])
        _SELECT_LOCK.release()

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


def select_for_voice(name, voice, language="Korean", save=True):
    """为一个已录入的音色跑选型，把最优组合写回 voices。

    voice: voices[name] 字典，至少含 path / ref_text
    返回写入的 auto 字典（失败返回 None）。
    """
    ref_wav = voice.get("path") or ""
    ref_text = voice.get("ref_text") or ""
    if not ref_wav or not os.path.exists(ref_wav):
        log.warning("音色 %s 参考音频不存在，跳过选型", name)
        return None
    t0 = time.time()
    results = benchmark_voice(ref_wav, ref_text, language=language)
    best = pick_best(results)
    if not best:
        log.warning("音色 %s 选型无可用结果", name)
        return None

    auto = {
        "engine": best.get("engine"),
        "emotion": best.get("key"),
        "instruct": best.get("instruct") or "",
        "cer": best.get("cer"),
        "dyn": best.get("dyn"),
        "score": best.get("score"),
        "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed": round(time.time() - t0, 1),
        "candidates": [
            {k: r.get(k) for k in ("key", "engine", "cer", "dyn", "score", "ok")}
            for r in results
        ],
    }
    log.info("音色 %s 自动选型 -> 引擎=%s 情感=%s (CER=%s 动态=%s)，候选 %d 个，耗时 %.0fs",
             name, auto["engine"], auto["emotion"], auto["cer"], auto["dyn"],
             len(results), time.time() - t0)
    if save:
        try:
            import server
            v = server.voices.get(name)
            if v is not None:
                v["auto"] = auto
                server.save_voices(server.voices)
        except Exception as e:
            # 落盘失败不能让整个选型白跑：结果照样返回，调用方本次仍用得上
            log.warning("音色 %s 选型结果写盘失败: %s", name, repr(e)[:160])
    return auto
