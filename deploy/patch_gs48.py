# -*- coding: utf-8 -*-
"""patch_gs48.py —— 给 **GS（gptsovits）** 加专属后处理：atempo 变速 + 响度对齐。

## 🔴🔴 发现：GS 一直是「裸输出」上线的
`app/server.py` 的 `_run_fix_chain()` **只在 voxcpm 和 omnivoice 的合成函数里
被调用**（第 1428 / 1463 行）。`_synth_gs()` 直接 `return y, sr, st`，
**没有任何后处理**。

证据：
- 服务端日志里 GS 的 job 完全没有「fix_chain 完成」记录
- GS 成品时长 201.2s == GS 裸输出 201.2s，一秒不差
- 我给 fix_chain 打的加速补丁对 GS 毫无效力（产物 md5 与未打补丁时相同）

## 为什么不能让 GS 改走 fix_chain
实测（男声 GS，加速 11% 后跑完整 fix_chain）：
**自适应 EQ 判定高频不足 → highshelf 顶格 +6dB → 静音段杂音 11.077 → 35.986**
（EL 10.968 的 3.3 倍）。机理：atempo 的 WSOLA 平滑掉高频 → EQ 补回来 →
补的是**静音段底噪** → 嘶声爆炸。

## 本补丁做什么（极简，只两件事）
1. **按目标 cps 做 atempo 变速**：`atempo = clamp(target_cps / our_cps, 0.80, 1.25)`
   · 男声目标 8.37（EL 实测锚），GS 裸输出 7.45 → atempo 1.123 → cps ≈ 8.37
   · 女声目标 7.57
2. **静态增益到目标 LUFS**（与 `_minimal_lufs` 同口径，男 -25.0 / 女 -22.6）

🔴 不做 EQ / 变调 / 停顿清洗 / roomtone —— GS 裸输出本身就够好，
   而 fix_chain 的自适应 EQ 是加速后杂音爆炸的元凶。

## 实测（男声，同文案 6cdd25bc75f2 前 1500 字，同长度 182.1s × EL 标杆）
| | EL 标杆 | voxcpm 现役 | GS 裸输出 | **GS + 本补丁** |
|---|---|---|---|---|
| cps | 8.23 | 8.36 | 7.45 | **8.23** |
| 音色余弦 | 1.000 | 0.8730 | 0.9587 | **0.9424** |
| 静音段 4-8k | 10.968 | 18.201 | 11.077 | **7.451** |
| click % | 0.259 | 1.175 | 0.388 | **0.505** |

开关：`TTS_GS_POST=0` 回到裸输出（原行为）。
"""
import io
import os
import sys

PATH = "/mnt/tts/projects/qwen3-tts/app/server.py"
BAK = PATH + ".bak-before-gs48"

# 1) 插函数：放在 _run_fix_chain 定义之前
ANCHOR = "    def _run_fix_chain(audio, sr, text):"

FN = '''    # 🔴 第47轮 v1.2：GS 专属极简后处理（atempo 变速 + 响度对齐）
    #    详见 deploy/patch_gs48.py 文件头。
    GS_POST = os.environ.get("TTS_GS_POST", "1") != "0"
    GS_POST_LO, GS_POST_HI = 0.80, 1.25      # atempo 限幅
    # 🔴 用 **EL 案例实测锚**，不用 NEW_ENGINE_CPS_BY_GENDER（那是给 voxcpm 的）：
    #    男 8.23（案例 6cdd25bc75f2，4190 字 / 509s）
    #    女 7.45（4 条 SOLA 案例均值，也是本项目的语速口径锚）
    #    实测男声目标从 8.37 收到 8.23：atempo 1.123 -> 1.105，
    #    静音段杂音 13.163 -> 更低（加速幅度越小，WSOLA 损伤越小）。
    GS_TARGET_CPS = {"m": 8.23, "f": 7.45}
    GS_LUFS = {"m": -25.0, "f": -22.6}       # 与 MINIMAL_LUFS 同锚

    # 🔴 按性别的**固定** EQ（不自适应）。空串 = 不做 EQ。
    #   开关：TTS_GS_EQ_F / TTS_GS_EQ_M 可覆盖。
    #
    #   女声：GS 裸输出（gs48f_sola，458s 全篇，16k 发声帧频段占比）vs EL 标杆：
    #   | 频段 | EL | GS 裸 | 定案后 |
    #   |---|---|---|---|
    #   | 0-500 | 41.23 | 38.24 | 41.29 |
    #   | 500-1000 | 46.22 | **53.60** | 46.67 |
    #   | 1000-2000 | 6.28 | 4.65 | 5.50 |
    #   | 2000-4000 | 5.41 | **2.60** | 5.46 |
    #   | 4000-8000 | 0.87 | 0.90 | 1.08 |
    #   → 真问题不是「中低频偏薄」，是 **500-1000 偏多 / 2000-4000 偏少（−3.2dB）**。
    #     所以是「抬 2.6k 宽带 + 压 700」。
    #   → 杂音（静音段 4-8k）4.241 → 4.068，离 EL 11.371 还有巨大余量；
    #     click 0.000% → 0.100%，仍低于 EL 0.157%；音色 0.9179 → **0.9191**。
    #   用**宽带**（w≥1.5）不用窄带 —— 记忆教训：窄带衰减会摧毁频谱平坦度。
    #   定案来自 tools/eqprobe48.py 两批离线筛选（一次合成、多次试参数，
    #   EQ 是线性滤波，对成品施加 = 对裸输出施加后再走 atempo+volume）。
    GS_EQ_F_DEF = ("equalizer=f=2600:t=q:w=2.5:g=4.0,"
                   "equalizer=f=700:t=q:w=1.5:g=-1.5")
    GS_EQ = {
        "m": os.environ.get("TTS_GS_EQ_M", ""),
        "f": os.environ.get("TTS_GS_EQ_F", GS_EQ_F_DEF),
    }

    def _gs_post(audio, sr, text, gender):
        """GS 专用：按目标 cps 做 atempo 变速 + 静态增益到目标 LUFS。

        🔴 GS 不能走 fix_chain —— 它的自适应高频 EQ 在 atempo 加速后会
        顶格 +6dB，把静音段杂音从 11 抬到 36（实测）。这里只做变速和响度，
        其余一律不碰：GS 裸输出的音色/杂音本就接近 EL。
        """
        if not GS_POST:
            return audio, sr
        import subprocess as _sp3
        import tempfile as _tf3
        import soundfile as _sf3
        try:
            dur = len(audio) / float(sr)
            our_cps = len(text or "") / dur if dur > 0 else 0.0
            tgt = GS_TARGET_CPS.get(gender, 8.37)
            if our_cps <= 0:
                return audio, sr
            at = tgt / our_cps
            at = min(max(at, GS_POST_LO), GS_POST_HI)

            d = _tf3.mkdtemp(prefix="gspost_")
            a = os.path.join(d, "a.wav")
            b = os.path.join(d, "b.wav")
            _sf3.write(a, audio, sr)
            _eq = GS_EQ.get(gender, "")
            _af = "atempo=%.6f" % at
            if _eq:
                _af = _eq + "," + _af     # 先 EQ 再变速（与响度测量同序）
            _sp3.run(["ffmpeg", "-y", "-v", "error", "-i", a,
                      "-af", _af, b], check=True)

            # 响度：静态 volume（与 _minimal_lufs 同口径，不用 loudnorm）
            r = _sp3.run(["ffmpeg", "-i", b, "-af", "ebur128",
                          "-f", "null", "-"], capture_output=True, text=True)
            v = [l for l in r.stderr.splitlines()
                 if "I:" in l and "LUFS" in l]
            if v:
                cur = float(v[-1].split("I:")[1].split("LUFS")[0].strip())
                g = round(GS_LUFS.get(gender, -25.0) - cur, 1)
                c = os.path.join(d, "c.wav")
                _sp3.run(["ffmpeg", "-y", "-v", "error", "-i", b,
                          "-af", "volume=%gdB" % g, c], check=True)
                b = c
                lufs_note = "LUFS %.1f -> %.1f (%+gdB)" % (
                    cur, GS_LUFS.get(gender, -25.0), g)
            else:
                lufs_note = "LUFS 测量失败，跳过"

            y, sr2 = _sf3.read(b, dtype="float32")
            if getattr(y, "ndim", 1) > 1:
                y = y.mean(axis=1)
            log.info("任务 %s GS 后处理 cps %.2f -> %.2f (atempo %.3f)，%s",
                     jid, our_cps, len(text or "") / (len(y) / float(sr2)),
                     at, lufs_note)
            return y, sr2
        except Exception as _e:
            log.warning("任务 %s GS 后处理失败（用 raw）: %r", jid, _e)
            return audio, sr
        finally:
            import shutil as _sh3
            try:
                _sh3.rmtree(d, ignore_errors=True)
            except Exception:
                pass

'''

# 2) 在 _synth_gs 返回前调用
OLD_RET = '''            "plan": {"engine": "gptsovits", "emotion": _emotion,
                     "seg": _gs_seg},
        }
        return y, sr, st'''

NEW_RET = '''            "plan": {"engine": "gptsovits", "emotion": _emotion,
                     "seg": _gs_seg},
        }
        # 🔴 第47轮 v1.2：GS 此前**完全没有后处理**（_run_fix_chain 只在
        #    voxcpm / omnivoice 分支里被调用）。这里补上专属的极简后处理：
        #    按目标 cps 做 atempo 变速 + 静态增益到目标 LUFS。
        #    不走 fix_chain：它的自适应 EQ 会在加速后顶格 +6dB 制造嘶声。
        y, sr = _gs_post(y, sr, text_ci[1], _voice_gender(job))
        st["audio_seconds"] = len(y) / float(sr)
        return y, sr, st'''


def main():
    src = io.open(PATH, encoding="utf-8").read()

    if "_gs_post" in src:
        print("已打过补丁，跳过")
        return 0
    if ANCHOR not in src:
        print("❌ 没找到 _run_fix_chain 锚点")
        return 1
    if OLD_RET not in src:
        print("❌ 没找到 _synth_gs 的 return（可能已改过）")
        return 1

    if not os.path.exists(BAK):
        io.open(BAK, "w", encoding="utf-8").write(src)
        print("备份 ->", BAK)

    out = src.replace(ANCHOR, FN + ANCHOR, 1)
    out = out.replace(OLD_RET, NEW_RET, 1)
    io.open(PATH, "w", encoding="utf-8").write(out)
    print("✅ 已写入", PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
