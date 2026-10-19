# -*- coding: utf-8 -*-
"""第50轮：修 GS **上线级硬阻塞** —— 参考音频 15 秒会被 GS 拒绝（HTTP 500）。

## 为什么必须修

GS 要求参考音频 **3~10 秒**（见项目规范「GS 硬坑」1），超长直接 500。
而员工默认音色 `BEST_CLONE = {"male": "EL_male", "female": "EL_female"}`
（app/server.py），这两个 wav **都是 15 秒**：

    EL_female.wav  15.080s
    EL_male.wav    14.980s

也就是说：**只要把默认引擎切到 gptsovits，员工每一次提交都会 500**。
实测（2026-10-17，直接打 GS 服务 /synth）：

    ref_wav=EL_female.wav(15.08s)              -> HTTP 500
    ref_wav=prompt_sola_00b6a45a.wav(10.00s)   -> HTTP 200

这个坑以前只在「手工跑实验」的路径上绕过（实验时手工传合规切片），
一旦走 /api/jobs 的 BEST_CLONE 默认路径就没人挡得住。

## 修法

GS 分支把「超长的生产参考音频」换成**同人**的 GS 专用短切片，
并且 **ref_text 必须跟着换**（换音频不换文本 = 语速崩 + 把参考文本当正文反复念，
这是规范里记过的 CER 0.66 事故）。

配置：值是 **voices.json 里的音色名**（不是路径），这样 path 和 ref_text
都直接取自音色表，与实验时用的完全一致，不会再次对不齐：

    TTS_GS_REF_MAP="EL_female=SOLA_gs10,EL_male=yohan_gs_f8"

（这两个就是 z20 里 GS 素材实际用的音色：gs48f_eq -> SOLA_gs10、
 gs48m_post2 -> yohan_gs_f8，见 /mnt/tts/data/jobs.json）

🔴 **不配置就完全走原逻辑**，不设环境变量 = 行为与补丁前一致，不误伤任何引擎。

## 用法

    python3 tools/patch_gsref50.py            # 干跑（只打印要不要改）
    python3 tools/patch_gsref50.py --apply    # 真的打补丁（自动 .bak）
"""
import argparse
import io
import os
import shutil
import sys
import time

TARGET = "/mnt/tts/projects/qwen3-tts/app/server.py"
MARKER = "_gs_ref_pair"

# ---- 要插入的模块级代码（放在 _gs_group_of 之前）----
BLOCK = '''
# ---------------------------------------------------------------------------
# 🔴 第50轮：GS 参考音频替换表（GS 上线前的硬阻塞，详见 deploy/patch_gsref50.py）
#
# GS 要求参考音频 **3~10 秒**，超长直接 HTTP 500。而员工默认音色
# BEST_CLONE = EL_female / EL_male 都是 **15 秒** —— 只要把默认引擎切成
# gptsovits，员工每一次提交都会 500。实测：
#   ref=EL_female.wav(15.08s)            -> HTTP 500
#   ref=prompt_sola_00b6a45a.wav(10.00s) -> HTTP 200
#
# 修法：GS 分支把超长参考音频换成**同人**的 GS 专用短切片，且 ref_text 一并
# 跟随（换音频不换文本 = 语速崩 + 把参考文本当正文念，CER 0.66 事故）。
# 配置：值是 **voices.json 里的音色名**（不是路径），path + ref_text 都取自
# 音色表，保证与实验素材逐字一致：
#   TTS_GS_REF_MAP="EL_female=SOLA_gs10,EL_male=yohan_gs_f8"
# 🔴 不配置就完全走原逻辑，不设环境变量 = 行为与打补丁前一致。
# ---------------------------------------------------------------------------
GS_REF_MAX_SEC = float(os.environ.get("TTS_GS_REF_MAX_SEC", "10"))
_GS_REF_MAP = {}
for _pair in os.environ.get("TTS_GS_REF_MAP", "").split(","):
    if "=" not in _pair:
        continue
    _k, _v = _pair.split("=", 1)
    _k, _v = _k.strip(), _v.strip()
    if _k and _v:
        _GS_REF_MAP[_k] = _v
if _GS_REF_MAP:
    log.info("GS 参考音频替换表: %s", _GS_REF_MAP)


def _gs_ref_pair(wav, text, voice_name=""):
    """GS 分支专用：超长参考音频 -> 同人短切片，ref_text 一并跟随。

    返回 (wav_path, ref_text)。
    🔴 只在「参考音频超过 GS_REF_MAX_SEC」时才换：本来就合规的参考音频
    保持原样，避免管理员手工传了好切片却被替换掉。
    命中不了也原样返回（GS 服务会报 500），但这里先打一条 warning，
    排查时不用去翻 gs_service.log。
    """
    if not wav or not os.path.exists(wav):
        return wav, text
    try:
        import soundfile as _sfi
        _dur = float(_sfi.info(wav).duration)
    except Exception:
        _dur = 0.0
    if _dur and _dur <= GS_REF_MAX_SEC:
        return wav, text                      # 本来就合规
    _key = (_GS_REF_MAP.get((voice_name or "").strip())
            or _GS_REF_MAP.get(os.path.splitext(os.path.basename(wav))[0]))
    if not _key:
        log.warning("GS 参考音频 %.2fs 超过 %.0fs 且没有替换目标（%s）"
                    "-> GS 大概率报 500", _dur, GS_REF_MAX_SEC, wav)
        return wav, text
    # 值是音色名：path 与 ref_text 都从音色表取，和实验素材逐字一致
    _av = voices.get(_key) or {}
    alt = _av.get("path") or ""
    new_text = (_av.get("ref_text") or "").strip() or text
    if not alt or not os.path.exists(alt):
        log.warning("GS 参考替换音色 %s 不存在或 path 无效（%s），沿用原参考",
                    _key, alt)
        return wav, text
    if not (_av.get("ref_text") or "").strip():
        log.warning("GS 替换音色 %s 没有 ref_text，沿用原 ref_text"
                    "（语速可能崩）", _key)
    log.info("GS 参考音频替换：%s(%.2fs) -> 音色 %s（%s，ref_text %d 字）",
             os.path.basename(wav), _dur, _key, os.path.basename(alt),
             len(new_text or ""))
    return alt, new_text


'''

OLD_CALL = """        ok, payload = gs_client.synth(
            text=text_ci[1],
            ref_wav=ref_wav_path or "",
            ref_text=ref_text or "","""
NEW_CALL = """        # 🔴 第50轮：GS 参考音频必须 3~10 秒，而员工默认音色是 15 秒的
        #    EL_female / EL_male -> 不换就是全员 HTTP 500。见 _gs_ref_pair。
        _gw, _gt = _gs_ref_pair(ref_wav_path, ref_text,
                                (job.get("voice") or "").strip())
        ok, payload = gs_client.synth(
            text=text_ci[1],
            ref_wav=_gw or "",
            ref_text=_gt or "","""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    src = io.open(TARGET, encoding="utf-8").read()
    if MARKER in src:
        print("已经打过补丁（找到 %s），跳过" % MARKER)
        return 0

    todo = []
    if "def _gs_group_of(job):" not in src:
        print("❌ 找不到锚点 def _gs_group_of(job): —— 文件结构变了，"
              "不要盲打")
        return 1
    todo.append("插入 _gs_ref_pair（锚点 _gs_group_of）")

    if OLD_CALL not in src:
        print("❌ 找不到 gs_client.synth 调用块 —— 文件结构变了，不要盲打")
        return 1
    todo.append("改写 _synth_gs 的 ref_wav/ref_text")

    print("将做：")
    for t in todo:
        print("  - " + t)
    if not a.apply:
        print("\n（干跑，加 --apply 才真的改）")
        return 0

    bak = TARGET + ".bak-" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(TARGET, bak)
    out = src.replace("def _gs_group_of(job):", BLOCK.lstrip("\n")
                      + "def _gs_group_of(job):", 1)
    out = out.replace(OLD_CALL, NEW_CALL, 1)
    io.open(TARGET, "w", encoding="utf-8").write(out)

    # 语法自检：改坏了就直接回滚，绝不留一个起不来的服务
    import subprocess
    r = subprocess.run([sys.executable, "-c",
                        "import ast,io;ast.parse(io.open(%r,encoding='utf-8').read())"
                        % TARGET])
    if r.returncode != 0:
        shutil.copy2(bak, TARGET)
        print("❌ 语法检查失败，已回滚到 %s" % bak)
        return 1
    print("✅ 补丁完成，备份 %s" % os.path.basename(bak))
    print("⚠️ 下一步：重启 qwen-tts 生效（改 app/ 代码后约 10s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
