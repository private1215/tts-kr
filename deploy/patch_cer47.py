# -*- coding: utf-8 -*-
"""第47轮：给所有引擎加「内容闸门」标记（C 级改动，但**不改音频**）。

背景：CER 回检原本只对 cosyvoice3 生效（超阈值回退 qwen），其它引擎即使
CER=0.66（三分之二的字对不上 = 员工说的"乱说话"）也会原样出成品，
一路上到员工耳朵里才发现。

这里**不重掷、不改音频**（重掷会显著拖慢，且改生成路径风险高），
只在 job 上打 content_suspect 标记 + 告警日志，让问题第一时间可见。
开关：TTS_CER_MARK=0 可关闭。
幂等：已打过补丁会跳过。
"""
import shutil, os, sys

P = "/mnt/tts/projects/qwen3-tts/app/server.py"
BAK = P + ".bak-before-cer47"

src = open(P, encoding="utf-8").read()

MARK = "content_suspect"
if MARK in src:
    print("[skip] 已打过补丁")
    sys.exit(0)

shutil.copy2(P, BAK)
print("已备份 ->", BAK)

# 1) 加开关常量
anchor_env = 'COSY_FALLBACK_CER = float(os.environ.get("TTS_COSY_FALLBACK_CER", "0.15"))'
assert anchor_env in src, "找不到 COSY_FALLBACK_CER 定义行"
add_const = anchor_env + (
    '\n\n# 🔴 第47轮：内容闸门标记——**对所有引擎**生效（不只是 cosyvoice3）。\n'
    '# 原逻辑里 CER 超标只让 cosyvoice3 回退；gptsovits/qwen/voxcpm 即使\n'
    '# CER 0.66（= 三分之二的字对不上 = 员工反馈的"乱说话"）也会原样出成品。\n'
    '# 这里只标记 + 告警，不改音频、不重掷（避免拖慢生成）。\n'
    'CER_MARK = os.environ.get("TTS_CER_MARK", "1") != "0"')
src = src.replace(anchor_env, add_const, 1)

# 2) 在 best 选定后插入标记逻辑
anchor = """        audio, sr, st, res = best
        # 质量闸门：CosyVoice3 带了情绪，但念错太多就必须退回 Qwen 重出一遍。"""
assert anchor in src, "找不到 `audio, sr, st, res = best` 锚点"

insert = """        audio, sr, st, res = best
        # 🔴 第47轮 内容闸门：所有引擎通用。重掷若干次仍 CER 超标，说明这一块
        #    内容与原文对不上（乱说话 / 漏读 / 重复念参考文本），必须留痕，
        #    不能像以前那样静悄悄地把可疑成品发出去。
        #    只标记 + 告警，不动音频；排查时看 jobs.json 的 content_suspect。
        if (CER_MARK and res and not res.get("skipped")
                and not res.get("ok")):
            _cer = res.get("cer") or 0.0
            job["content_suspect"] = True
            agg["content_suspect"] = True
            job.setdefault("content_suspect_detail", []).append(
                {"chunk": ci + 1, "cer": round(_cer, 4),
                 "engine": _engine_name})
            log.warning(
                "任务 %s 第 %d/%d 块 CER=%.4f 超过 %.2f：内容与原文对不上"
                "（乱说话/漏读），已标记 content_suspect（引擎 %s）",
                jid, ci + 1, n_chunk, _cer,
                getattr(asr_check, "CER_THRESHOLD", 0.35), _engine_name)
        # 质量闸门：CosyVoice3 带了情绪，但念错太多就必须退回 Qwen 重出一遍。"""
src = src.replace(anchor, insert, 1)

# 3) agg 初始化里补 content_suspect
a2 = '"abnormal_segments": 0, "slow_fixed": 0, "voice_matched": 0,\n'
n = src.count(a2)
src = src.replace(a2, '"abnormal_segments": 0, "slow_fixed": 0, "voice_matched": 0,\n'
                      '            "content_suspect": False,\n')
print(f"agg 初始化补了 {n} 处")

open(P, "w", encoding="utf-8").write(src)
print("[ok] 补丁已写入")

# 语法检查
import py_compile
py_compile.compile(P, doraise=True)
print("[ok] 语法检查通过")
