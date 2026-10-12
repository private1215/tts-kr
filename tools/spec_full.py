# -*- coding: utf-8 -*-
"""生成「案例库**完整长文案**」的三引擎对照清单（tools/engine_score_all.py 用）。

🔴 为什么必须用完整长文案，而不是 60 秒切片
------------------------------------------------
2026-10-12 用户纠正：「测试的都是不到一分钟的案例，这种没有实际参考对比
价值，肯定得用案例库里的长文本去测试生成才有对比价值」。

真实业务是**单篇 3000~4000 字、成片 5~8 分钟**。60 秒切片测不出：
  · 长文本自回归发散（GS 整段喂会跑飞，覆盖率 0.15）
  · 分段拼接的段间连续性 / 韵律漂移
  · 后处理（响度闭环、peakfit、PBA）在长音频上的累积效应
这些恰恰是"员工一听就出戏"的地方。所以对照一律走**完整稿**。

🔴 三条硬性口径
----------------
1. 参照 = 案例原声 merged.wav（ElevenLabs 产出，非真人录音）。
   案例在本项目里**同时是音色来源和质量标杆**，是同一个东西。
2. 三条文案都来自**测试区**（hold 案例整条不进训练集），
   所有引擎都没背过，分数诚实。
3. 一份 spec 只放**同一条文案** —— engine_score_all 的语义是
   「一份 spec 共享同一参照、同一份文案」。混着放会让覆盖率失效
   （实测 0.85 → 0.046）。

用法：
    python tools/spec_full.py          # 写 tools/score_full_{f,m}.json
"""
import json
import os

OUT = "/mnt/tts/projects/gpt-sovits/out"
DIAG = "/mnt/tts/diag"
HERE = os.path.dirname(os.path.abspath(__file__))

# (说话组, 是否男声, 线上标杆音色, 文案文件名, 案例 id, 输出 spec)
GROUPS = [
    ("SOLA", False, "EL_female",
     "full_00b6a45ad6aa.txt", "00b6a45ad6aa", "score_full_f.json"),
    ("yohan", True, "EL_male",
     "full_1a6c20a60481.txt", "1a6c20a60481", "score_full_m.json"),
]


def build(spk, is_male, qwen_voice, txt_name, cid, pf, cosy, online=None):
    """pf = 文件名前缀：full-f / full-m
    cosy = Cosy 长文案产物路径（None 表示还没跑）
    online = 走线上链路的三引擎产物（tools/engine_full_cmp.py 的 fullp_*）。
            🔴 2026-10-12：**优先用线上链路产出**。CLI 裸推理没过后处理链，
            拿来跟线上成品比韵律等于让 GS/Cosy 裸奔上场（已犯过一次）。
            online 里的项存在就覆盖同引擎的 CLI 项。
    """
    ref = "/mnt/tts/data/cases/%s/merged.wav" % cid
    txt = os.path.join(OUT, txt_name)
    cands = [
        ("案例原声(参照)", ref, True),
        ("GS微调+分段",
         os.path.join(OUT, "%s_%s_ft_e40_seg50.wav"
                      % (pf, txt_name.replace(".txt", ""))), False),
        ("GS零样本",
         os.path.join(OUT, "%s_%s_zs_e40_seg50.wav"
                      % (pf, txt_name.replace(".txt", ""))), False),
        ("Qwen " + qwen_voice,
         os.path.join(DIAG, "qwen_%s_%s.wav"
                      % (pf.replace("-", ""), qwen_voice)), False),
    ]
    # Cosy 产物在才收进来（没跑完不塞 None，否则引擎会把它当成缺失报错）
    if cosy and os.path.exists(cosy):
        cands.append(("Cosy零样本", cosy, False))
    # 线上链路产出：同引擎的 CLI 项被它顶掉
    for name, path in (online or []):
        if not os.path.exists(path):
            continue
        cands = [(n, p, r) for (n, p, r) in cands
                 if not n.startswith(name.split()[0])]
        cands.append((name, path, False))
    return [{"name": n, "path": p, "is_male": is_male, "ref": r,
             "text_file": txt} for n, p, r in cands]


def main():
    for spk, is_male, voice, txt_name, cid, fn in GROUPS:
        pf = "full-f" if not is_male else "full-m"
        gp = "f" if not is_male else "m"
        cosy = os.path.join(OUT, "cosy_full_%s.wav" % ("m" if is_male else "f"))
        online = [
            ("Qwen " + voice, os.path.join(DIAG, "fullp_%s_qwen.wav" % gp)),
            ("Cosy零样本", os.path.join(DIAG, "fullp_%s_cosyvoice3.wav" % gp)),
            ("GS微调+分段", os.path.join(DIAG, "fullp_%s_gptsovits.wav" % gp)),
        ]
        spec = build(spk, is_male, voice, txt_name, cid, pf, cosy, online)
        miss = [s["path"] for s in spec if not os.path.exists(s["path"])]
        with open(os.path.join(HERE, fn), "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        print("写 %-20s %d 项，缺失 %d %s"
              % (fn, len(spec), len(miss), miss[:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
