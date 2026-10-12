# -*- coding: utf-8 -*-
"""生成「多文案 × 多引擎」同文案对照清单（tools/engine_score_all.py 用）。

🔴 为什么这次跑 3 段文案而不是 1 段
----------------------------------------
之前只在 hold 案例的**前 60 秒**上比过一次。单一样本可能被偶然性污染
（那段刚好顺口 / 刚好绕口）。这次从同一条 hold 案例上切 3 段
（开头 / 中段 / 后段），每段让各引擎各生成一遍，看结论稳不稳。

⚠️ 每个文案组**单独出一个 spec 文件**：engine_score_all.py 的语义是
   「一份 spec 内共享同一条参照、同一份文案」；把 3 段文案塞进同一份
   spec 会让不同文案互相比，覆盖率直接失效（实测掉到 0.046）。

⚠️ hold 案例整条都不进训练集（gs_prep_group.py 里 train_cids 排除 hold），
   所以三段文案对所有引擎都是**没背过**的，分数诚实。

文件名约定（由 gs_batch_cmp.py 产出）：
    cmp-<f|m>_<spk>_<slot>_<ft|zs>_e40_seg50.wav
参照原声（由 gs_case_texts.py 产出）：
    probe_<spk>_<slot>.wav

用法：
    python tools/cmp_specs.py
    # 写 tools/cmp_f_000.json ... cmp_m_300.json 共 6 份
"""
import json
import os

OUT = "/mnt/tts/projects/gpt-sovits/out"
HERE = os.path.dirname(os.path.abspath(__file__))
SLOTS = ["000", "150", "300"]
QWEN_DIR = "/mnt/tts/output"

# (说话组, 是否男声, 线上标杆音色, 文件名前缀)
GROUPS = [
    ("SOLA", False, "EL_female", "f"),
    ("yohan", True, "EL_male", "m"),
]


def build(spk, is_male, qwen_voice, pf, slot):
    ref = os.path.join(OUT, "probe_%s_%s.wav" % (spk.lower(), slot))
    txt = os.path.join(OUT, "text_%s_%s.txt" % (spk.lower(), slot))
    cands = [
        ("案例原声(参照)", ref, True),
        ("GS微调+分段",
         os.path.join(OUT, "cmp-%s_%s_%s_ft_e40_seg50.wav"
                      % (pf, spk.lower(), slot)), False),
        ("GS零样本",
         os.path.join(OUT, "cmp-%s_%s_%s_zs_e40_seg50.wav"
                      % (pf, spk.lower(), slot)), False),
        ("Qwen " + qwen_voice,
         os.path.join(QWEN_DIR, "c%s%s-%s" % (pf, slot, qwen_voice),
                   "c%s%s-%s-1.mp3" % (pf, slot, qwen_voice)), False),
    ]
    return [{"name": n, "path": p, "is_male": is_male, "ref": r,
             "text_file": txt} for n, p, r in cands]


def main():
    n = 0
    for spk, is_male, voice, pf in GROUPS:
        for slot in SLOTS:
            spec = build(spk, is_male, voice, pf, slot)
            fn = "cmp_%s_%s.json" % (pf, slot)
            with open(os.path.join(HERE, fn), "w", encoding="utf-8") as f:
                json.dump(spec, f, ensure_ascii=False, indent=2)
            miss = [s["path"] for s in spec if not os.path.exists(s["path"])]
            print("写 %-18s %d 项，缺失 %d %s"
                  % (fn, len(spec), len(miss), miss[:2]))
            n += 1
    print("共 %d 份清单" % n)


if __name__ == "__main__":
    main()
