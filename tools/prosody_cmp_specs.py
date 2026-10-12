# -*- coding: utf-8 -*-
"""生成 tools/prosody_cmp.py 用的候选清单（男女各一份）。

⚠️ 参照（ref）必须是**案例未训练区的原声**——即 GPT-SoVITS 微调时
hold 掉的那条案例的前 60 秒。用训练区当参照等于自己跟自己对答案。
"""
import json
import os

GS = "/mnt/tts/projects/gpt-sovits/out"

# 🔴 seg50 = 外部分段推理（每段 ≤50 字）的产物。
# 整段喂给 GS 会触发长文本发散：女声实测覆盖率 0.150 / CER 0.808，
# 分段后 0.956 / 0.045。所以「整段」那一栏只作为**反面参照**保留，
# 上线候选一律看 seg 版。
FEMALE = [
    ("案例原声(参照)", f"{GS}/ref_sola_00b6a45a.wav", True),
    ("GS微调e40+分段", f"{GS}/gs_sola_00b6a45a_ft_e40_seg50.wav", False),
    ("GS微调e40整段", f"{GS}/gs_sola_00b6a45a_ft_e40.wav", False),
    ("GS零样本整段", f"{GS}/gs_sola_00b6a45a_zeroshot.wav", False),
    ("Qwen EL_female", "/mnt/tts/output/gs-cmp-EL_female/gs-cmp-EL_female-1.mp3", False),
    ("Qwen SOLA_female", "/mnt/tts/output/gs-cmp-SOLA_female/gs-cmp-SOLA_female-1.mp3", False),
]

MALE = [
    ("案例原声(参照)", f"{GS}/ref_yohan_1a6c20a6.wav", True),
    ("GS微调e40+分段", f"{GS}/gs_yohan_1a6c20a6_ft_e40_seg50.wav", False),
    ("GS微调e40整段", f"{GS}/gs_yohan_1a6c20a6_ft_e40.wav", False),
    ("GS零样本整段", f"{GS}/gs_yohan_1a6c20a6_zeroshot.wav", False),
    ("Qwen EL_male", "/mnt/tts/output/gs-cmp-m-EL_male/gs-cmp-m-EL_male-1.mp3", False),
    ("Qwen yohan_male", "/mnt/tts/output/gs-cmp-m-yohan_male/gs-cmp-m-yohan_male-1.mp3", False),
]

TEXT = {"f": f"{GS}/text_sola_00b6a45a.txt", "m": f"{GS}/text_yohan_1a6c20a6.txt"}


def build(rows, text_file, is_male, out):
    spec = [{"name": n, "path": p, "ref": r, "is_male": is_male,
             "text_file": text_file} for n, p, r in rows]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(spec, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("写", out, len([1 for s in spec if os.path.exists(s["path"])]),
          "/", len(spec), "个文件存在")


if __name__ == "__main__":
    d = os.path.dirname(os.path.abspath(__file__))
    build(FEMALE, TEXT["f"], False, os.path.join(d, "prosody_cmp_f.json"))
    build(MALE, TEXT["m"], True, os.path.join(d, "prosody_cmp_m.json"))
