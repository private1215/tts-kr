# -*- coding: utf-8 -*-
"""cut_voice53b.py —— 修正版：按 voices.json **实际 schema** 补回 elara / yo han。

## 上一版（cut_voice53.py）踩的坑
1. 🔴 voices.json 是**扁平字典**（键就是音色名），**没有** `{"voices": ...}` 包装。
   上一版写了 `d["voices"] = vs`，凭空多出一个叫 "voices" 的假音色
   （audit53 报 `FAIL voices -> None`）。
2. 🔴 字段名不匹配：实际用 `path` / `dur` / `cps` / `cps_lock` / `label`，
   不是 `ref_audio` / `duration`。字段不对服务端可能读不出来。
→ 本版**从备份 `voices.json.bak-before-cutvoice53-*` 起步**，按 yohan_male 的结构补。

## 内容（ASR 已验证，见 cut_voice53.py 输出）
| 音色 | 源 | 窗 | 字数 | cps |
|---|---|---|---|---|
| elara | `a3d39a095bac/seg0.wav` | 14.36~22.50 (8.14s) | 62 | 7.62（EL女 7.45）|
| yo han | `799e4d7060ce/seg0.wav` | 136.02~144.68 (8.66s) | 76 | 8.78（EL男 8.37）|

🔴 写完后**必须重启 qwen-tts**，否则任务结束时内存音色表写回会冲掉。

用法: venv/bin/python tools/cut_voice53b.py --apply
"""
import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime

V = "/mnt/tts/data/voices/voices.json"

ADD = {
    "elara": {
        "path": "/mnt/tts/data/voices/elara.wav",
        "label": {"zh": "elara 女声（案例重建）", "en": "elara (female) (case)",
                  "ko": "elara 여성 음성 (케이스)",
                  "ja": "elara 女性ボイス（ケース）"},
        "ref_text": ("주가를 위아래로 마구 흔들면서 따라 붙는 개미와 차익을 "
                     "내려낸 개미를 동시에 털어내는 과정이 나올 수밖에 없어요"),
        "sr": 44100, "dur": 8.14, "engine": "qwen", "gender": "female",
        "bench_case": "a3d39a095bac", "origin": "cases",
        "src_case": "a3d39a095bac", "src_speaker": "elara",
        "src_dur": 173.087, "clip": [14.36, 22.5],
        "cps_lock": 7.45, "cps": 7.62,
    },
    "yo han": {
        "path": "/mnt/tts/data/voices/yo_han.wav",
        "label": {"zh": "yo han 男声（案例重建）", "en": "yo han (male) (case)",
                  "ko": "yo han 남성 음성 (케이스)",
                  "ja": "yo han 男性ボイス（ケース）"},
        "ref_text": ("주리니 분들도 바로 써먹을 수 있게 떠먹여드리고 있어요 "
                     "그렇게 잘 알면 너 혼자 꿀빨지 왜 공짜로 푸냐 이러시는 "
                     "분들 분명 계실 겁니다"),
        "sr": 44100, "dur": 8.66, "engine": "qwen", "gender": "male",
        "bench_case": "799e4d7060ce", "origin": "cases",
        "src_case": "799e4d7060ce", "src_speaker": "yohan",
        "src_dur": 261.878, "clip": [136.02, 144.68],
        "cps_lock": 8.37, "cps": 8.78,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    for name, v in ADD.items():
        ok = os.path.exists(v["path"])
        print("%-8s 切片 %s  %s" % (name, v["path"],
                                    "OK" if ok else "❌不存在"))
    if not a.apply:
        print("\n[dry-run] 加 --apply 才写入")
        return 0

    # 从最近的 cutvoice53 备份起步（那是写入前的干净状态）
    baks = sorted(glob.glob(V + ".bak-before-cutvoice53-*"))
    if not baks:
        print("❌ 找不到 cutvoice53 备份")
        return 1
    src = baks[-1]
    shutil.copy2(src, V)
    d = json.load(open(V))
    print("\n从 %s 恢复基线，keys = %s" % (os.path.basename(src),
                                       sorted(d.keys())))
    if "voices" in d:
        del d["voices"]
    for name, v in ADD.items():
        d[name] = v
    bak = V + ".bak-before-cutvoice53b-" + datetime.now().strftime(
        "%Y%m%d_%H%M%S")
    shutil.copy2(V, bak)
    tmp = V + ".tmp"
    json.dump(d, open(tmp, "w"), ensure_ascii=False, indent=2)
    os.replace(tmp, V)

    d2 = json.load(open(V))
    print("\n✅ 已写入（备份 %s）" % os.path.basename(bak))
    print("   音色数 = %d：%s" % (len(d2), ", ".join(sorted(d2.keys()))))
    assert "voices" not in d2, "不该再有 voices 包装键"
    for name in ADD:
        assert name in d2 and "path" in d2[name], name
    print("   自检通过（无 voices 假键，字段齐全）")
    print("🔴 现在必须重启 qwen-tts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
