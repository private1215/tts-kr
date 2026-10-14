#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把两个「标杆音色」切到新引擎：男 → VoxCPM2，女 → OmniVoice。

只改 engine 字段 + cps_lock（分块换算用），其余（ref_text / label / gender）
一律不动，方便一键回退。
"""
import json
import os
import shutil
import sys

P = "/mnt/tts/data/voices/voices.json"
MAP = {"EL_male": ("voxcpm", 8.37), "EL_female": ("omnivoice", 7.57)}

DRY = "--apply" not in sys.argv
v = json.load(open(P, encoding="utf-8"))
for k, (eng, cps) in MAP.items():
    if k not in v:
        print("!! 音色不存在:", k)
        continue
    old = v[k].get("engine")
    print("%-12s engine %s -> %s | cps_lock %s -> %s"
          % (k, old, eng, v[k].get("cps_lock"), cps))
    if not DRY:
        v[k]["engine"] = eng
        v[k]["cps_lock"] = cps

if DRY:
    print("\n(dry-run，未写入。加 --apply 生效)")
else:
    bak = P + ".bak-before-newengine"
    if not os.path.exists(bak):
        shutil.copy2(P, bak)
    json.dump(v, open(P, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写入，备份:", bak)
