#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""切换 EL_male 的引擎（voxcpm → omnivoice），带备份与回滚提示。

为什么换（第26轮 s 轮 3 份答卷完全一致）
    s1 ① 完全不能接受  s2 ① 去掉电子感  s3 **OmniVoice**
→ 三人无一例外，且这是项目至今一致性最高的一轮。

🔴 切换前已做的验证（按用户铁律「短文案验收没意义」）
    没有直接用 161 秒的 eng-omnivoice 样本就换，
    而是先生成 **4143 字 / 553 秒真长文** 测：
      Σz²=166.0（8 个维度 6 个达标）
      沿时间轴 4k~8k 极差 **81.2%**（现役男声 94.9%）
    → 数据支持切换，才执行。

用法：
    echo '<sudo密码>' | sudo -S -p "" python3 swap_engine.py            # 换
    echo '<sudo密码>' | sudo -S -p "" python3 swap_engine.py --rollback # 回滚
"""
import argparse
import json
import shutil
import sys
import time

P = "/mnt/tts/data/voices/voices.json"
BAK = P + ".bak-before-engineswap"
OLD, NEW = "voxcpm", "omnivoice"


def read():
    with open(P, encoding="utf-8") as f:
        return json.load(f)


def write(d):
    with open(P, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollback", action="store_true")
    a = ap.parse_args()

    if a.rollback:
        if not os.path.exists(BAK):
            sys.exit("没有备份 %s，无法回滚" % BAK)
        shutil.copyfile(BAK, P)
        print("已回滚，EL_male.engine =", read()["EL_male"].get("engine"))
        return

    import os                                   # noqa: F811
    cur = read()
    before = cur["EL_male"].get("engine")
    print("换前 EL_male.engine =", before)
    if before == NEW:
        print("已经是 %s，无需切换" % NEW)
        return
    if not os.path.exists(BAK):
        shutil.copyfile(P, BAK)
        print("已备份 ->", BAK)
    cur["EL_male"]["engine"] = NEW
    write(cur)
    print("换后 EL_male.engine =", read()["EL_male"].get("engine"))
    print("\n请执行：systemctl restart qwen-tts")
    print("回滚：python3 %s --rollback && systemctl restart qwen-tts" % sys.argv[0])


if __name__ == "__main__":
    import os
    main()