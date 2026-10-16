# -*- coding: utf-8 -*-
"""patch_cpsfloor48.py —— 放开 fix_chain 时长对齐的**加速**能力（男声语速根因）。

## 问题
`app/fix_chain.py` 第 ~309 行：

    fs = [min(max(f * (r / mean_r) if r > 0 else f, 1.0), 1.35) for r in rc]

`max(..., 1.0)` 把每窗系数**下限钳在 1.0**。而 fs 的语义是「时长 ×fs」，
fs > 1 = 拉慢，**fs < 1 = 加速**。所以这条链路**只能拉慢，永远不能加速**。

后果（2026-10-16 实测）：

| 引擎 | 裸输出 cps | 目标 cps | 需要 | 实际生效 | 成品 cps |
|---|---|---|---|---|---|
| voxcpm（男） | 8.94 | 8.37 | 拉慢 1.068× | ✅ 生效 | 8.36 |
| **gptsovits（男）** | **7.45** | 8.37 | **加速 0.89×** | ❌ 被钳成 1.0 | **7.45** |

→ 男声 GS 明明音色 0.9576（EL 同人上限 0.9628）、杂音 11.077≈EL 10.968、
  click 0.388% 远优于 voxcpm 的 1.175%，却被语速卡住上不了线
  （7.45 比 EL 8.23 **慢 9.5%**）。

## 修复
把下限从写死的 1.0 改成可配置的 `CPS_FLOOR`（默认 **0.80**，与 server.py 的
`CPS_ATEMPO_LO=0.80` 同一档，避免把语速拉到荒唐的程度）。

只动 `--target-cps` 分支（NEW_ENGINES 快路径走这条）；
`--target-rate` 的 else 分支保持原样，避免波及 legacy 链（qwen/cosy）。

## 已验证
对 GS 裸输出单独跑 atempo 1.105（= cps 7.45→8.23）实测：
杂音 4-8k 13.213 → **7.451**，click 0.368% → 0.505%
—— **加速不恶化杂音**（WSOLA 时域压缩反而平滑了高频瞬态），
click 略升但仍远低于 voxcpm 的 1.175%。

开关：`TTS_CPS_FLOOR=1.0` 可回到旧行为（只拉慢不加速）。
"""
import io
import os
import re
import sys

PATH = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"
BAK = PATH + ".bak-before-cpsfloor48"

OLD = '''        fs = [min(max(f * (r / mean_r) if r > 0 else f, 1.0), 1.35) for r in rc]'''

NEW = '''        # 🔴 第47轮 v1.2：下限从写死的 1.0 放开到 CPS_FLOOR。
        #    fs 语义是「时长 ×fs」，fs<1 = **加速**。原写法 max(...,1.0) 让
        #    这条链路只能拉慢、永远不能加速 —— 男声 GS 裸输出 7.45 字/秒、
        #    目标 8.37，需要加速 0.89× 却被钳成 1.0，成品就一直停在 7.45
        #    （比 EL 8.23 慢 9.5%），白白浪费它 0.9576 的音色。
        #    默认 0.80 与 server.py 的 CPS_ATEMPO_LO 同一档。
        #    回退：TTS_CPS_FLOOR=1.0
        fs = [min(max(f * (r / mean_r) if r > 0 else f, CPS_FLOOR), 1.35)
              for r in rc]'''

ANCHOR = '''CPS_ATEMPO_LO'''
# 插在 DEFAULT_CPS 附近：找一个稳定的模块级常量定义行
CONST_OLD = '''HF_FREQ = 4000'''
CONST_NEW = '''# 🔴 第47轮 v1.2：--target-cps 分支每窗系数的**下限**。
#    fs<1 = 加速。原来是写死的 1.0（= 只能拉慢，永远不能加速），
#    男声 GS 因此卡在 7.45 字/秒上不了线。见 deploy/patch_cpsfloor48.py。
CPS_FLOOR = float(os.environ.get("TTS_CPS_FLOOR", "0.80"))

HF_FREQ = 4000'''


def main():
    src = io.open(PATH, encoding="utf-8").read()

    if "CPS_FLOOR" in src:
        print("已打过补丁，跳过")
        return 0
    if OLD not in src:
        print("❌ 没找到目标 fs 赋值行")
        return 1
    if CONST_OLD not in src:
        print("❌ 没找到常量插入锚点 HF_FREQ")
        return 1
    if "\nimport os\n" not in src and "\nimport os" not in src:
        print("❌ fix_chain.py 没有 import os，CPS_FLOOR 取不到环境变量")
        return 1

    if not os.path.exists(BAK):
        io.open(BAK, "w", encoding="utf-8").write(src)
        print("备份 ->", BAK)

    out = src.replace(OLD, NEW, 1)
    out = out.replace(CONST_OLD, CONST_NEW, 1)
    io.open(PATH, "w", encoding="utf-8").write(out)
    print("✅ 已写入", PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
