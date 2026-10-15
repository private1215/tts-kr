#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 declick 接入 server.py 的编码链（一次性脚本，跑完即可删）。

插入点：`roomtone.apply()` 之后、`libmp3lame` 之前。
🔴 位置不能变：陷阱 #18 —— 链尾 libmp3lame 会重新定一遍电平，
   放在它之前做的处理会被重置。
"""
import ast
import sys

P = "/mnt/tts/projects/qwen3-tts/app/server.py"

OLD_IMP = ("import peakfit            # 峰值定点：把真峰值软削到案例的水平"
           "（r23，已证伪关）")
NEW_IMP = (OLD_IMP + "\nimport declick           # 去咔：给极低电平段补环境声，"
           "消除「瞬间卡顿」（第28轮）")

OLD_CALL = "                _nx = roomtone.apply(_nx, _nsr, _rt_db)"
NEW_CALL = (
    OLD_CALL + "\n"
    "                # 🔴 2026-10-15 第28轮：员工反馈「还是存在声音卡顿」。\n"
    "                #    实测 EL 标杆硬切 0 处，我们 10 处（旧引擎 22 处），\n"
    "                #    原因是我们输出的停顿电平低到 -85~-146dBFS（近乎绝对零），\n"
    "                #    而真人停顿是有环境声的。补一层 -70dB 极轻噪声即可消除。\n"
    "                #    必须在 roomtone 之后、libmp3lame 之前（陷阱 #18）。\n"
    "                if declick.ENABLED:\n"
    "                    _nx = declick.apply(_nx, _nsr)")


def main():
    src = open(P, encoding="utf-8").read()
    if "import declick" in src:
        print("已经接过 declick 了，跳过")
        return 0
    if OLD_IMP not in src:
        print("🔴 导入锚点没找到，server.py 结构可能变了")
        return 1
    src = src.replace(OLD_IMP, NEW_IMP, 1)
    if OLD_CALL not in src:
        print("🔴 roomtone 调用锚点没找到")
        return 1
    src = src.replace(OLD_CALL, NEW_CALL, 1)
    ast.parse(src)                       # 先验语法再写
    open(P, "w", encoding="utf-8").write(src)
    print("✅ 已插入 import + declick 调用，语法校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())