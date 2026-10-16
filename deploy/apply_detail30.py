#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第30轮：把「补噪声细节」接入链路（默认关闭，幂等）

插在 declick 同一处（两条分支都会经过；`if _rt_db is None:` 的快路径
会绕过 roomtone，所以不能插在它后面）。

开关：`TTS_DETAIL_K`（浮点，0 = 关闭，标定值 0.06）
未设置 → 完全不改变现网行为。

## 用法
    ./venv/bin/python deploy/apply_detail30.py
"""
import ast
import sys

P = "/mnt/tts/projects/qwen3-tts/app/server.py"

OLD_IMP = "import declick            # 去咔：给数字零的停顿段补环境声（r28）"
NEW_IMP = (OLD_IMP + "\n"
           "import detail             # 补细节：发声区补回送气/摩擦噪声（r30）")

OLD_IF = "                if declick.ENABLED:"
NEW_IF = "                if declick.ENABLED or detail.ENABLED:"

OLD_AP = "                    _dx = declick.apply(_dx, _dsr)"
NEW_AP = (OLD_AP + "\n"
          "                    # 🔴 第30轮：补回发声区的噪声细节（治电子音/吃字/卡字）。\n"
          "                    #    默认关闭（TTS_DETAIL_K 未设），上线前必须过员工 A/B。\n"
          "                    _dx = detail.apply(_dx, _dsr)")


def main():
    s = open(P, encoding="utf-8").read()
    if "import detail " in s and "detail.apply" in s:
        print("已打过补丁，跳过（幂等）")
        return 0
    n = 0
    for old, new in ((OLD_IMP, NEW_IMP), (OLD_IF, NEW_IF), (OLD_AP, NEW_AP)):
        if old in s:
            s = s.replace(old, new, 1)
            n += 1
        else:
            print("❌ 未找到：%s" % old[:60])
    if n != 3:
        print("❌ 只替换 %d/3，放弃" % n)
        return 1
    try:
        ast.parse(s)
    except SyntaxError as e:
        print("❌ 语法错误 %s" % e)
        return 1
    open(P, "w", encoding="utf-8").write(s)
    print("✅ 已写入 %s（%d 处，语法 OK）" % (P, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
