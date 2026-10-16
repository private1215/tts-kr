# -*- coding: utf-8 -*-
"""给月份特殊读法加防御注释：10월=시월 不是 십월。
用户 2026-10-16 明确纠正过一次，防止以后被"规范化"改错。
"""
import shutil, sys
P = "/mnt/tts/projects/qwen3-tts/app/konum.py"
BAK = P + ".bak-before-month47"
src = open(P, encoding="utf-8").read()

OLD = '''# 月份特殊读法
MONTHS = {1: "일월", 2: "이월", 3: "삼월", 4: "사월", 5: "오월", 6: "유월",
          7: "칠월", 8: "팔월", 9: "구월", 10: "시월", 11: "십일월",
          12: "십이월"}'''

NEW = '''# 月份特殊读法
# 🔴🔴 不要按汉字数词"规范化"改这张表！韩语月份有固定异读，改了就读错：
#     6월  = 유월  （不是 육월）
#     10월 = 시월  （不是 십월）  ← 2026-10-16 用户纠正过：改成 십월 是错的
#     11월 = 십일월 / 12월 = 십이월（这两个按汉字数词，没有异读）
#    其余 1/2/3/4/5/7/8/9 月按汉字数词，无特殊情况。
MONTHS = {1: "일월", 2: "이월", 3: "삼월", 4: "사월", 5: "오월", 6: "유월",
          7: "칠월", 8: "팔월", 9: "구월", 10: "시월", 11: "십일월",
          12: "십이월"}'''

if "改成 십월 是错的" in src:
    print("[skip] 已打过补丁"); sys.exit(0)
assert OLD in src, "找不到 MONTHS 锚点"
shutil.copy2(P, BAK)
open(P, "w", encoding="utf-8").write(src.replace(OLD, NEW, 1))
import py_compile; py_compile.compile(P, doraise=True)
print("[ok] 已加防御注释并备份 ->", BAK)
