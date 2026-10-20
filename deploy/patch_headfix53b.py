#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deploy/patch_headfix53b.py —— v3.5 headfix 判据收紧（修 v3.4 误切实词）。

## v3.4 的缺陷（2026-10-20 对照实验发现）
判据「碎片组 <=350ms + 间隙 >=150ms」会把**短实词**当碎片切掉：
  · `한 퍼센트 정도 올랐습니다` seed42：首词「한」= 170ms + 其后 150ms 停顿，
    四项判据全中 -> 切掉，ASR 由「한 퍼센트 …」变成「퍼센트 …」= **吃掉一个实词**。
  · 决定性证据：TTS_HEADFIX=0 重跑同一条，dur 4.4s 且 ASR 完整含「한」；
    开启态 dur 4.18s 且「한」消失。

## 新阈值怎么定的（13 个样本，无一批判错）
在 headrepro-1019 十条（v3.4 之前的未切除原始产物）+ 病态数字实验 6 单上，
量每个头部语音段的 (时长, 间隙)：

    真碎片          20ms/510ms   20ms/490ms   60ms/270ms
                    70ms/560ms  150ms/630ms   30ms/410ms  20ms/260ms
    真实首词/正常    90ms/120ms  100ms/140ms  150ms/170ms  170ms/150ms
                    50ms/190ms  100ms/ 50ms  110ms/ 50ms  130ms/ 40ms

→ **间隙时长**是唯一干净的分界：真碎片 >=260ms，真实首词 <=190ms。
  碎片时长不能单独用（真碎片 150ms vs 真词 90ms 有交叉）。
  gapmin_db 无区分度（-52dB 两边都有）。

## 改动
  间隙阈值 150ms -> **250ms**
  碎片组上限 350ms -> **200ms**（真碎片最大 150ms，留 50ms 余量；
    真词 170ms 虽仍过线，但被间隙挡住，双重保险）
其余三项（碎片组恰好 1 组 / 起点 <0.30s / 切点后 400ms 恢复语音）不变。

用法：
    python3 deploy/patch_headfix53b.py           # 干跑
    python3 deploy/patch_headfix53b.py --apply   # 打补丁（.bak + 编译自检 + 失败回滚）
"""
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "app", "fix_chain.py")

REPL = [
    # 1) 间隙阈值 150ms -> 250ms（真碎片 >=260ms / 真实首词 <=190ms）
    ("                if (i - j) * 0.01 >= 0.15:",
     "                if (i - j) * 0.01 >= 0.25:"),
    # 2) 碎片组上限 350ms -> 200ms
    ("        if groups[0][1] - groups[0][0] > 0.35 or groups[0][0] > 0.30:",
     "        if groups[0][1] - groups[0][0] > 0.20 or groups[0][0] > 0.30:"),
    # 3) 文档同步
    ('    """v3.4 文件头孤立段切除（deploy/patch_headfix53.py）。',
     '    """v3.5 文件头孤立段切除（deploy/patch_headfix53.py + patch_headfix53b.py）。'),
    ("    缺陷形态：起音即碎片（<350ms 的 >-30dB 段紧贴文件头）+ >=150ms 间隙",
     "    缺陷形态：起音即碎片（<200ms 的 >-30dB 段紧贴文件头）+ >=250ms 间隙\n"
     "    🔴 v3.5：间隙阈值 150->250ms（v3.4 会把「한 퍼센트」的实词「한」\n"
     "      170ms+150ms 当碎片切掉，实测吃掉首词；13 样本上 250ms 是干净分界）"),
]


def main():
    apply = "--apply" in sys.argv
    src = open(TARGET, encoding="utf-8").read()
    missing = [a for a, _ in REPL if a not in src]
    if missing:
        print("!! 以下锚点未找到（补丁可能已打过或代码已变）：")
        for m in missing:
            print("   %r" % m[:70])
        sys.exit(1)
    out = src
    for a, b in REPL:
        out = out.replace(a, b, 1)
    print("将替换 %d 处" % len(REPL))
    if not apply:
        print("---- 干跑：加 --apply 实打 ----")
        return
    bak = TARGET + ".bak-before-headfix53b-" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copyfile(TARGET, bak)
    print("备份 -> %s" % bak)
    open(TARGET, "w", encoding="utf-8").write(out)
    # 🔴 自检必须 compile()，ast.parse() 不走符号表抓不到作用域错误
    import py_compile
    try:
        py_compile.compile(TARGET, doraise=True)
        print("编译自检通过")
    except Exception as e:
        print("!! 编译失败，回滚: %s" % e)
        shutil.copyfile(bak, TARGET)
        sys.exit(1)
    print("补丁完成：gap 150->250ms，frag 350->200ms")


if __name__ == "__main__":
    main()
