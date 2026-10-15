#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修正上一版补丁：250Hz 衰减不能放在 BOOSTS 里。

上一版把 equalizer=f=250:g=-0.75 加进了 EQ_MALE_BOOSTS，结果 Σz² 44.1 -> **78.4**（更差）。
原因：自适应高频是 **按 boost 之后、highshelf 之前的 t5 实测 b4000_8000 反算**的
（`adaptive_hf_db(t5)`）。把衰减插在中间会改变 t5 的频谱，于是 highshelf 算出来的
补偿量跟着变，500~1k 被推到 +5.80σ、4k~8k 到 +4.31σ。

✅ 正确位置：**自适应之后、volume 之前**。这样不影响 adaptive_hf_db 的输入，
   也等价于「在成品上叠加」的实测顺序（-0.75dB 就是那么量出来的）。
   响度会被链尾的 lufs 二次校正吸收。
"""
import sys

P = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"
s = open(P, encoding="utf-8").read()
open(P + ".bak-before-250hz-v2", "w", encoding="utf-8").write(s)

# 1) BOOSTS 恢复原样（去掉 250Hz）
CUR_B = ('EQ_MALE_BOOSTS = ("equalizer=f=90:t=q:w=1.1:g=7.5,"\n'
         '                  "equalizer=f=350:t=q:w=1.1:g=3,"\n'
         '                  "equalizer=f=1400:t=q:w=1.1:g=3,"\n'
         '                  "equalizer=f=250:t=q:w=1.1:g=-0.75,lowpass=f=14000")')
ORIG_B = ('EQ_MALE_BOOSTS = ("equalizer=f=90:t=q:w=1.1:g=7.5,"\n'
          '                  "equalizer=f=350:t=q:w=1.1:g=3,"\n'
          '                  "equalizer=f=1400:t=q:w=1.1:g=3,lowpass=f=14000")')
if CUR_B not in s:
    sys.exit("没找到上一版 BOOSTS，文件状态不对")
s = s.replace(CUR_B, ORIG_B)
print("reverted BOOSTS")

# 2) 新增常量，插在自适应之后
ANCHOR = 'HF_FREQ = 4000\n'
CONST = ('# 🔴 250Hz 衰减（2026-10-15）：修男声 150~300Hz +3.96σ 的「感冒闷」。\n'
         '#    两篇交叉验证（8197 / 6cdd25 都是 +3.96σ，系统性）：-0.75dB 让\n'
         '#    Σz² 44.1->22.9 / 46.9->21.3，150~300 归零且其他项不恶化；\n'
         '#    -1.5dB 过冲（Σz² 反而升到 44.9 / 33.7）。\n'
         '#    ⚠️ 必须放在**自适应高频之后**：自适应是按 boost 后的频谱反算\n'
         '#       highshelf 的，衰减插在前面会把那个计算带偏（实测 Σz² -> 78.4）。\n'
         'MALE_250 = "equalizer=f=250:t=q:w=1.1:g=-0.75,"\n')
if ANCHOR not in s:
    sys.exit("没找到 HF_FREQ 锚点")
s = s.replace(ANCHOR, CONST + ANCHOR, 1)
print("added MALE_250")

# 3) 自适应分支的 af 里插入
OLD_AF = ('        af = ("highshelf=f=%d:g=%g," % (HF_FREQ, hf_db)) '
          '+ "volume=%gdB" % gain')
NEW_AF = ('        # 🔴 250Hz 衰减放在**自适应之后**：不影响 adaptive_hf_db(t5) 的输入\n'
          '        af = (("highshelf=f=%d:g=%g," % (HF_FREQ, hf_db)) + MALE_250\n'
          '             + "volume=%gdB" % gain)')
if OLD_AF not in s:
    sys.exit("没找到自适应分支的 af，可能被改过")
s = s.replace(OLD_AF, NEW_AF)
print("patched adaptive af")

# 4) 非自适应分支也补上（TMPL 已含 250Hz，这里不再重复）
open(P, "w", encoding="utf-8").write(s)
print("written", P)
