#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""男声 EQ 加 250Hz -0.75dB —— 修「感冒闷」的客观对应项。

现役男声成品 150~300Hz = +3.96σ（8197 与 6cdd25 两篇**完全一致**，系统性），
与员工 j 轮 `m_cold` 选「我们像感冒鼻塞」精确对应。
两篇交叉验证：250Hz -0.75dB 让 Σz² 44.1->22.9 / 46.9->21.3，150~300 归零，
且其他项全部不恶化；-1.5dB 会过冲（Σz² 反而升到 44.9 / 33.7）。

🔴 滤波器必须放在**最后**（1400 之后）：-0.75dB 这个量是在**成品**上实测的
   增量，串联顺序变了数值就不对。
"""
import sys

P = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"

OLD_B = ('EQ_MALE_BOOSTS = ("equalizer=f=90:t=q:w=1.1:g=7.5,"\n'
         '                  "equalizer=f=350:t=q:w=1.1:g=3,"\n'
         '                  "equalizer=f=1400:t=q:w=1.1:g=3,lowpass=f=14000")')
NEW_B = ('# 🔴 末尾的 250Hz -0.75dB（2026-10-15）：修 150~300Hz +3.96σ 的「感冒闷」。\n'
         '#    它是**系统性**的（8197 与 6cdd25 两篇都是 +3.96σ），根源是上面\n'
         '#    90Hz +7.5dB / 350Hz +3dB 两个低频 boost 把这段抬过头了。\n'
         '#    必须放在最后：数值是在成品上实测的增量。\n'
         'EQ_MALE_BOOSTS = ("equalizer=f=90:t=q:w=1.1:g=7.5,"\n'
         '                  "equalizer=f=350:t=q:w=1.1:g=3,"\n'
         '                  "equalizer=f=1400:t=q:w=1.1:g=3,"\n'
         '                  "equalizer=f=250:t=q:w=1.1:g=-0.75,lowpass=f=14000")')

OLD_T = ('EQ_MALE_TMPL = ("equalizer=f=90:t=q:w=1.1:g=7.5,"\n'
         '                "equalizer=f=350:t=q:w=1.1:g=3,"\n'
         '                "equalizer=f=1400:t=q:w=1.1:g=3,"\n'
         '                "highshelf=f=4000:g=%g,lowpass=f=14000")')
NEW_T = ('EQ_MALE_TMPL = ("equalizer=f=90:t=q:w=1.1:g=7.5,"\n'
         '                "equalizer=f=350:t=q:w=1.1:g=3,"\n'
         '                "equalizer=f=1400:t=q:w=1.1:g=3,"\n'
         '                "equalizer=f=250:t=q:w=1.1:g=-0.75,"\n'
         '                "highshelf=f=4000:g=%g,lowpass=f=14000")')

s = open(P, encoding="utf-8").read()
open(P + ".bak-before-250hz", "w", encoding="utf-8").write(s)

for old, new, tag in ((OLD_B, NEW_B, "BOOSTS"), (OLD_T, NEW_T, "TMPL")):
    if old not in s:
        sys.exit("没找到 %s，文件已被改过？" % tag)
    s = s.replace(old, new)
    print("patched", tag)

open(P, "w", encoding="utf-8").write(s)
print("written", P)
