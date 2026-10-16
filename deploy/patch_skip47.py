# -*- coding: utf-8 -*-
"""第47轮：R/fs 接近 1.0 时整窗跳过 asetrate 三件套。

实测（女声，200s 片段，16k 统一口径）：
  EL 标杆        F0 275.9  静音4-8k 11.68
  引擎裸输出      F0 275.9  静音4-8k 11.72   ← 与 EL 一致，不需要变调
  step3(不变调)  F0 271.2  静音4-8k 11.64
  step4(变调后)  F0 271.2  静音4-8k 13.00   ← F0 没变，只多了 +1.35 杂音

即：女声的变调是**纯亏损**——F0 一点没动，却把高频噪声地板抬上去。
原因：R 不精确等于 1.0，所以原跳过条件（<1e-6）永远不成立，
     asetrate+aresample+atempo 每窗都白跑一遍。

改法：偏离 1.0 在 TTS_PITCH_SKIP（默认 3%，约 0.5 半音，听不出）以内就跳过。
男声 R≈1.11（F0 167 -> 目标 188）不受影响，仍会变调。
"""
import shutil, sys, os, py_compile

P = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"
BAK = P + ".bak-before-skip47"
src = open(P, encoding="utf-8").read()

if "PITCH_SKIP" in src:
    print("[skip] 已打过补丁"); sys.exit(0)

shutil.copy2(P, BAK)
print("已备份 ->", BAK)

OLD = "            if abs(R - 1.0) < 1e-6 and abs(fs_sub[i] - 1.0) < 1e-6:"
assert OLD in src, "找不到跳过判断行"

NEW = ("            # 🔴 第47轮：偏离 1.0 在 PITCH_SKIP 以内就整窗跳过。\n"
       "            #    原阈值 1e-6 太严，R 差一点点也照样跑 asetrate 三件套，\n"
       "            #    实测女声 F0 一点没变（271.2->271.2）却多出 +1.35 的\n"
       "            #    静音段 4-8k 噪声。3% ≈ 0.5 半音，人耳听不出。\n"
       "            if (abs(R - 1.0) < PITCH_SKIP\n"
       "                    and abs(fs_sub[i] - 1.0) < PITCH_SKIP):")
src = src.replace(OLD, NEW, 1)

anchor = 'DEFAULT_F0 = {"f": 276.0, "m": 188.0}'
assert anchor in src
src = src.replace(anchor, anchor +
    '\n\n# 第47轮：变调/变速系数偏离 1.0 在这个范围内就整窗跳过重采样。\n'
    '# 0.03 ≈ 0.5 半音，听不出；但能省掉 asetrate+aresample+atempo 一趟，\n'
    '# 实测女声可把静音段 4-8kHz 从 13.0 降回 11.6（EL 11.7）。\n'
    'PITCH_SKIP = float(os.environ.get("TTS_PITCH_SKIP", "0.03"))', 1)

open(P, "w", encoding="utf-8").write(src)
py_compile.compile(P, doraise=True)
print("[ok] 补丁已写入，语法通过")
