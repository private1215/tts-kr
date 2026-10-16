# -*- coding: utf-8 -*-
"""修正 v1 补丁：只跳过**变调**，变速必须保留。

v1 的问题：原跳过分支是「既不变调也不变速」，而女声 fs≈1.02（需要变速 2%），
一起跳掉导致时长从 200.0s 变成 203.1s（语速偏离目标），杂音也没降下来
（12.997 -> 13.269）。

正确逻辑：
  R≈1 且 fs≈1  -> 原样保留
  R≈1 但 fs≠1  -> **只跑 atempo**（不变调，省掉 asetrate+aresample）
  R≠1          -> 原 asetrate 三件套（融合 fs）
"""
import shutil, sys, py_compile

P = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"
src = open(P, encoding="utf-8").read()

if "只跑 atempo" in src:
    print("[skip] 已是 v2"); sys.exit(0)
shutil.copy2(P, P + ".bak-before-skip47b")

OLD = """            if (abs(R - 1.0) < PITCH_SKIP
                    and abs(fs_sub[i] - 1.0) < PITCH_SKIP):
                new_segs.append(s2)       # 既不改速也不变调，原样保留
                continue"""
assert OLD in src, "找不到 v1 的判断块"

NEW = """            if abs(R - 1.0) < PITCH_SKIP:
                # 🔴 第47轮 v2：不需要变调时**只做变速**，绝不跑 asetrate 三件套
                #    （那是静音段 4-8k 噪声的唯一来源，实测 F0 一点没变纯亏损）。
                #    ⚠️ v1 把变速也一起跳了，导致语速偏离目标 —— 这里分开处理。
                if abs(fs_sub[i] - 1.0) < PITCH_SKIP:
                    new_segs.append(s2)   # 既不变调也不变速
                    continue
                t1 = os.path.join(tmpd, "q%d_in.wav" % i)
                t2 = os.path.join(tmpd, "q%d_out.wav" % i)
                sf.write(t1, s2, sr)
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", t1,
                                "-af", "atempo=%.6f" % fs_sub[i], t2],
                               check=True)
                s3, _ = sf.read(t2, dtype="float32")
                new_segs.append(s3)
                continue"""
src = src.replace(OLD, NEW, 1)
open(P, "w", encoding="utf-8").write(src)
py_compile.compile(P, doraise=True)
print("[ok] v2 补丁已写入，语法通过")
