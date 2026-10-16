#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第30轮：把 fix_chain 的「两次 atempo」融合成一次（幂等）

## 依据（tools/atempo_ab2.py，24k，EL 真人素材）
| 处理 | 平坦度 | 相对 EL 保持率 |
|---|---|---|
| EL 原样 | 0.04928 | 100% |
| 只 atempo=0.988 | 0.04857 | **98.6%** ← 温和拉伸几乎无损 |
| 只重采样（变调） | 0.04361 | 88.5% |
| **A 现役：先 atempo(1/fs) 再 asetrate+atempo(1/R)** | 0.04133 | **83.9%** |
| **B 融合：asetrate+atempo(1/(R·fs))** | 0.04355 | **88.4%** |

## 为什么会浪费一遍
变调 R<1（降调）**本身就把时长拉长 1/R**：
   R=0.92 → 1/0.92 = 1.087×（已经慢了 8.7%）
而语速锚 fs=1.10 正好也要慢 10%。
现役写法是「先 atempo=1/1.10=0.909 放慢，再 atempo=1/0.92=1.087 拉快」
—— 两次 WSOLA 方向相反、几乎互相抵消，等于把语音白揉两遍。
融合后单次 atempo = 1/(R·fs) = 1/(0.92×1.10) = **0.988**，几乎不动。

WSOLA 反复揉捏听感就是**起音被抹、辅音糊掉、过渡发硬**，
与员工反馈的「吃字 / 卡字」吻合。

## 数学等价性校验
  现役：时长 orig × fs × 1 = orig·fs      音高 ×R
  融合：asetrate(R) → orig/R；atempo(1/(R·fs)) → orig/R × (R·fs) = orig·fs ✓
  音高同为 ×R ✓

## 用法
    ./venv/bin/python deploy/apply_fuse30.py
"""
import ast
import os
import re
import sys

P = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"

OLD_TEMPO = '''        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", t1, "-af",
                        "atempo=%.4f" % (1.0 / fs[i]), t2], check=True)'''

NEW_TEMPO = '''        # 🔴 第30轮：变速**不再单独走一遍 atempo**。
        #    变调 R<1 本身就把时长拉长 1/R（R=0.92 → 1.087×），
        #    而语速锚 fs 也是要放慢 —— 两步 WSOLA 方向相反、几乎互相抵消，
        #    等于把语音白揉两遍（现役平坦度保持率 83.9%，融合后 88.4%）。
        #    融合后单次 atempo = 1/(R·fs) ≈ 0.988，几乎不动（实测 98.6% 保持）。
        #    这里保留原始片段，fs 折进变调步的那一次 atempo（见下方 FUSE）。
        # （ratios 在下面才算，这里不能引用；反正两步都不需要拉伸）
        shutil.copyfile(t1, t2)'''

OLD_GATE = '''    if any(abs(r - 1.0) > 1e-6 for r in ratios):
        new_segs = []
        for i, s2 in enumerate(segs):
            R = ratios[i]
            if abs(R - 1.0) < 1e-6:
                new_segs.append(s2)
                continue
            t1 = os.path.join(tmpd, "p%d_in.wav" % i)'''

NEW_GATE = '''    # 🔴 第30轮 FUSE：不管变调比例是不是 1，都要走这一步 ——
    #    因为语速 fs 也折进了这里的 atempo。R==1 且 fs==1 才真正跳过。
    if True:
        new_segs = []
        for i, s2 in enumerate(segs):
            R = ratios[i]
            if abs(R - 1.0) < 1e-6 and abs(fs[i] - 1.0) < 1e-6:
                new_segs.append(s2)       # 既不改速也不变调，原样保留
                continue
            t1 = os.path.join(tmpd, "p%d_in.wav" % i)'''

OLD_AF = '''            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", t1, "-af",
                            "asetrate=%d,aresample=%d,atempo=%.6f"
                            % (int(round(sr * R)), sr, 1.0 / R),
                            t2], check=True)'''

NEW_AF = '''            # 🔴 FUSE：把语速系数 fs[i] 折进这一次 atempo。
            #    asetrate(R) → 时长 orig/R；atempo(1/(R·fs)) → orig/R × R·fs
            #    = orig·fs ✓   音高 ×R ✓   与现役完全等价，但只揉一遍。
            _fuse_t = 1.0 / (R * fs[i]) if fs[i] > 0 else 1.0 / R
            _fuse_t = min(max(_fuse_t, 0.5), 2.0)   # atempo 合法区间
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", t1, "-af",
                            "asetrate=%d,aresample=%d,atempo=%.6f"
                            % (int(round(sr * R)), sr, _fuse_t),
                            t2], check=True)'''

OLD_PRINT = '''    print("分段拉伸后 %.1fs" % (len(buf) / sr), flush=True)'''
NEW_PRINT = '''    print("分段切窗后 %.1fs（变速已折进变调步，此处不再单独拉伸）"
          % (len(buf) / sr), flush=True)'''


def main():
    s = open(P, encoding="utf-8").read()
    if "FUSE" in s and "_fuse_t" in s:
        print("已打过补丁，跳过（幂等）")
        return 0

    n = 0
    for old, new in ((OLD_TEMPO, NEW_TEMPO), (OLD_GATE, NEW_GATE),
                     (OLD_AF, NEW_AF), (OLD_PRINT, NEW_PRINT)):
        if old in s:
            s = s.replace(old, new, 1)
            n += 1
        else:
            print("❌ 未找到片段：%s" % old.split("\n")[0][:60])
    if n != 4:
        print("❌ 只替换了 %d/4 处，放弃写入" % n)
        return 1

    # ratios 必须在变速之前就算出来（融合后变速依赖它）
    if "ratios = [a.pitch]" not in s:
        print("⚠️ 未定位到 ratios 赋值，请人工确认顺序")
    # 语法校验
    try:
        ast.parse(s)
    except SyntaxError as e:
        print("❌ 语法错误: %s" % e)
        return 1
    open(P, "w", encoding="utf-8").write(s)
    print("✅ 已写入 %s（替换 %d 处，语法校验通过）" % (P, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
