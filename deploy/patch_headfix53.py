#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deploy/patch_headfix53.py —— v3.4 文件头孤立段切除，接入 fix_chain.py。

## 背景（2026-10-20 实测）
员工症状「0秒 开头 怪音」「句首紧贴碎片——文案里没的字也发音」的根因：
模型（omni 女声）在**外语专有名词开头**（로보티즈 类）时，起音抛一段
100-200ms 的垃圾碎片，随后 200-600ms 近静音，才进正文。
  · headrepro-1019 十条：60%（6/10）触发，ASR 对照证明碎片是幻觉级垃圾
    （干净单句里被听成「자」「그러니까」「MBC 뉴스 이덕영입니다」），
    切掉后首词完整、内容零损失。
  · 变量分离：女声+로보티즈 3/3、男声+로보티즈 2/3（10ms 级）、
    女声+正常韩语开头（반갑습니다/아모텍）0~1/3 → 文案触发。
  · 音色 0.9994（切前 vs 切后）、click 0->0、头部结构修复后 = EL 同款。

## 修法（判据四条件全满足才切，宁漏勿误）
  1. S = 持续语音起点（> -25dB 且其后 300ms 均值 > -28dB）
  2. S 之前有 >=150ms 间隙（切点 = 间隙终点）
  3. 头部区域 > -30dB 的碎片组恰好 1 组、总时长 <=350ms、组起点 <0.30s
     —— 「恰好 1 组」排除词内断裂型（首词被撕两段，切了吃字，headrepro #4）
  4. 安全闸：切点后 400ms 内恢复持续正常语音
  任何异常/不满足 -> 原样返回。TTS_HEADFIX=0 关闭。

## 🔴 位置铁律
必须插在 fix_chain.py 的 load16 之后、任何音量增益之前——所有阈值都是
**原始 dBFS 口径**（离线验证全部在原始电平上做）。切完要把 a.raw 重指到
切过的文件，否则下面 detect_gaps_wav(a.raw) 的时间轴会整体晚一个切长，
60s 窗边界全部偏 0.7s。

用法：
    python3 deploy/patch_headfix53.py           # 干跑（只打印将插入的代码）
    python3 deploy/patch_headfix53.py --apply   # 打补丁（自动 .bak + 编译自检 + 失败回滚）
"""
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "app", "fix_chain.py")

FUNC = '''

def _headfix_cut(y, sr):
    """v3.4 文件头孤立段切除（deploy/patch_headfix53.py）。

    缺陷形态：起音即碎片（<350ms 的 >-30dB 段紧贴文件头）+ >=150ms 间隙
    + 持续语音。ASR 对照证明碎片是幻觉级垃圾，切掉首词完整、音色 0.9994、
    click 不变。词内断裂型由「碎片组恰好 1 组」排除。

    🔴 任何异常/判据不满足一律原样返回，绝不让整单失败。
    🔴 必须在 load16 之后、任何增益之前调用（阈值是原始 dBFS 口径）。
    """
    try:
        hop = max(1, int(0.01 * sr))
        n = len(y) // hop
        if n < 100:
            return y, {}
        v = 20 * np.log10(np.sqrt(
            (y[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-12)
        # 1) 持续语音起点：首个 >-25dB 且其后 300ms 均值 >-28dB
        si = None
        for i in range(len(v) - 30):
            if v[i] > -25 and v[i:i + 30].mean() > -28:
                si = i
                break
        if si is None:
            return y, {}
        # 2) 从 S 往前跳过 lead-in，找第一个 >=150ms 间隙（切点 = 间隙终点）
        i = si - 1
        while i >= 0:
            if v[i] > -30:
                j = i
                while j >= 0 and v[j] > -30:
                    j -= 1
                i = j
            else:
                j = i
                while j >= 0 and v[j] <= -30:
                    j -= 1
                if (i - j) * 0.01 >= 0.15:
                    break
                i = j
        if i < 0:
            return y, {}
        cut = (i + 1) * 0.01
        # 3) 头部区域 [0, 断口) 内 >-30dB 碎片组：恰好 1 组、<=350ms、起点 <0.30s
        groups = []
        k = 0
        while k <= i:
            if v[k] > -30:
                m = k
                while m <= i and v[m] > -30:
                    m += 1
                groups.append((k * 0.01, m * 0.01))
                k = m
            else:
                m = k
                while m <= i and v[m] <= -30:
                    m += 1
                k = m
        if len(groups) != 1:
            return y, {}
        if groups[0][1] - groups[0][0] > 0.35 or groups[0][0] > 0.30:
            return y, {}
        # 4) 安全闸：切点后 400ms 内恢复持续正常语音
        k_end = min(len(v), i + 1 + 40)
        if not (i + 1 < len(v) and v[i + 1:k_end].max() > -25):
            return y, {}
        ncut = int(round(cut * sr))
        if ncut <= 0 or ncut >= len(y) - int(0.5 * sr):
            return y, {}
        y2 = y[ncut:].copy()
        fade = min(len(y2), max(1, int(0.005 * sr)))
        y2[:fade] *= np.linspace(0.0, 1.0, fade)
        return y2, {"cut": round(cut, 2), "frag": groups[0],
                    "S": round(si * 0.01, 2)}
    except Exception:                                          # noqa: BLE001
        return y, {}

'''

CALL_ANCHOR = """    y, sr = load16(a.raw)
    dur = len(y) / sr
"""

CALL = """    y, sr = load16(a.raw)
    # 🔴 v3.4 头部孤立段切除（deploy/patch_headfix53.py）：必须在任何增益之前
    #    （阈值是原始 dBFS 口径）。切完把 a.raw 重指到切过的文件——否则下面
    #    detect_gaps_wav(a.raw) 的时间轴会整体晚一个切长，60s 窗边界全偏。
    if os.environ.get("TTS_HEADFIX", "1") != "0":
        y, _hx = _headfix_cut(y, sr)
        if _hx.get("cut"):
            _hd = tempfile.mkdtemp(prefix="headfix_")
            _hp = os.path.join(_hd, "cut.wav")
            sf.write(_hp, y, sr)
            a.raw = _hp
            print("头部孤立段切除 %.2fs（碎片 %.2f-%.2f，持续语音 %.2fs 起）"
                  % (_hx["cut"], _hx["frag"][0], _hx["frag"][1],
                     _hx.get("S") or -1), flush=True)
    dur = len(y) / sr
"""


def main():
    apply = "--apply" in sys.argv
    src = open(TARGET, encoding="utf-8").read()
    if "_headfix_cut" in src:
        print("已打过补丁（_headfix_cut 已存在），跳过")
        return
    if CALL_ANCHOR not in src:
        print("!! 锚点未找到：\n%s" % CALL_ANCHOR)
        sys.exit(1)
    print("将插入函数 %d 行 + 调用 %d 行" % (len(FUNC.splitlines()),
                                            len(CALL.splitlines())))
    if not apply:
        print("---- 干跑：加 --apply 实打 ----")
        print(FUNC)
        print(CALL)
        return
    bak = TARGET + ".bak-before-headfix53-" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copyfile(TARGET, bak)
    print("备份 -> %s" % bak)
    out = src.replace(CALL_ANCHOR, CALL, 1)
    # 函数插在 main() 定义前
    anchor2 = "\ndef main():"
    if anchor2 not in out:
        print("!! main() 锚点未找到，回滚")
        shutil.copyfile(bak, TARGET)
        sys.exit(1)
    out = out.replace(anchor2, FUNC + "\ndef main():", 1)
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
    print("补丁完成")


if __name__ == "__main__":
    main()
