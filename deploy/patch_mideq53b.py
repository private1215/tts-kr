# -*- coding: utf-8 -*-
"""patch_mideq53b.py —— 把**保采样率窄带中频 EQ** 接进生产链（只 omnivoice + 男声）。

## 为什么
现役男声中频只有 **0.86×EL**（质心 0.98 / 中频 0.86 / 杂音 0.94），
这是 v3.1 之后男声唯一剩下的客观差距。L3 全篇实测（pipe_1019g/h）：

    +2dB（窄带 EQ）→ 中频 **1.01** ✅ 完美对齐、click 0.11×、杂音 0.85、质心 1.01
    +3dB          → 中频 1.11（过冲）、click 0.06×、杂音 0.78、质心 1.04

🔴🔴 **为什么不用全频段 LTAS**（pipe_1019e 实测）：
LTAS 要对全频段做多段 FIR，振铃/时域涂抹把 click 从现役的 **0.05×EL 打到 1.72×EL**
（反而比 EL 差 72%）。click 是唯一与员工评分单调对应的指标，且现役本来已比 EL
好 20 倍 → 按铁律「已比 EL 好的不动」，这个代价不能接受。
窄带 firequalizer（1k:0 / 1.8k:+G / 3k:0）涂抹量小得多。

→ 定案 **+2dB**（对齐标杆，不过冲）。

## 只作用 omnivoice + 男声
- 男声中频 0.86 → 要修
- 女声中频已 **1.01**（已对齐，且女声 LTAS 实测无效、click 恶化 19%）→ 不动

## 开关
`TTS_MID_EQ_MALE_DB`（默认 `2`；留空 / `0` = 关闭）

## 🔴🔴 铁律：补丁自检必须用 `compile()`，不能用 `ast.parse()`
`_run_fix_chain` 是 `async def run_job()` 里的**嵌套函数**（4 空格）。
按 0 缩进插 helper 会把外层 async 函数**截断** →
`SyntaxError: 'await' outside async function`，服务启动即崩。
（2026-10-19 真的崩过一次；`ast.parse` 只建语法树、不走符号表，抓不到这个错。）
helper 必须**同缩进**（def 4 空格 / 函数体 8 空格）。

用法: venv/bin/python deploy/patch_mideq53b.py [--check]

## 🔴🔴🔴 三个连环坑（都真的踩过，2026-10-19）
1. **`*` 解包少参数**：写成 `return _clickfix53b_wrap(*_mideq53b_wrap(...))`，
   而 `_mideq53b_wrap` 返回 `(audio, sr)` 二元组，解包后只够 audio+sr，
   **缺 gender/engine_name** → TypeError → 整块回退 raw，**EQ 和 clickfix 全都没跑**
   （L3 实测中频仍 0.85、click 1.18×EL）。
   → 修正：本函数**只返回 audio**，调用处显式传 `sr2, _g, _engine_name`。
2. **FIR 输出变长**：`if len(_y) != len(audio): return audio` 判等，
   而 firequalizer 输出比输入长 320 样本（16k，20ms）→ **每次都退回**，
   EQ 永远不生效（L1 实测中频仍 0.83×EL，靠临时日志 `MIDDBG` 才看到）。
   → 修正：**居中裁剪**回原长（线性相位 FIR 群延迟对称，居中即对齐），
     只有差得离谱（>0.5s）才真退回。
3. **audit53 只查字符串在不在，查不到"没被执行"**：上面两个坑它全绿。
   → 必须跑 `diag/l1_mideq53.py` 实测中频是否抬到 ≈1.0×EL。
"""
import argparse
import os
import sys
from datetime import datetime

TARGET = "/mnt/tts/projects/qwen3-tts/app/server.py"

ANCHOR_DEF = "    def _run_fix_chain(audio, sr, text):"
ANCHOR_RET = "            return _clickfix53b_wrap(y2, sr2, _g, _engine_name)"
NEW_RET = ("            return _clickfix53b_wrap(\n"
           "                _mideq53b_wrap(y2, sr2, _g, _engine_name),\n"
           "                sr2, _g, _engine_name)")

# 🔴 def 4 空格、函数体 8 空格 —— 与 `_clickfix53b_wrap` 同级
# 🔴🔴 2026-10-19 第一版踩的坑：写成 `return _clickfix53b_wrap(*_mideq53b_wrap(...))`，
#    而本函数返回 `(audio, sr)` 二元组，`*` 解包后只够 audio+sr 两个位置参数，
#    **缺 gender/engine_name** → `TypeError` → 整块回退 raw，**EQ 和 clickfix
#    全都没跑**（L3 实测中频仍 0.85、click 1.18×EL，日志里能直接看到这个 TypeError）。
#    → 修正：本函数**只返回 audio**（sr 不变），调用处显式传 sr2, _g, _engine_name。
HELPER = '''    def _mideq53b_wrap(audio, sr, gender, engine_name, _tag=""):
        """保采样率**窄带中频 EQ**（1k:0 / 1.8k:+G / 3k:0）。只 omni+男声。
        **只返回 audio**（sr 不变）——🔴 别返回 (audio, sr) 再 `*` 解包，
        那会让外层 _clickfix53b_wrap 少收 gender/engine_name 两个参数。
        任何异常一律原样返回 audio，绝不让整单失败。"""
        try:
            if engine_name != "omnivoice" or (gender or "").lower()[:1] != "m":
                return audio
            _g = str(os.environ.get("TTS_MID_EQ_MALE_DB", "2") or "").strip()
            if not _g or float(_g) == 0:
                return audio
            _g = float(_g)
            import subprocess as _sp
            import tempfile as _tf
            import soundfile as _sf
            _d = _tf.mkdtemp(prefix="mideq53b_")
            _src = os.path.join(_d, "in.wav")
            _dst = os.path.join(_d, "out.wav")
            _sf.write(_src, audio, sr)
            _sp.run(["ffmpeg", "-y", "-v", "error", "-i", _src, "-af",
                     "firequalizer=gain_entry='entry(1000,0);"
                     "entry(1800,%g);entry(3000,0)'" % _g,
                     "-ar", str(sr), "-ac", "1", _dst],
                    check=True, timeout=300)
            _y, _ = _sf.read(_dst, dtype="float32")
            if getattr(_y, "ndim", 1) > 1:
                _y = _y.mean(axis=1)
            # 🔴🔴 firequalizer 是 **FIR**，输出比输入**长** filter_size-1 个样本
            #   （16k 实测长 320 样本 = 20ms）。第二版用 `if len(_y) != len(audio):
            #   return audio` 判等，**每次都触发退回** → EQ 永远不生效。
            #   L1 实测中频仍 0.83×EL，加临时日志才看到
            #   `MIDDBG ffmpeg后 len(_y)=206400 len(audio)=206080 → 早退: 长度不一致`。
            #   → 改成**居中裁剪**回原长：线性相位 FIR 的群延迟是对称的，
            #     居中即对齐，总时长不变。
            #   → 只有差得离谱（>0.5s）才真退回，防止 ffmpeg 出怪东西。
            _extra = len(_y) - len(audio)
            if _extra < 0 or _extra > int(0.5 * sr):
                return audio
            if _extra:
                _o = _extra // 2
                _y = _y[_o:_o + len(audio)]
            return _y
        except Exception:
            return audio

'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只自检不落盘")
    a = ap.parse_args()

    if not os.path.exists(TARGET):
        print("❌ %s 不存在" % TARGET)
        return 1
    src = open(TARGET, encoding="utf-8").read()

    if "_mideq53b_wrap" in src:
        print("已打过补丁（_mideq53b_wrap 已在）。")
        print("  若这是第一版（有 `*_mideq53b_wrap` 解包 bug），"
              "请先从 .bak-before-mideq53b-* 恢复 server.py 再重打。")
        return 0
    if ANCHOR_DEF not in src:
        print("❌ 找不到锚点 %r（server.py 结构变了？）" % ANCHOR_DEF)
        return 1
    if ANCHOR_RET not in src:
        print("❌ 找不到锚点 %r" % ANCHOR_RET)
        return 1

    new = src.replace(ANCHOR_DEF, HELPER + ANCHOR_DEF, 1)
    new = new.replace(ANCHOR_RET, NEW_RET, 1)
    assert new != src

    # 🔴🔴 必须 compile()：ast.parse 抓不到 await 作用域错误
    try:
        compile(new, TARGET, "exec")
    except SyntaxError as e:
        print("❌ 补丁后语法检查失败：line %s %s" % (e.lineno, e.msg))
        print("   很可能又是缩进问题（helper 必须与 _run_fix_chain 同为 4 空格）")
        return 1
    print("✅ compile() 语法自检通过")
    # 顺带确认缩进正确
    i = new.index("    def _mideq53b_wrap")
    assert new[i - 5:i] != "\n\n\n\n", "缩进异常"
    print("✅ helper 缩进 = 4 空格（与 _run_fix_chain 同级）")

    if a.check:
        print("[check] 未落盘")
        return 0

    bak = TARGET + ".bak-before-mideq53b-" + datetime.now().strftime(
        "%Y%m%d_%H%M%S")
    with open(bak, "w", encoding="utf-8") as f:
        f.write(src)
    tmp = TARGET + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(new)
    os.replace(tmp, TARGET)
    print("✅ 已落盘（备份 %s）" % os.path.basename(bak))
    print("🔴 现在必须重启 qwen-tts 才生效")
    print("🔴 环境变量 TTS_MID_EQ_MALE_DB 默认 2；要关就设为空或 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
