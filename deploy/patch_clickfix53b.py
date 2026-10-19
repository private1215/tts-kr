# -*- coding: utf-8 -*-
"""第53轮 b：把**保采样率**的 clickfix 接进生产链（仅男声）。

## 为什么现在能接
click 是唯一与员工评分**单调对应**的客观指标（女 2.0×→1.5 分；男 9.9×→0.5 分）。
2026-10-19 用**当前生产链**（omnivoice + 极简后处理，已不是文档写文章时的
voxcpm）重跑了有 EL 同文本的案例 d172b8c4efb3：

    男声 click = 1.21 × EL   ← 我们确实比 EL 差，是缺陷，该修
    女声 click = 0.61 × EL   ← 我们已经比 EL 干净，不动（文档 §1.2 第三步
                               明写「清洗目标是对齐 EL 密度，严禁过度清洗」）

→ 只动「我们比 EL 差」的那一项。女声不加。

## 为什么必须是 53b（保采样率）而不是 53
`clickfix53.py` 写死 `SR = 16000`，成品一律降到 16 kHz。实测链式后果：

    44.1k wav → 44.1k mp3   音色互比 0.9999（mp3 透明）
    16k   wav → 16k   mp3   音色互比 0.7900（mp3 伤音色）
    16k 管线成品 vs EL       音色 0.6903 ❌不同人（基线 0.9331 ✅）

而项目所有客观指标（measure53 / click / 杂音）口径都是「先降到 16k 再测」，
**这件事在所有指标里完全隐形**，只有 campplus 抓得到。
`clickfix53b.py` 在 16k 上**定位**、在原生采样率上**改样本**，成品保持 44.1k。

## 实测（当前链路，561s，EL = d172b8c4efb3）
    click   1.21×EL → 0.10×EL
    音色    0.9331 → 0.9342（零损失）
    质心 0.97 / 中频 0.84 / 杂音 0.98 / cps 8.19  全部不变
    改动样本 0.0529%

## 安全
- 🔴 只作用在 **omnivoice + 男声**；其他引擎/性别完全不碰。
- 🔴 `TTS_CLICKFIX_MALE_TH` 留空或设为 0 = 关闭，行为与打补丁前一致。
- 🔴 任何异常都**原样退回未修的音频**（后处理挂掉不能让整单失败）。

用法：
    python3 deploy/patch_clickfix53b.py           # 干跑
    python3 deploy/patch_clickfix53b.py --apply   # 打补丁（自动 .bak + 语法自检 + 失败回滚）
"""
import ast
import os
import shutil
import sys
import time

TARGET = "/mnt/tts/projects/qwen3-tts/app/server.py"
MARKER = "_clickfix53b_wrap"
ANCHOR = "    def _run_fix_chain(audio, sr, text):"
OLD_RET = "            return y2, sr2"
NEW_RET = "            return _clickfix53b_wrap(y2, sr2, _g, _engine_name)"

BLOCK = '''
# ---------------------------------------------------------------------------
# 🔴 第53轮 b：保采样率 clickfix（详见 deploy/patch_clickfix53b.py）
#
# click 是唯一与员工评分单调对应的客观指标。当前链路男声 click = 1.21×EL
# （确实比 EL 差），女声 = 0.61×EL（已比 EL 干净）→ **只修男声**。
#
# 🔴 必须用 clickfix53b（保采样率），不能用 clickfix53：后者写死 16k 输出，
#    而 16k 音频压 mp3 会把音色从 0.9331 打到 0.6903（不同人）。
#    项目所有客观指标都在 16k 口径下测，这个降采样在所有指标里是隐形的。
#
# TTS_CLICKFIX_MALE_TH 留空 / 设为 0 = 关闭，行为与打补丁前完全一致。
# ---------------------------------------------------------------------------
CLICKFIX_MALE_TH = os.environ.get("TTS_CLICKFIX_MALE_TH", "0.30")
CLICKFIX_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "tools", "clickfix53b.py")


def _clickfix53b_wrap(audio, sr, gender, engine_name, _tag=""):
    """保采样率 click 修复。🔴 任何异常都原样退回，绝不让整单失败。"""
    try:
        if engine_name != "omnivoice" or (gender or "").lower()[:1] != "m":
            return audio, sr
        th = str(CLICKFIX_MALE_TH or "").strip()
        if not th or float(th) <= 0:
            return audio, sr
        import subprocess as _sp
        import tempfile as _tf
        import soundfile as _sf
        if not os.path.exists(CLICKFIX_BIN):
            log.warning("clickfix53b 不存在（%s），跳过", CLICKFIX_BIN)
            return audio, sr
        d = _tf.mkdtemp(prefix="clickfix_")
        try:
            src = os.path.join(d, "in.wav")
            outp = os.path.join(d, "out.wav")
            _sf.write(src, audio, sr)
            r = _sp.run([sys.executable, CLICKFIX_BIN, src, outp,
                         "--th", th, "--rounds", "3"],
                        capture_output=True, text=True, timeout=1800)
            if r.returncode != 0 or not os.path.exists(outp):
                log.warning("clickfix53b 失败（用未修音频）：%s",
                            (r.stderr or "")[-300:])
                return audio, sr
            y, sr2 = _sf.read(outp, dtype="float32")
            if getattr(y, "ndim", 1) > 1:
                y = y.mean(axis=1)
            if len(y) < int(0.5 * sr):
                log.warning("clickfix53b 产物过短，用未修音频")
                return audio, sr
            log.info("clickfix53b 完成 th=%s（%d Hz 保持不变）", th, sr2)
            return y, sr2
        finally:
            try:
                import shutil as _sh
                _sh.rmtree(d, ignore_errors=True)
            except Exception:
                pass
    except Exception as _e:                                  # noqa: BLE001
        log.warning("clickfix53b 异常（用未修音频）: %r", (_e,))
        return audio, sr


'''


def _indent(block, n=4):
    """🔴🔴 `_run_fix_chain` 是 `async def run_job()` 里的**嵌套函数**（4 空格）。

    第一版补丁把 helper 按 0 缩进插进去，等于把外层 async 函数**截断**了，
    后面的 `await` 全变成「await outside async function」，服务启动即崩。
    （血的教训：`ast.parse` **抓不到**这个错 —— 它只做语法树，不走符号表；
      必须 `compile()` 才会在编译期报出来。）
    → helper 必须同样缩进，作为同层嵌套函数插入。
    """
    out = []
    for line in block.split("\n"):
        out.append((" " * n + line) if line.strip() else line)
    return "\n".join(out)


def _check(src):
    """编译期自检（不是 ast.parse —— 它抓不到 await 作用域错误）。"""
    try:
        compile(src, TARGET, "exec")
        return None
    except SyntaxError as e:
        return e


def main():
    apply_ = "--apply" in sys.argv
    src = open(TARGET, encoding="utf-8").read()
    if MARKER in src:
        print("已打过补丁（发现 %s），跳过" % MARKER)
        return 0
    if ANCHOR not in src:
        print("❌ 找不到锚点：%r" % ANCHOR)
        return 1
    n = src.count(OLD_RET)
    if n != 1:
        print("❌ `return y2, sr2` 出现 %d 次（应为 1），不敢自动改" % n)
        return 1
    new = src.replace(ANCHOR, _indent(BLOCK.lstrip("\n")).lstrip("\n")
                      + ANCHOR, 1)
    new = new.replace(OLD_RET, NEW_RET, 1)
    e = _check(new)
    if e:
        print("❌ 编译自检失败，未落盘：%r" % e)
        return 1
    print("✅ 编译自检通过（compile，非 ast.parse）")
    print("   插入 %d 行 helper；改写 return 1 处" % BLOCK.count("\n"))
    if not apply_:
        print("[DRY-RUN] 加 --apply 才落盘")
        return 0
    bak = TARGET + ".bak-before-clickfix53b-" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(TARGET, bak)
    print("   备份 -> %s" % bak)
    tmp = TARGET + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(new)
    os.replace(tmp, TARGET)
    e = _check(open(TARGET, encoding="utf-8").read())
    if e:
        shutil.copy2(bak, TARGET)
        print("❌ 落盘后编译异常，已回滚：%r" % e)
        return 1
    print("✅ 已写入 %s（服务需重启才生效）" % TARGET)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
