# -*- coding: utf-8 -*-
"""patch_female53.py —— 把保采样率 中频EQ + clickfix 从「只男声」扩展到女声。

## 为什么要扩（2026-10-19 实测，pipe_f53）
女声权威基线重测后（**合规标杆 = cases/00b6a45ad6aa/merged.wav**）：

    我们             EL-B            我们/EL
    click   0.0022%     0.0018%        1.24   ← 但在 EL 自身方差内（见下）
    中频    5.82%       6.65%          **0.88** ← 唯一真实缺口
    质心    580.9       599.8          0.97
    杂音    1.59%       1.25%          1.27
    音色    —           —              **0.9204**

**EL 自身运行间方差**（同文本同嗓音两次录音，ASR 已坐实同文案）：
    qwen_fullf_EL_female.wav : click 0.0046% / 中频 6.5% / 质心 574 / 杂音 7.3%
    00b6a45ad6aa/merged.wav  : click 0.0018% / 中频 6.7% / 质心 600 / 杂音 1.3%
    → click 相差 **2.56×**、杂音相差 **5.6×**。
→ 我们 click 0.0022% 落在两个 EL 值**之间**，不是真差距；
  但中频 5.8% **低于两个 EL 值（6.5 / 6.7）**，是真差距 → 该修。

## 修什么（女声离线同源对照，pipe_f53）
    f0 原样          : click 1.24 / 中频 0.88 / 杂音 1.27 / 音色 0.9204
    f2 仅 EQ+2       : click 2.11 / 中频 1.02 / 杂音 1.09 / 音色 0.9207
    **f3 EQ+2+cf0.26**: click **0.02** / 中频 **1.02** / 杂音 **1.09** / 音色 **0.9207**

女声 clickfix 阈值扫描（与男声同型，0.40 反向恶化）：
    th=0.20 改 0.196% 样本 → click 0.02×
    **th=0.26 改 0.034% 样本 → click 0.02×**   ← 效率最优点
    th=0.32 改 0.0053%     → click 1.09×
    th=0.40 改 0.0005%     → click 2.07×（278→274，等于没修）

音色代价：th=0.26 为 −0.0005（th=0.20 为 −0.0019）→ 0.26 更省。

## 改动
1. `_clickfix53b_wrap`：性别守卫从 `!= "m"` 改为 m/f 分流，
   新增 `TTS_CLICKFIX_FEMALE_TH`（默认 0.26）。
2. `_mideq53b_wrap`：同上，新增 `TTS_MID_EQ_FEMALE_DB`（默认 2）。
3. 两处默认值都可通过环境变量关（留空/0 = 完全关闭，回到打补丁前行为）。

## 🔴 三个连环坑（前两个已经踩过，都写进注释）
① `return _clickfix53b_wrap(*_mideq53b_wrap(...))` —— 解包二元组少收
   gender/engine_name → TypeError → 整块回退 raw，EQ 和 clickfix 全都没跑。
   → `_mideq53b_wrap` **只返回 audio**。
② `if len(_y) != len(audio): return audio` —— firequalizer 是 FIR，输出比输入
   长 filter_size-1 个样本 → **每次都退回**。→ 用**居中裁剪**。
③ 本补丁只改守卫和阈值来源，**不动返回结构、不动居中裁剪**，避免重新引入 ①②。

## 自检
必须用 `compile(src, TARGET, "exec")`，**不能用 ast.parse()**（不走符号表，
抓不到 await 作用域错误）。且必须确认插进去的 helper 与 `_run_fix_chain`
（`async def run_job()` 的嵌套函数）**同缩进**（def 4 空格 / 函数体 8 空格）。

用法: python3 deploy/patch_female53.py --apply   （不带 --apply 只做检查）
"""
import os
import shutil
import sys
import time

TARGET = "/mnt/tts/projects/qwen3-tts/app/server.py"

# ---------------------------------------------------------------- 旧片段（精确匹配）
OLD_CLICKFIX_TH = (
    '    CLICKFIX_MALE_TH = os.environ.get("TTS_CLICKFIX_MALE_TH", "0.30")\n'
)

OLD_CLICKFIX_GUARD = (
    '            if engine_name != "omnivoice" or (gender or "").lower()[:1] != "m":\n'
    '                return audio, sr\n'
    '            th = str(CLICKFIX_MALE_TH or "").strip()\n'
)

OLD_MIDEQ_GUARD = (
    '            if engine_name != "omnivoice" or (gender or "").lower()[:1] != "m":\n'
    '                return audio\n'
    '            _g = str(os.environ.get("TTS_MID_EQ_MALE_DB", "2") or "").strip()\n'
)

# ---------------------------------------------------------------- 新片段
NEW_CLICKFIX_TH = (
    '    CLICKFIX_MALE_TH = os.environ.get("TTS_CLICKFIX_MALE_TH", "0.30")\n'
    '    # 第53轮 f：女声也接 clickfix（阈值 0.26，与男声 0.30 分性别）\n'
    '    # 女声 th 扫描（pipe_f53，EQ+2 之后）：0.20 改 0.196% 样本 / **0.26 改 0.034%** /\n'
    '    # 0.32 改 0.0053% / 0.40 改 0.0005%（278→274 等于没修）。0.26 是效率最优点。\n'
    '    CLICKFIX_FEMALE_TH = os.environ.get("TTS_CLICKFIX_FEMALE_TH", "0.26")\n'
)

NEW_CLICKFIX_GUARD = (
    '            if engine_name != "omnivoice":\n'
    '                return audio, sr\n'
    '            _gd = (gender or "").lower()[:1]\n'
    '            if _gd == "m":\n'
    '                th = str(CLICKFIX_MALE_TH or "").strip()\n'
    '            elif _gd == "f":\n'
    '                th = str(CLICKFIX_FEMALE_TH or "").strip()\n'
    '            else:\n'
    '                return audio, sr\n'
)

NEW_MIDEQ_GUARD = (
    '            if engine_name != "omnivoice":\n'
    '                return audio\n'
    '            _gd = (gender or "").lower()[:1]\n'
    '            _eqenv = {"m": "TTS_MID_EQ_MALE_DB",\n'
    '                      "f": "TTS_MID_EQ_FEMALE_DB"}.get(_gd)\n'
    '            if _eqenv is None:\n'
    '                return audio\n'
    '            _g = str(os.environ.get(_eqenv, "2") or "").strip()\n'
)

# 注释里「只修男声 / 只 omni+男声」的过期描述也要改，免得下次照旧注释做决定
OLD_CMT_CLICKFIX = (
    '    # click 是唯一与员工评分单调对应的客观指标。当前链路男声 click = 1.21×EL\n'
    '    # （确实比 EL 差），女声 = 0.61×EL（已比 EL 干净）→ **只修男声**。\n'
)
NEW_CMT_CLICKFIX = (
    '    # click 是唯一与员工评分单调对应的客观指标。\n'
    '    # 第53轮 f 起男女声都修：男声 click 1.21×EL，女声 1.24×EL（合规标杆\n'
    '    # cases/00b6a45ad6aa/merged.wav）→ 阈值分性别：男 0.30 / 女 0.26。\n'
)

OLD_CMT_MIDEQ = (
    '        """保采样率**窄带中频 EQ**（1k:0 / 1.8k:+G / 3k:0）。只 omni+男声。\n'
)
NEW_CMT_MIDEQ = (
    '        """保采样率**窄带中频 EQ**（1k:0 / 1.8k:+G / 3k:0）。omni+男女声。\n'
)


def selfcheck(src):
    """🔴 必须 compile()，不能用 ast.parse()。"""
    compile(src, TARGET, "exec")
    chk = [
        ("CLICKFIX_FEMALE_TH" in src, "女声 clickfix 阈值已定义"),
        ('"f": "TTS_MID_EQ_FEMALE_DB"' in src, "女声 EQ 环境变量分流已写"),
        ('elif _gd == "f":' in src, "clickfix 性别分流已写"),
        (OLD_CLICKFIX_GUARD not in src, "旧 clickfix 守卫（只男声）已替换"),
        (OLD_MIDEQ_GUARD not in src, "旧 mideq 守卫（只男声）已替换"),
        ("*_mideq53b_wrap" not in src, "没有 *解包 误用（那会让 EQ+clickfix 全不跑）"),
        ("_extra = len(_y) - len(audio)" in src
         and "_y[_o:_o + len(audio)]" in src,
         "FIR 输出变长用**居中裁剪**（用 != 判等会让 EQ 永远退回）"),
        ("return _clickfix53b_wrap(\n"
         "                _mideq53b_wrap(y2, sr2, _g, _engine_name),"
         in src, "调用处显式传 sr2/_g/_engine_name（不是 *解包）"),
        ("只 omni+男声" not in src, "过期注释「只 omni+男声」已更新"),
    ]
    bad = [m for ok, m in chk if not ok]
    for ok, m in chk:
        print(("  ✅ " if ok else "  ❌ ") + m)
    if bad:
        raise SystemExit("自检失败：%s" % bad)


def main():
    apply = "--apply" in sys.argv
    if not os.path.exists(TARGET):
        raise SystemExit("找不到 %s" % TARGET)
    src = open(TARGET, encoding="utf-8").read()
    if "CLICKFIX_FEMALE_TH" in src:
        print("已打过本补丁，跳过。")
        selfcheck(src)
        return 0
    n = 0
    for old, new, what in [
        (OLD_CLICKFIX_TH, NEW_CLICKFIX_TH, "clickfix 阈值定义"),
        (OLD_CLICKFIX_GUARD, NEW_CLICKFIX_GUARD, "clickfix 性别守卫"),
        (OLD_MIDEQ_GUARD, NEW_MIDEQ_GUARD, "mideq 性别守卫"),
        (OLD_CMT_CLICKFIX, NEW_CMT_CLICKFIX, "clickfix 过期注释"),
        (OLD_CMT_MIDEQ, NEW_CMT_MIDEQ, "mideq 过期注释"),
    ]:
        if src.count(old) != 1:
            raise SystemExit("❌ %s 匹配到 %d 次（要恰好 1 次），中止"
                             % (what, src.count(old)))
        src = src.replace(old, new)
        n += 1
        print("替换 %s" % what)
    print("共替换 %d 处" % n)
    selfcheck(src)
    if not apply:
        print("\n（未加 --apply，只做了检查）")
        return 0
    bak = "%s.bak-before-female53-%s" % (
        TARGET, time.strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(TARGET, bak)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)
    compile(open(TARGET, encoding="utf-8").read(), TARGET, "exec")
    print("\n✅ 已写入 %s\n✅ 备份 %s\n✅ 落盘后再次 compile() 通过" % (TARGET, bak))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
