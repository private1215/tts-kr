# -*- coding: utf-8 -*-
"""patch_mideqlog53.py —— 给 `_mideq53b_wrap` 补成功日志，闭合审计盲区。

## 为什么值得单发一个补丁
第53轮已经因为「不知道补丁有没有真执行」连栽两招，两招 audit53 都全绿：
  ① `return _clickfix53b_wrap(*_mideq53b_wrap(...))` 少收参数 → TypeError
     → 整块回退 raw，EQ 和 clickfix **全都没跑**；
  ② `if len(_y) != len(audio): return audio` 判等 → firequalizer 输出长 320 样本
     → **每次都退回**。
两招都是靠**加临时日志**才发现的（`MIDDBG ...`）。

`clickfix53b_wrap` 有 `log.info("clickfix53b 完成 th=%s（%d Hz 保持不变）")`，
所以 clickfix 好不好一 grep 就知道；而 **`_mideq53b_wrap` 成功路径一行日志都没有**，
异常又被 `except Exception: return audio` 静默吞掉 → EQ 有没有跑只能靠猜。
→ 补一行 info 日志，把最后一个盲区关上。**只加日志，不动任何音频行为。**

日志内容带：增益 dB / 采样率 / FIR 多出来的样本数 / 是否居中裁剪。
用法: python3 deploy/patch_mideqlog53.py --apply
"""
import os
import shutil
import sys
import time

TARGET = "/mnt/tts/projects/qwen3-tts/app/server.py"

OLD = (
    '            if _extra:\n'
    '                _o = _extra // 2\n'
    '                _y = _y[_o:_o + len(audio)]\n'
    '            return _y\n'
)

NEW = (
    '            if _extra:\n'
    '                _o = _extra // 2\n'
    '                _y = _y[_o:_o + len(audio)]\n'
    '            log.info("mideq53b 完成 %+g dB（%d Hz 保持不变，FIR 多 %d 样本已居中裁剪）",\n'
    '                     _g, sr, _extra)\n'
    '            return _y\n'
)


def selfcheck(src):
    compile(src, TARGET, "exec")
    chk = [
        ('log.info("mideq53b 完成' in src, "EQ 成功日志已加"),
        ("_extra = len(_y) - len(audio)" in src
         and "_y[_o:_o + len(audio)]" in src, "居中裁剪逻辑仍在（没被日志改动破坏）"),
        ("return _y\n" in src, "helper 仍只返回 audio（没变成二元组）"),
        ("*_mideq53b_wrap" not in src, "没有 *解包 误用"),
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
    if 'log.info("mideq53b 完成' in src:
        print("已打过本补丁，跳过。")
        selfcheck(src)
        return 0
    if src.count(OLD) != 1:
        raise SystemExit("❌ 锚点匹配到 %d 次（要恰好 1 次），中止"
                         % src.count(OLD))
    src = src.replace(OLD, NEW)
    print("替换 1 处（_mideq53b_wrap 成功路径加 info 日志）")
    selfcheck(src)
    if not apply:
        print("\n（未加 --apply，只做了检查）")
        return 0
    bak = "%s.bak-before-mideqlog53-%s" % (TARGET, time.strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(TARGET, bak)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)
    compile(open(TARGET, encoding="utf-8").read(), TARGET, "exec")
    print("\n✅ 已写入 %s\n✅ 备份 %s\n✅ 落盘后再次 compile() 通过" % (TARGET, bak))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
