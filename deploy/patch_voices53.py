#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deploy/patch_voices53.py —— 修 voices.json 被「内存快照整表写回」冲掉的问题。

## 根因（2026-10-20 定位）
`voices = load_voices()` 启动时只读一次进内存；而**每单跑完**的语速自校准会
执行 `save_voices(voices)`，把内存快照**整表**写回磁盘。于是管理员/外部脚本
（tools/mkcase_voices.py 之类）在磁盘上改的 `engine` 字段、新加的音色，
等服务跑完下一单就被冲掉。以前的规避办法是「改完必须重启 qwen-tts」。

## 修法（两条，互不替代）
1. **`patch_voice(name, patch)`**：测量型字段（cps / cps_lock）改走这里 ——
   **先重读磁盘**，只改目标音色那一个条目，再原子写回（os.replace）。
   别的音色、以及同一音色里外部手改的 engine/path/label 一律不动。
   → 替换掉两个自动写回点（run_job 的 cps_lock、跑完单的 cps 自校准）。
2. **`refresh_voices_if_changed()`**：每单开始前 stat 一次 mtime，磁盘比上次
   加载新就原地 reload（voices.clear() + update，不改引用）。
   → 外部改动**不用重启**也能生效，顺带让内存不再"旧于磁盘"。

⚠️ 没做「save_voices 合并磁盘新增条目」：那需要区分"外部新增"和"内存里被
   删除"，会把删除的音色复活。CRUD 接口（录入/改引擎/改元数据/删除）仍是
   整表写回 —— 它们是用户主动操作，且已有 /api/voices/reload 兜底。

用法：
    python3 deploy/patch_voices53.py           # 干跑
    python3 deploy/patch_voices53.py --apply   # 打补丁（.bak + 编译自检 + 回滚）
"""
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "app", "server.py")

ANCHOR_DEF = "voices = load_voices()\n"

HELPERS = '''voices = load_voices()


# ============================ voices.json 热重载 / 单点写入 ============================
# 🔴 背景：voices 启动时只读一次进内存，而每单跑完的 cps 自校准会
#   save_voices(voices) 整表写回 —— 管理员/外部脚本在磁盘上改的 engine 字段、
#   新加的音色，等服务跑完下一单就被冲掉（以前只能靠重启规避）。
#   2026-10-20 起：① 测量型字段走 patch_voice（重读磁盘只改一个音色）
#                  ② 每单开始前自动热重载外部改动
_VOICES_MTIME = [0.0]


def _voices_mtime():
    try:
        return os.path.getmtime(VOICES_FILE)
    except OSError:
        return 0.0


_VOICES_MTIME[0] = _voices_mtime()


def refresh_voices_if_changed():
    """磁盘 voices.json 被外部改过就原地重新读进内存（不改 dict 引用）。

    🔴 必须 clear()+update()：voices 是模块级 dict，别处拿的是同一个引用，
       写成 `voices = load_voices()` 只会换掉这个局部名字。
    """
    m = _voices_mtime()
    if m <= _VOICES_MTIME[0]:
        return False
    try:
        fresh = load_voices()
    except Exception as _e:                                    # noqa: BLE001
        log.warning("voices.json 读取失败，保持内存版本: %s", _e)
        return False
    if not isinstance(fresh, dict) or not fresh:
        log.warning("voices.json 为空或格式不对，保持内存版本")
        return False
    voices.clear()
    voices.update(fresh)
    _VOICES_MTIME[0] = m
    log.info("检测到 voices.json 被外部修改，已自动重载：%d 个音色", len(voices))
    return True


def patch_voice(name, patch):
    """只改一个音色的若干字段：**先重读磁盘**再改再原子写回。

    测量型字段（cps / cps_lock）必须走这里 —— 磁盘为底只改目标音色，
    别的音色完全不动，同一音色里外部手改的 engine/path/label 也保住。
    """
    try:
        d = load_voices()
    except Exception as _e:                                    # noqa: BLE001
        log.warning("patch_voice 读取失败，退回内存版本: %s", _e)
        d = dict(voices)
    e = dict(d.get(name) or voices.get(name) or {})
    e.update(patch)
    d[name] = e
    tmp = VOICES_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as _f:
        json.dump(d, _f, ensure_ascii=False, indent=2)
    os.replace(tmp, VOICES_FILE)
    _VOICES_MTIME[0] = _voices_mtime()
    if name in voices:
        voices[name].update(patch)
    else:
        voices[name] = e
    return e
'''

# 每单开始前热重载
OLD_RUN = '''async def run_job(job):
    loop = asyncio.get_running_loop()
    job["status"] = "running"'''
NEW_RUN = '''async def run_job(job):
    loop = asyncio.get_running_loop()
    # 🔴 每单开始前先看 voices.json 有没有被外部改过：改过就原地重载，
    #    这样外部改动不用重启也能生效，本单跑完的 cps 写回也不会冲掉它。
    refresh_voices_if_changed()
    job["status"] = "running"'''

# 自动写回点 ①：分块用的 cps_lock
OLD_LOCK = '''        _vmeta["cps_lock"] = _vcps_chunk
        save_voices(voices)'''
NEW_LOCK = '''        _vmeta["cps_lock"] = _vcps_chunk
        patch_voice(job["voice"], {"cps_lock": _vcps_chunk})'''

# 自动写回点 ②：跑完单的 cps 自校准
OLD_CPS = '''                _v["cps"] = round(_new_cps if not _old
                                  else 0.8 * _old + 0.2 * _new_cps, 3)
                save_voices(voices)'''
NEW_CPS = '''                _v["cps"] = round(_new_cps if not _old
                                  else 0.8 * _old + 0.2 * _new_cps, 3)
                patch_voice(_vc, {"cps": _v["cps"]})'''

REPL = [(OLD_RUN, NEW_RUN), (OLD_LOCK, NEW_LOCK), (OLD_CPS, NEW_CPS)]


def main():
    apply = "--apply" in sys.argv
    src = open(TARGET, encoding="utf-8").read()
    if "def patch_voice(" in src:
        print("!! 补丁看起来已经打过")
        sys.exit(1)
    if ANCHOR_DEF not in src:
        print("!! 锚点未找到：%r" % ANCHOR_DEF)
        sys.exit(1)
    missing = [a for a, _ in REPL if a not in src]
    if missing:
        print("!! 以下替换锚点未找到：")
        for m in missing:
            print("   %r" % m[:70])
        sys.exit(1)
    print("将插入 helper %d 行 + 替换 %d 处"
          % (len(HELPERS.splitlines()), len(REPL)))
    if not apply:
        print("---- 干跑：加 --apply 实打 ----")
        return
    bak = TARGET + ".bak-before-voices53-" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copyfile(TARGET, bak)
    print("备份 -> %s" % bak)
    out = src.replace(ANCHOR_DEF, HELPERS, 1)
    for a, b in REPL:
        out = out.replace(a, b, 1)
    open(TARGET, "w", encoding="utf-8").write(out)
    # 🔴 自检必须 compile()，ast.parse() 不走符号表抓不到 await 作用域错误
    import py_compile
    try:
        py_compile.compile(TARGET, doraise=True)
        print("编译自检通过")
    except Exception as e:
        print("!! 编译失败，回滚: %s" % e)
        shutil.copyfile(bak, TARGET)
        sys.exit(1)
    print("补丁完成：patch_voice + 每单热重载 voices.json")


if __name__ == "__main__":
    main()
