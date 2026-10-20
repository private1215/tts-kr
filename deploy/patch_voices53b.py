#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deploy/patch_voices53b.py —— 修 patch_voice 原子写导致的**属主被改成 root**。

## 事故
patch_voices53 的 `os.replace(tmp, VOICES_FILE)` 把 tmp（服务以 root 运行，
tmp 属主 root）的 inode 换成正式文件 → voices.json 从 zheng:zheng 变成
root:root，外部脚本/管理员（zheng）再也改不动它，等于把「外部可编辑」这个
前提给废了 —— 正好是本补丁要保护的东西。

（对比：原来的 `save_voices` 是 open(w) 直接覆盖，不改 inode，属主保持不变，
所以以前没这个问题。）

## 修法
替换前把原文件的 uid/gid/mode 搬到 tmp 上，再 os.replace。
"""
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "app", "server.py")

OLD = '''    tmp = VOICES_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as _f:
        json.dump(d, _f, ensure_ascii=False, indent=2)
    os.replace(tmp, VOICES_FILE)
    _VOICES_MTIME[0] = _voices_mtime()'''

NEW = '''    tmp = VOICES_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as _f:
        json.dump(d, _f, ensure_ascii=False, indent=2)
    # 🔴 原子替换会把 tmp（服务以 root 跑 -> 属主 root）的 inode 换成正式文件。
    #    必须把原文件的属主/权限搬过来，否则 voices.json 变成 root:root，
    #    外部脚本（zheng）就改不动了 —— 那正好是本补丁要保护的东西。
    try:
        _st = os.stat(VOICES_FILE)
        os.chown(tmp, _st.st_uid, _st.st_gid)
        os.chmod(tmp, _st.st_mode & 0o7777)
    except Exception as _e:                                    # noqa: BLE001
        log.warning("patch_voice 继承属主失败（不影响写入）: %s", _e)
    os.replace(tmp, VOICES_FILE)
    _VOICES_MTIME[0] = _voices_mtime()'''


def main():
    apply = "--apply" in sys.argv
    src = open(TARGET, encoding="utf-8").read()
    if OLD not in src:
        print("!! 锚点未找到（补丁可能已打过）")
        sys.exit(1)
    print("将替换 1 处（补属主继承）")
    if not apply:
        print("---- 干跑：加 --apply 实打 ----")
        return
    bak = TARGET + ".bak-before-voices53b-" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copyfile(TARGET, bak)
    print("备份 -> %s" % bak)
    open(TARGET, "w", encoding="utf-8").write(src.replace(OLD, NEW, 1))
    import py_compile
    try:
        py_compile.compile(TARGET, doraise=True)
        print("编译自检通过")
    except Exception as e:
        print("!! 编译失败，回滚: %s" % e)
        shutil.copyfile(bak, TARGET)
        sys.exit(1)
    print("补丁完成：patch_voice 写回时继承原文件属主")


if __name__ == "__main__":
    main()
