# -*- coding: utf-8 -*-
"""给 audit53.py 补「反馈批次轴」的检查项。

背景（2026-10-19 批次漂移事故）：
    `release_version.py` 的 `rewrite_items()` 旧实现只换 `id` 的版本前缀、
    不换 `z` 批次段 → 升到 v3.3 后题库成了 `v3.3-z32-*`，
    而当时的 `/api/version` 只校验版本、`consistent` 仍报 true。
    修法见 `tools/patch_verround53.py`（两条轴一起改、一起校验）。

本补丁只做**追加**，不改原有任何检查：
    1. VERSION 解析时一并读 `round =`，打印出来；
    2. 静态确认 `app/relver.py` 真的有 round 校验（防被回滚掉）；
    3. `/api/version` 增加 `version_consistent` / `round_consistent` 两条断言。

用法: python3 tools/patch_audit53_round.py --apply
"""
import io
import os
import shutil
import sys
import time

TARGET = "/mnt/tts/projects/qwen3-tts/tools/audit53.py"

OLD_VER = '''    m = re.search(r"^version\\s*=\\s*(\\S+)", open(vp, encoding="utf-8").read(), re.M)
    print("       当前版本 = %s" % (m.group(1) if m else "?"))
rv = os.path.join(ROOT, "tools", "release_version.py")
chk(os.path.exists(rv), "release_version.py 存在（唯一版本号写入口）")'''

NEW_VER = '''    src_ver = open(vp, encoding="utf-8").read()
    m = re.search(r"^version\\s*=\\s*(\\S+)", src_ver, re.M)
    mr = re.search(r"^round\\s*=\\s*(\\S+)", src_ver, re.M)
    print("       当前版本 = %s   当前批次 = %s"
          % (m.group(1) if m else "?", mr.group(1) if mr else "?"))
    chk(bool(mr), "\\U0001f534 VERSION 里标了 round（反馈批次轴）")
rv = os.path.join(ROOT, "tools", "release_version.py")
chk(os.path.exists(rv), "release_version.py 存在（唯一版本号写入口）")
if os.path.exists(rv):
    _rv = open(rv, encoding="utf-8").read()
    # \\U0001f534 静态确认批次轴真的实现着，被回滚会立刻显形
    chk("new_round" in _rv, "\\U0001f534 rewrite_items 支持同步批次（new_round）")
    chk("RE_ROUND" in _rv, "\\U0001f534 release_version.py 能识别 z 批次段（RE_ROUND）")
_rl = os.path.join(ROOT, "app", "relver.py")
chk(os.path.exists(_rl), "app/relver.py 存在（凭证快照）")
if os.path.exists(_rl):
    _rls = open(_rl, encoding="utf-8").read()
    chk("round_ok" in _rls, "\\U0001f534 relver.snapshot 校验批次（round_ok）")
    chk("round_consistent" in _rls, "\\U0001f534 relver 输出 round_consistent 字段")'''

OLD_API = '''    chk(j.get("consistent") is True, "/api/version consistent = %s"
        % j.get("consistent"))'''

NEW_API = '''    chk(j.get("consistent") is True, "/api/version consistent = %s"
        % j.get("consistent"))
    chk(j.get("version_consistent") is True, "/api/version version_consistent = %s"
        % j.get("version_consistent"))
    chk(j.get("round_consistent") is True,
        "\\U0001f534 /api/version round_consistent = %s（false = 批次漂移："
        "题库 id 第二段和 VERSION 的 round 不一致）" % j.get("round_consistent"))
    _fb = j.get("fb") or {}
    chk(not _fb.get("round") or _fb.get("round") == j.get("round"),
        "题库自报批次 = %s，真相源 %s" % (_fb.get("round"), j.get("round")))'''


def main():
    apply = "--apply" in sys.argv
    if not os.path.exists(TARGET):
        raise SystemExit("找不到 %s" % TARGET)
    src = io.open(TARGET, encoding="utf-8").read()
    if "round_consistent" in src:
        print("已打过，跳过")
        return 0
    for old, new, what in ((OLD_VER, NEW_VER, "VERSION/relver 检查项"),
                           (OLD_API, NEW_API, "/api/version 批次断言")):
        if src.count(old) != 1:
            raise SystemExit("❌ %s 匹配 %d 次（要 1 次）" % (what, src.count(old)))
        src = src.replace(old, new)
        print("替换 %s" % what)
    # 🔴 必须 compile()，ast.parse 抓不到作用域错误
    compile(src, TARGET, "exec")
    print("compile() 通过")
    if not apply:
        print("（未加 --apply）")
        return 0
    bak = "%s.bak-before-auditround-%s" % (TARGET, time.strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(TARGET, bak)
    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)
    compile(io.open(TARGET, encoding="utf-8").read(), TARGET, "exec")
    print("✅ 写入 %s  备份 %s" % (TARGET, bak))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
