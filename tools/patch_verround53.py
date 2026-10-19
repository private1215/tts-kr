# -*- coding: utf-8 -*-
"""patch_verround53.py —— 把「项目版本」和「反馈批次」两条轴都变成凭证。

## 规范怎么定的（`项目规范.md` §1 版本标识 + §5 硬性规则 11）
版本号 `v<主>.<次>`，真相源 `<项目根>/VERSION`；**只能**由
`tools/release_version.py --bump` 写。
反馈题的 `id` 前缀 = `v<版本>-<z批次>-<维度>`，`ver` 字段同值。

→ 也就是说 id 里**装了两条独立的轴**：
    `v3.3` = **项目版本**（链路到哪一步了）
    `z33`  = **反馈批次**（第几批发出去的问题）
  两者必须各自同步，混着用事后就没法按批次回溯。

## 实测到的两个缺口（2026-10-19）
1. `rewrite_items()` 只换 id 第一段（版本），**不换第二段（z 批次）**
   → 升到 v3.3 后题库变成 `v3.3-z32-*`：版本是新的、批次还是上一批。
   而 `relver.fb_state()` 恰恰是取 `id.split("-")[1]` 当批次 →
   `/api/version` 里出现 `version=v3.3, fb.round=z32` 的混合态。
2. `consistent` 只比 `fb_ver == server_version`，**完全不看批次**
   → 上面那个混合态 `consistent=true`，一点不报警。
   （当时我还以为凭证是好的，直到肉眼看 id 才发现。）

## 改什么
A. `tools/release_version.py`
   - `rewrite_items(paths, new_ver, dry, new_round="")`：新增 `new_round`，
     把 id 第二段（形如 `z32`）也改写成 `--round` 的值。
   - 调用处把 `a.round` 传进去（`--items` 的语义不变，仍是显式指定题库）。
   - `check()` 增加一行：批次不一致时明确报出来。
B. `app/relver.py`
   - `snapshot()` 增加 `round_consistent`；`consistent` 改为
     「版本对上 **且** 没混版本 **且** 批次对上」。
   - 保留 `version_consistent` 单独字段，方便区分是哪条轴出问题。

## 不改什么（刻意）
- **不加**「bump 时自动扫描 items.json」那种隐式行为。规范写的是显式
  `--items <题库>`，加了自动发现反而多一条不写在规范里的路径。
- 不改 `--items` / `--bump` / `--set` 的任何现有语义。

用法: python3 tools/patch_verround53.py --apply
"""
import io
import os
import shutil
import sys
import time

REL = "/mnt/tts/projects/qwen3-tts/tools/release_version.py"
RV = "/mnt/tts/projects/qwen3-tts/app/relver.py"

# ---------------------------------------------------------------- release_version.py
R_OLD_SIG = '''def rewrite_items(paths, new_ver, dry=False):
    """把题库里的 id 前缀和 ver 统一到 new_ver。

    \U0001f534 id 形如 `v1.1-z17-spd`：只换第一段（版本号），保留 `-z17-spd`，
       这样历史回溯时看 id 就知道「第几轮、问的什么」。
    """
'''

R_NEW_SIG = '''def rewrite_items(paths, new_ver, dry=False, new_round=""):
    """把题库的 id 前缀 + z 批次段 + ver 统一到 new_ver / new_round。

    \U0001f534 id 形如 `v1.1-z17-spd`：**第一段是项目版本，第二段是反馈批次**，
       两条独立的轴 —— 版本由 --bump 决定，批次由 --round 决定。
       旧实现只换版本、不换批次，升到 v3.3 后题库会变成 `v3.3-z32-*`，
       版本与批次脱节（2026-10-19 实测踩过，且 consistent 当时不报警）。
       现在两段一起同步。
    """
'''

R_OLD_LOOP = '''            parts = old.split("-")
            if len(parts) >= 2 and RE_VER.match(parts[0]):
                new = "-".join([new_ver] + parts[1:])
                if new != old:
                    it["id"] = new
                    changed += 1
'''

R_NEW_LOOP = '''            parts = old.split("-")
            if len(parts) >= 2 and RE_VER.match(parts[0]):
                rest = list(parts[1:])
                # \U0001f534 批次段只在形如 z32 时才替换，别把 `m_v4` 之类的维度当批次改掉
                if new_round and rest and RE_ROUND.match(rest[0]):
                    rest[0] = new_round
                new = "-".join([new_ver] + rest)
                if new != old:
                    it["id"] = new
                    changed += 1
'''

R_OLD_RE = '''RE_VER = re.compile(r"^v(\\d+)\\.(\\d+)$")
'''

R_NEW_RE = '''RE_VER = re.compile(r"^v(\\d+)\\.(\\d+)$")
RE_ROUND = re.compile(r"^z\\d+$")
'''

R_OLD_CALL = '''    if a.items:
        print("  统一题库版本：")
        rewrite_items(a.items, new, a.dry)
'''

R_NEW_CALL = '''    if a.items:
        print("  统一题库版本（版本 %s / 批次 %s）：" % (new, a.round or "不变"))
        rewrite_items(a.items, new, a.dry, a.round)
'''

R_OLD_CHK = '''    print("缺音频         : %s" % (miss or "无"))
'''

R_NEW_CHK = '''    print("缺音频         : %s" % (miss or "无"))
    if r.get("round_consistent") is False:
        print("\U0001f534 反馈批次落后：服务端 round=%s，题库 id 里是 %s"
              % (r.get("round"), r.get("fb", {}).get("round")))
        print("   修：release_version.py --set <当前版本> --round <服务端 round>"
              " --items <题库文件>")
'''

# ---------------------------------------------------------------- relver.py
V_OLD = '''    mixed = len(st["vers"]) > 1
    return {
'''

V_NEW = '''    mixed = len(st["vers"]) > 1
    # \U0001f534 v3.3 起：**批次也是凭证**。项目版本（v3.3）与反馈批次（z33）是两条
    #   独立的轴：版本对了但批次还留在上一批，等于「题问的是 v3.3、却挂在上一批
    #   发出去的清单里」，事后没法按批次回溯。旧实现只看版本，这种混合态不报警。
    ver_ok = bool(fb_ver) and fb_ver == sv
    round_ok = (not m.get("round")) or (not fb_round) or (
        fb_round == m.get("round"))
    return {
'''

V_OLD_CONS = '''        "consistent": bool(fb_ver) and fb_ver == sv,
        "mixed_versions": mixed,
'''

V_NEW_CONS = '''        "consistent": ver_ok and (not mixed) and round_ok,
        "version_consistent": ver_ok,
        "round_consistent": round_ok,
        "mixed_versions": mixed,
'''


def apply_one(path, pairs, what):
    src = io.open(path, encoding="utf-8").read()
    n = 0
    for old, new, tag in pairs:
        if src.count(old) != 1:
            raise SystemExit("❌ [%s] %s 匹配到 %d 次（要恰好 1 次）"
                             % (what, tag, src.count(old)))
        src = src.replace(old, new)
        n += 1
        print("  [%s] 替换 %s" % (what, tag))
    compile(src, path, "exec")
    print("  [%s] compile() 通过（%d 处）" % (what, n))
    return src


def main():
    apply = "--apply" in sys.argv
    for p in (REL, RV):
        if not os.path.exists(p):
            raise SystemExit("找不到 %s" % p)
    if 'RE_ROUND = re.compile' in io.open(REL, encoding="utf-8").read():
        print("已打过本补丁，跳过。")
        return 0

    rel_src = apply_one(REL, [
        (R_OLD_SIG, R_NEW_SIG, "rewrite_items 签名+docstring"),
        (R_OLD_LOOP, R_NEW_LOOP, "id 循环同时换版本+批次"),
        (R_OLD_RE, R_NEW_RE, "新增 RE_ROUND"),
        (R_OLD_CALL, R_NEW_CALL, "调用处传 a.round"),
        (R_OLD_CHK, R_NEW_CHK, "check() 报批次不一致"),
    ], "release_version.py")

    rv_src = apply_one(RV, [
        (V_OLD, V_NEW, "snapshot 增加 round_ok"),
        (V_OLD_CONS, V_NEW_CONS, "consistent 纳入批次+混版本"),
    ], "relver.py")

    if not apply:
        print("\n（未加 --apply，只做了检查）")
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    for path, src in ((REL, rel_src), (RV, rv_src)):
        bak = "%s.bak-before-verround53-%s" % (path, ts)
        shutil.copy2(path, bak)
        with io.open(path, "w", encoding="utf-8") as f:
            f.write(src)
        compile(io.open(path, encoding="utf-8").read(), path, "exec")
        print("✅ 写入 %s（备份 %s）" % (path, bak))
    print("\n✅ 两个文件落盘后都再次 compile() 通过")
    print("⚠️ 改了 app/relver.py → **必须重启 qwen-tts** 才生效")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
