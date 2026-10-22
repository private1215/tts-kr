# -*- coding: utf-8 -*-
"""release_version.py —— 发布版本号**自动升级**（版本凭证的唯一写入口）。

## 为什么不能用手工维护（第48轮实测踩出来的）
题目标注的版本号此前靠出题脚本的环境变量 `TTS_VER` 传，忘了传就用
脚本里的默认值 —— 结果 `items_z17.json` 标着 `v1.1`，素材却是用 **v1.2**
链路（GS 专属后处理）生成的 gs48m_post2，**版本号失真且没人发现**。
员工反馈「v1.1 这条有问题」时，定位到的是错误的改动清单。

→ 所以：版本号只能由**这一个脚本**写入服务端真相源（`<项目根>/VERSION`），
  写完立刻用 /api/version 自检 `consistent` 是否为 true。
  手工改 VERSION 属于违规。

## 用法（在服务器上跑）

    # 1) 先看看现在什么情况
    venv/bin/python release_version.py --check

    # 2) 次版本 +1（v1.3 -> v1.4），并可顺带把题库统一到新版本
    venv/bin/python release_version.py --bump minor \\
        --round z18 --note "发布 z18 女声 GS 裁决题（5 题）" \\
        --items /mnt/tts/fb/items_z16.json /mnt/tts/fb/items_z17.json

    # 3) 主版本 +1（引擎/参考音频/后处理链级改动）
    venv/bin/python release_version.py --bump major --round z19 ...

## bump 规则（照 VERSION 文档「版本号规则」）
| 位 | 何时递增 | 例子 |
|---|---|---|
| 主版本 | 引擎 / 参考音频 / 后处理链级别的改动 | 换引擎、换参考音频切片、改后处理链 |
| 次版本 | 参数微调 | EQ 参数、阈值、增益、开关、出新一批反馈题 |

## 写什么
1. 服务端 `<项目根>/VERSION`（version / round / bump / updated_at / note）
2. `--items` 指定的题库：id 前缀 `v<旧>-z18-x` → `v<新>-z18-x`，ver 字段同步
   🔴 id 格式是文档定死的 `v<版本>-<z批次>-<维度>`，**不要改格式**
3. 本地仓库 VERSION 文档的「当前版本」行（在本机跑时生效）
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

SERVER_PROJ = "/mnt/tts/projects/qwen3-tts"
SERVER_VER = os.path.join(SERVER_PROJ, "VERSION")
FB_DIR = "/mnt/tts/fb"
LOCAL_VER = "/Users/mac/Desktop/oulian-js/tts-kr/VERSION"
API = "http://127.0.0.1:8000/api/version"

RE_VER = re.compile(r"^v(\d+)\.(\d+)$")
RE_ROUND = re.compile(r"^z\d+$")
ON_SERVER = os.path.isdir(SERVER_PROJ)


def now_bj():
    """服务器是 UTC，北京时间 = UTC+8。"""
    return datetime.now(timezone(timedelta(hours=8))).strftime(
        "%Y-%m-%dT%H:%M:%S+08:00")


def read_server():
    """读服务端真相源。返回 (dict, missing)。

    🔴 `missing` 必须被调用方检查：真相源丢了不能静默当 v0.0 用 ——
       实测就踩过：文件被删后 `--bump minor` 直接算出 v0.1，
       服务端因此报出一个从没存在过的版本号。宁可报错也不要编造。
    """
    d = {"version": "v0.0", "round": "", "bump": "", "updated_at": "",
         "note": ""}
    missing = not os.path.exists(SERVER_VER)
    if missing:
        return d, True
    for line in open(SERVER_VER, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        d[k.strip()] = v.strip()
    return d, False


def bump(ver, kind):
    m = RE_VER.match(ver or "")
    if not m:
        raise SystemExit("版本号格式不对（应为 v主.次）：%r" % ver)
    a, b = int(m.group(1)), int(m.group(2))
    if kind == "major":
        return "v%d.0" % (a + 1)
    return "v%d.%d" % (a, b + 1)


def rewrite_items(paths, new_ver, dry=False, new_round=""):
    """把题库的 id 前缀 + z 批次段 + ver 统一到 new_ver / new_round。

    🔴 id 形如 `v1.1-z17-spd`：**第一段是项目版本，第二段是反馈批次**，
       两条独立的轴 —— 版本由 --bump 决定，批次由 --round 决定。
       旧实现只换版本、不换批次，升到 v3.3 后题库会变成 `v3.3-z32-*`，
       版本与批次脱节（2026-10-19 实测踩过，且 consistent 当时不报警）。
       现在两段一起同步。
    """
    for p in paths:
        if not os.path.exists(p):
            print("  ⚠️ 跳过（不存在）%s" % p)
            continue
        j = json.load(open(p, encoding="utf-8"))
        if not (isinstance(j, dict) and "items" in j):
            print("  ❌ 不是 {'items':[...]} 格式：%s" % p)
            continue
        changed = 0
        for it in j["items"]:
            old = str(it.get("id") or "")
            parts = old.split("-")
            if len(parts) >= 2 and RE_VER.match(parts[0]):
                rest = list(parts[1:])
                # 🔴 批次段只在形如 z32 时才替换，别把 `m_v4` 之类的维度当批次改掉
                if new_round and rest and RE_ROUND.match(rest[0]):
                    rest[0] = new_round
                new = "-".join([new_ver] + rest)
                if new != old:
                    it["id"] = new
                    changed += 1
            if it.get("ver") != new_ver:
                it["ver"] = new_ver
                changed += 1
        if changed and not dry:
            json.dump(j, open(p, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        print("  %s %s（改写 %d 处，%d 题）"
              % ("[dry]" if dry else "✅", os.path.basename(p), changed,
                 len(j["items"])))


def update_local_doc(new_ver, round_, note, dry=False):
    """更新本地仓库 VERSION 文档的「当前版本」行。"""
    if not os.path.exists(LOCAL_VER):
        print("  ⚠️ 本地 VERSION 文档不存在，跳过（本机才有）：%s" % LOCAL_VER)
        return
    txt = open(LOCAL_VER, encoding="utf-8").read()
    new_line = "## 当前版本：%s（%s）" % (new_ver, note or round_ or "自动升级")
    out = re.sub(r"^## 当前版本：.*$", new_line, txt, count=1, flags=re.M)
    if out != txt and not dry:
        open(LOCAL_VER, "w", encoding="utf-8").write(out)
    print("  %s 本地 VERSION 文档：%s"
          % ("[dry]" if dry else "✅", new_line))


def check():
    if not ON_SERVER:
        print("不在服务器上，跳过 live 自检（用 scp 到服务器再跑）")
        return 0
    import urllib.request
    try:
        r = json.loads(urllib.request.urlopen(API, timeout=20).read())
    except Exception as e:                                  # noqa: BLE001
        print("❌ /api/version 不可达：%r" % (e,))
        return 1
    print("服务端 version : %s" % r.get("version"))
    print("题库自报 ver   : %s（批 %s，%d 题）"
          % (r.get("fb", {}).get("ver"), r.get("fb", {}).get("round"),
             r.get("fb", {}).get("n_items")))
    miss = r.get("fb", {}).get("missing") or []
    print("缺音频         : %s" % (miss or "无"))
    if r.get("round_consistent") is False:
        print("🔴 反馈批次落后：服务端 round=%s，题库 id 里是 %s"
              % (r.get("round"), r.get("fb", {}).get("round")))
        print("   修：release_version.py --set <当前版本> --round <服务端 round>"
              " --items <题库文件>")
    if r.get("mixed_versions"):
        print("❌ 题库里混了多个版本：%s" % r.get("fb", {}).get("vers_found"))
    if r.get("consistent"):
        print("✅ consistent=true —— 凭证有效")
        return 0
    print("❌ consistent=false —— **题库标注与服务端版本不一致**")
    print("   修：release_version.py --bump minor --items <题库文件>")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bump", choices=("minor", "major"))
    ap.add_argument("--set", help="直接指定版本号（跳过规则）")
    ap.add_argument("--round", default="", help="本批发放批次，如 z18")
    ap.add_argument("--note", default="", help="本版改了什么（写进凭证）")
    ap.add_argument("--items", nargs="*", default=[], help="要统一版本的题库")
    ap.add_argument("--from", dest="base", default="",
                    help="起始版本基线。服务端真相源缺失时**必须**给，"
                         "从仓库 VERSION 文档的「当前版本」抄过来。")
    ap.add_argument("--check", action="store_true", help="只做版本凭证自检")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    if a.check:
        return check()
    if not a.bump and not a.set:
        return check()

    cur, missing = read_server()
    old = a.base or cur["version"]
    if missing and not a.base:
        raise SystemExit(
            "❌ 服务端真相源不存在：%s\n"
            "   拒绝从 v0.0 推算（那会报出一个从没存在过的版本号）。\n"
            "   请显式给基线：--from <仓库 VERSION 文档里的当前版本>"
            % SERVER_VER)
    if missing and a.base:
        print("  ⚠️ 真相源缺失，按 --from %s 建立（基线取自仓库 VERSION 文档）"
              % a.base)
    new = a.set or bump(old, a.bump)
    print("=== 版本自动升级 ===")
    print("  %s  ->  %s   (%s)" % (old, new, a.bump or "指定"))

    if a.items:
        print("  统一题库版本（版本 %s / 批次 %s）：" % (new, a.round or "不变"))
        rewrite_items(a.items, new, a.dry, a.round)

    if ON_SERVER:
        body = (
            "# tts-kr 服务端发布版本凭证（唯一真相源）\n"
            "# 由 tools/release_version.py 自动升级，勿手工改。\n"
            "# 语义见 VERSION 文档「版本号规则」+ deploy/patch_relver.py\n"
            "version = %s\n"
            "round = %s\n"
            "bump = %s\n"
            "updated_at = %s\n"
            "note = %s\n" % (new, a.round, a.bump or "set", now_bj(), a.note))
        if a.dry:
            print("  [dry] 将写入 %s：\n%s" % (SERVER_VER, body))
        else:
            if os.path.exists(SERVER_VER):
                bak = SERVER_VER + ".bak-" + old
                open(bak, "w", encoding="utf-8").write(
                    open(SERVER_VER, encoding="utf-8").read())
                print("  备份 ->", bak)
            open(SERVER_VER, "w", encoding="utf-8").write(body)
            print("  ✅ 写入", SERVER_VER)
    else:
        print("  ⚠️ 不在服务器上，未写服务端真相源")

    update_local_doc(new, a.round, a.note, a.dry)

    if ON_SERVER and not a.dry:
        print()
        print("=== 自检 ===")
        return check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
