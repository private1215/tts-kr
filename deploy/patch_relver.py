# -*- coding: utf-8 -*-
"""patch_relver.py —— 建立**服务端发布版本凭证**（第48轮）。

## 为什么必须做（用户原话）
> 「标记统一版本，自动升级版本，还有服务端版本标识，这是后面的版本凭证，避免混乱」

## 此前的问题（实测，不是推理）
版本号此前**只写在仓库的 VERSION 文档里**，服务端一无所知。
于是「这道题标注的是哪个版本」只能靠出题脚本的环境变量 `TTS_VER`
手工传；忘了传就沿用脚本里写死的默认值。后果已经发生：

| 题库文件 | 标的 ver | 素材实际生成时的链路 | 是否失真 |
|---|---|---|---|
| items_z15.json | v1.0 | v1.0 | 一致 |
| items_z16.json | v1.1 | v1.1 omnivoice 极简 | 一致 |
| **items_z17.json** | **v1.1** | **v1.2**（含 GS 专属后处理 `gs48m_post2`） | ❌ **失真** |

员工反馈回来写「v1.1 这条有杂音」，定位到的是错误的改动清单。

## 做法
1. 服务端存一份版本号 —— `<项目根>/VERSION`（`version = v1.4`），
   由 `tools/release_version.py` **自动升级**，不再手工维护。
2. 新增 `app/relver.py`：读服务端 VERSION + 读当前 items.json 里
   题目自报的 ver，比对两者是否一致。
3. `GET /api/version` 返回完整快照；`GET /api/fb/where` 也带上版本字段
   （那个端点本来就是"改完题库先自检"的口子）。

🔴 **凭证的核心**：`"consistent": true/false`。
服务端版本与题库标注不一致时立刻可见，杜绝"标错了没人知道"。

## 附：为什么 VERSION 文件比环境变量可靠
环境变量只在进程活着时有值、`systemctl restart` 后靠 drop-in 文本维持，
而 drop-in 属主是 root（zheng 改不了）—— 实测踩过。
**文件不受这些限制，且能被 git 管起来**，天然适合当凭证。
"""
import io
import os
import sys

PROJ = "/mnt/tts/projects/qwen3-tts"
APP = os.path.join(PROJ, "app")
SRV = os.path.join(APP, "server.py")
RELV = os.path.join(APP, "relver.py")
BAK = SRV + ".bak-before-relver48"

RELVER_PY = '''# -*- coding: utf-8 -*-
"""relver.py —— 发布版本凭证（服务端唯一真相源）。

详见 deploy/patch_relver.py 文件头。

服务端版本号存在 `<项目根>/VERSION`：

    version = v1.4
    round   = z18
    note    = 发布 z18 女声 GS 裁决题（5 题）
    updated_at = 2026-10-17T09:30:00+08:00

由 tools/release_version.py 自动升级，**勿手工改**。
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
VER_FILE = os.path.join(PROJ, "VERSION")

_DEFAULTS = {"version": "v0.0", "round": "", "note": "",
             "updated_at": "", "bump": ""}


def _parse(path=None):
    d = dict(_DEFAULTS)
    try:
        with open(path or VER_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    d[k.strip()] = v.strip()
    except Exception:
        pass
    return d


def meta():
    return _parse()


def server_version():
    """服务端当前版本 —— **唯一真相源**。"""
    return _parse().get("version") or "v0.0"


def fb_state(fb_dir):
    """读当前题库，回报它**自己标注**的版本 / 批次 / 题数 / 缺文件。"""
    out = {"n_items": 0, "vers": Counter(), "rounds": Counter(),
           "missing": [], "error": None}
    ip = os.path.join(fb_dir, "items.json")
    try:
        with open(ip, encoding="utf-8") as f:
            j = json.load(f)
        items = j.get("items") or []
    except Exception as e:                                  # noqa: BLE001
        out["error"] = repr(e)[:200]
        return out
    out["n_items"] = len(items)
    for it in items:
        if it.get("ver"):
            out["vers"][it["ver"]] += 1
        parts = str(it.get("id") or "").split("-")
        if len(parts) >= 3:
            out["rounds"][parts[1]] += 1     # id 形如 v1.4-z18-noise
        for o in (it.get("options") or []):
            fn = o.get("file")
            if fn and not os.path.exists(os.path.join(fb_dir, fn)):
                out["missing"].append(fn)
    return out


def snapshot(fb_dir):
    """版本凭证快照。

    🔴 `consistent` 是核心：服务端 version 与题库自报 ver 必须一致，
       不一致说明出题时忘了同步版本号 —— 以前这种失真没人发现
       （实测 items_z17.json 就标错了，见 patch_relver.py 文件头）。
    """
    m = meta()
    sv = m.get("version") or "v0.0"
    st = fb_state(fb_dir)
    fb_ver = st["vers"].most_common(1)[0][0] if st["vers"] else ""
    fb_round = st["rounds"].most_common(1)[0][0] if st["rounds"] else ""
    mixed = len(st["vers"]) > 1
    return {
        "version": sv,
        "round": m.get("round") or fb_round,
        "updated_at": m.get("updated_at") or "",
        "note": m.get("note") or "",
        "fb_dir": fb_dir,
        "fb": {
            "n_items": st["n_items"],
            "ver": fb_ver,
            "round": fb_round,
            "vers_found": dict(st["vers"]),
            "missing": st["missing"],
            "error": st["error"],
        },
        # 🔴🔴 凭证核心
        "consistent": bool(fb_ver) and fb_ver == sv,
        "mixed_versions": mixed,
    }
'''

# ---- server.py：加 /api/version，并给 fb/where 带版本字段 --------------
FN = '''def _relver_snapshot():
    """版本凭证快照。

    🔴 import 必须写在函数内：app/ 是**扁平导入**（systemd
    `WorkingDirectory=app`），顶部写 `from app import relver` 会让服务
    启动即 ModuleNotFoundError —— 2026-10-16 踩过，**连崩 7 次**
    restart counter 打满。两条路都留着。
    """
    try:
        import relver
    except ImportError:
        try:
            from app import relver                          # noqa: PLC0415
        except Exception as _e:                             # noqa: BLE001
            return {"error": "relver 不可用: %r" % (_e,)}
    try:
        return relver.snapshot(FB_DIR)
    except Exception as _e:                                 # noqa: BLE001
        return {"error": "snapshot 失败: %r" % (_e,)}


@app.get("/api/version")
async def api_version():
    """🔴 发布版本凭证（第48轮新增）。

    返回服务端版本号 **并校验题库标注是否与之相容**：
      · consistent=true  → 服务端 version 与题目 ver 一致，凭证有效
      · consistent=false → **题库标错版本了**，先跑
                           tools/release_version.py --items <文件> 修
    员工反馈「vX 这条有问题」时，用这个端点就能定位到确切的改动清单，
    不会像 items_z17.json 那样标着 v1.1、素材却是 v1.2 产物。
    """
    return JSONResponse(_relver_snapshot(),
                        headers={"Cache-Control": "no-store"})


'''

OLD_WHERE = '''        "votes": FB_VOTES if os.path.exists(FB_VOTES) else None,
    }, headers={"Cache-Control": "no-store"})'''
NEW_WHERE = '''        "votes": FB_VOTES if os.path.exists(FB_VOTES) else None,
        # 🔴 第48轮：诊断口也带版本凭证，换题库后一眼看出标没标错
        "server_version": _relver_snapshot().get("version"),
        "fb_version": _relver_snapshot().get("fb", {}).get("ver"),
        "version_consistent": _relver_snapshot().get("consistent"),
    }, headers={"Cache-Control": "no-store"})'''


def main():
    ver_file = os.path.join(PROJ, "VERSION")
    # 1) relver.py
    if os.path.exists(RELV) and "patch_relver" in open(
            RELV, encoding="utf-8").read():
        print("  relver.py 已存在，跳过")
    else:
        open(RELV, "w", encoding="utf-8").write(RELVER_PY)
        print("  ✅ 写入", RELV)

    # 2) server.py
    src = io.open(SRV, encoding="utf-8").read()
    if "def _relver_snapshot" in src:
        print("  server.py 已打过补丁，跳过")
        return 0
    anchor = '@app.get("/api/fb/where")'
    if anchor not in src:
        print("  ❌ 找不到 /api/fb/where 锚点")
        return 1
    if OLD_WHERE not in src:
        print("  ❌ 找不到 fb_where 的 return")
        return 1
    if not os.path.exists(BAK):
        io.open(BAK, "w", encoding="utf-8").write(src)
        print("  备份 ->", BAK)
    src = src.replace(anchor, FN + anchor, 1)
    src = src.replace(OLD_WHERE, NEW_WHERE, 1)
    io.open(SRV, "w", encoding="utf-8").write(src)
    print("  ✅ 已写入", SRV)

    # 3) 服务端 VERSION（首次建立，先写当前版本 v1.3）
    if not os.path.exists(ver_file):
        io.open(ver_file, "w", encoding="utf-8").write(
            "# tts-kr 服务端发布版本凭证（唯一真相源）\n"
            "# 由 tools/release_version.py 自动升级，勿手工改。\n"
            "# 语义见 VERSION 文档「版本号规则」+ deploy/patch_relver.py\n"
            "version = v1.3\n"
            "round = \n"
            "bump = \n"
            "updated_at = \n"
            "note = 初建（与仓库 VERSION 的 v1.3 对齐）\n")
        print("  ✅ 建立", ver_file, "(v1.3)")
    else:
        print("  VERSION 已存在，未动")
    return 0


if __name__ == "__main__":
    sys.exit(main())
