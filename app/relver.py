# -*- coding: utf-8 -*-
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
    # 🔴 v3.3 起：**批次也是凭证**。项目版本（v3.3）与反馈批次（z33）是两条
    #   独立的轴：版本对了但批次还留在上一批，等于「题问的是 v3.3、却挂在上一批
    #   发出去的清单里」，事后没法按批次回溯。旧实现只看版本，这种混合态不报警。
    ver_ok = bool(fb_ver) and fb_ver == sv
    round_ok = (not m.get("round")) or (not fb_round) or (
        fb_round == m.get("round"))
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
        "consistent": ver_ok and (not mixed) and round_ok,
        "version_consistent": ver_ok,
        "round_consistent": round_ok,
        "mixed_versions": mixed,
    }
