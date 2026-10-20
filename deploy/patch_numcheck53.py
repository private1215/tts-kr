#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deploy/patch_numcheck53.py —— v3.5 病态数字自动标注接入生产（只标注不拦截）。

## 为什么
L3 文案里出现过 `전일 대비 한 사쩜사퍼센트`：native 数词「한」(1) 与 Sino 数词
「사쩜사」(4.4) 混排，组不成合法数值，TTS 忠实念出「한 사쩜사 퍼센트」——
员工听到的就是错音。ASR 侧（q2_num_audit3）证明其余数字念读全对：**问题在文案，
不在链路**。所以不修音频，只在合成前把这类数字标出来，改不改由人定。

## 定位（用户原话）
「这类可以识别标注出来即可，不是强制修改的」→ 不阻塞任务、不改文案、不动音频。

## 做了什么
1. 把 tools/numcheck53.py 复制到 app/numcheck53.py（app/ 是扁平导入，
   `from app import xxx` 会启动即崩，只能 `import numcheck53`）。
2. server.py run_job() 分块之前扫描 job["text"]：
   · 命中 -> log.warning + job["num_warnings"]（含 pos/expr/rule/sev/why/suggest）
   · 未命中 -> job["num_warnings"] = []
   · 任何异常 -> 只记 warning，**绝不让整单失败**
3. TTS_NUMCHECK=0 可关闭（默认开）。

## 验证
    python3 deploy/patch_numcheck53.py --selftest     # 纯离线，不打补丁
    python3 deploy/patch_numcheck53.py --apply        # 打补丁（.bak + 编译自检 + 回滚）
然后提交一条含「한 사쩜사퍼센트」的任务，看 /api/jobs/<id> 的 num_warnings。
"""
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "app", "server.py")
SRC_MOD = os.path.join(ROOT, "tools", "numcheck53.py")
DST_MOD = os.path.join(ROOT, "app", "numcheck53.py")

ANCHOR = ("    # ---- 自动分块：整段文案按句子边界切成若干块，"
          "每块一个独立音频文件 ----")

BLOCK = '''    # ---- v3.5 病态数字标注（只提示：不改文案、不拦任务、不动音频）----
    # 员工写「한 사쩜사퍼센트」这类 native+Sino 混排，韩语里组不成合法数值，
    # TTS 忠实念出来就是错音/怪音。合成前标出来，改不改由人定。
    # 判据见 app/numcheck53.py（40 用例 + 22 条写实句零误报）。
    if os.environ.get("TTS_NUMCHECK", "1") != "0":
        try:
            import numcheck53
            _nf = numcheck53.scan_text(job.get("text") or "")
            job["num_warnings"] = [
                {"pos": _f["pos"], "expr": _f["expr"], "rule": _f["rule"],
                 "sev": _f["sev"], "why": _f["why"], "suggest": _f["suggest"]}
                for _f in _nf]
            if _nf:
                log.warning(
                    "任务 %s 文案检出病态数字 %d 处（INVALID %d / WARN %d）：%s",
                    job["id"], len(_nf),
                    sum(1 for _f in _nf if _f["sev"] == "INVALID"),
                    sum(1 for _f in _nf if _f["sev"] == "WARN"),
                    "；".join("%r %s" % (_f["expr"], _f["sev"]) for _f in _nf[:8]))
        except Exception as _ne:                                 # noqa: BLE001
            log.warning("病态数字扫描失败（不影响合成）: %s", _ne)
'''


def selftest():
    """离线验证检测器本身可用（不打任何补丁）。"""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import numcheck53
    cases = [
        ("전일 대비 한 사쩜사퍼센트 정도 띄워주고 있습니다.", 1),
        ("반갑습니다, 아모텍 주주 여러분. 이민정입니다.", 0),
        ("코스피가 사점사 퍼센트 상승했습니다.", 0),
    ]
    ok = True
    for t, exp in cases:
        got = len(numcheck53.scan_text(t))
        flag = "✅" if got == exp else "❌"
        if got != exp:
            ok = False
        print("%s 期望%d 实得%d  %s" % (flag, exp, got, t[:40]))
    return ok


def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    apply = "--apply" in sys.argv
    if not os.path.exists(SRC_MOD):
        print("!! 找不到 %s（先把 tools/numcheck53.py 放到服务器）" % SRC_MOD)
        sys.exit(1)
    src = open(TARGET, encoding="utf-8").read()
    if ANCHOR not in src:
        print("!! 分块锚点未找到：\n%s" % ANCHOR)
        sys.exit(1)
    if "numcheck53" in src:
        print("!! 补丁看起来已经打过（server.py 里已有 numcheck53）")
        sys.exit(1)
    print("将插入标注代码 %d 行到 run_job() 分块之前" % len(BLOCK.splitlines()))
    if not apply:
        print("---- 干跑：加 --apply 实打 ----")
        print(BLOCK)
        return
    # 1) 模块复制到 app/
    shutil.copyfile(SRC_MOD, DST_MOD)
    print("模块 -> %s" % DST_MOD)
    # 2) 打补丁
    bak = TARGET + ".bak-before-numcheck53-" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copyfile(TARGET, bak)
    print("备份 -> %s" % bak)
    out = src.replace(ANCHOR, BLOCK + ANCHOR, 1)
    open(TARGET, "w", encoding="utf-8").write(out)
    # 🔴 自检必须 compile()，ast.parse() 不走符号表抓不到 await 作用域错误
    import py_compile
    try:
        py_compile.compile(TARGET, doraise=True)
        py_compile.compile(DST_MOD, doraise=True)
        print("编译自检通过")
    except Exception as e:
        print("!! 编译失败，回滚: %s" % e)
        shutil.copyfile(bak, TARGET)
        sys.exit(1)
    print("补丁完成：run_job() 分块前扫描病态数字，写入 job['num_warnings']")


if __name__ == "__main__":
    main()
