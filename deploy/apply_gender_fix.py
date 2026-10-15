#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修「男声被当女声处理」的根因（2026-10-15 第30轮）

## 事故
第 29 轮 `swap_engine.py` 把 `EL_male.engine` 从 `voxcpm` 换成 `omnivoice`。
但代码里有两处**按引擎名判性别**的硬编码：

1. `_run_fix_chain`：`--gender "m" if _engine_name == "voxcpm" else "f"`
   → 男声走 omnivoice 时被当成**女声**，F0 目标被设成 **276Hz**（男声应是 197.7Hz）、
     LUFS/EQ/语速锚也全用女声档。
2. `_synth_omni`：`omni_client.synth(..., voice="female")` 写死。
   → 而 **OmniVoice 服务端只加载了 `["female"]`**（`/health` 实测），
     也就是说 omnivoice **根本没有男声**。

实测后果（4143 字同一文案）：
| | F0 中位 | 字/秒 |
|---|---|---|
| EL 男标杆 6cdd | **179.9 Hz** | 8.14 |
| 我们「男声」（换引擎后） | **277.3 Hz** | 7.43 |
| 我们女声 | 277.5 Hz | 7.45 |
→ **男声和女声的 F0 一模一样，比 EL 男声高 54%。**
→ 第 28/29 轮员工拿「我们的女声」跟「EL 男声」比，反馈被污染。

## 本脚本
把性别判定从「引擎名」改成「音色配置里的 gender / job["_is_male"]」，
并给「omnivoice + 男声」加一条响亮警告（omnivoice 没有男声，不该这么配）。
"""
import ast
import sys

P = "/mnt/tts/projects/qwen3-tts/app/server.py"

OLD1 = ('                         "--gender", "m" if _engine_name == "voxcpm" else "f",\n'
        '                         "--target-cps", str(NEW_ENGINE_CPS.get(_engine_name, 8.0)),\n')

NEW1 = ('                         # 🔴 第30轮：性别必须按**音色配置**判，不能按引擎名。\n'
        '                         #    换引擎前 voxcpm=男 / omnivoice=女 恰好成立，\n'
        '                         #    一旦把 EL_male 配成 omnivoice，男声就会被按女声处理\n'
        '                         #    （F0 目标 276Hz、语速锚 7.57、EQ/LUFS 全用女声档）。\n'
        '                         "--gender", _fc_gender,\n'
        '                         "--target-cps", str(NEW_ENGINE_CPS_BY_GENDER[_fc_gender]),\n')

OLD1b = ('    NEW_ENGINE_CPS = {"voxcpm": 8.37, "omnivoice": 7.57}\n')
NEW1b = ('    NEW_ENGINE_CPS = {"voxcpm": 8.37, "omnivoice": 7.57}\n'
         '    # 🔴 第30轮：语速锚按**性别**取（EL 实测男 8.37 / 女 7.57），\n'
         '    #    不再按引擎名——引擎和性别是可以任意组合的。\n'
         '    NEW_ENGINE_CPS_BY_GENDER = {"m": 8.37, "f": 7.57}\n')

OLD2 = ('        audio, sr, meta = omni_client.synth(\n'
        '            text=text_ci[1], voice="female",\n'
        '            on_progress=_mk_progress(text_ci[0]))\n')
NEW2 = ('        # 🔴 第30轮：voice 按**音色性别**传，不能写死 female。\n'
        '        #    ⚠️ OmniVoice 服务端目前只加载了 ["female"]（/health 实测），\n'
        '        #       男声配 omnivoice 会**静默产出女声**，这里大声报警。\n'
        '        _omni_voice = "male" if job.get("_is_male") else "female"\n'
        '        if _omni_voice == "male":\n'
        '            log.error("任务 %s 🔴 男声配了 omnivoice，而 OmniVoice 只有女声模型，"\n'
        '                      "产出会是女声！请把 EL_male.engine 改回 voxcpm", jid)\n'
        '        audio, sr, meta = omni_client.synth(\n'
        '            text=text_ci[1], voice=_omni_voice,\n'
        '            on_progress=_mk_progress(text_ci[0]))\n')


def main():
    src = open(P, encoding="utf-8").read()
    if "_fc_gender" in src:
        print("· 已改过，跳过")
        return 0
    for old, new, tag in ((OLD1b, NEW1b, "语速锚常量"),
                          (OLD1, NEW1, "fix_chain 性别/语速"),
                          (OLD2, NEW2, "omni voice 参数")):
        if old not in src:
            print("🔴 找不到：%s" % tag)
            return 1
        src = src.replace(old, new, 1)
        print("✅ 改好：%s" % tag)

    # 在 _run_fix_chain 里算出 _fc_gender（放在 _sf.write 之前）
    anchor = '        try:\n            _sf.write(src, audio, sr)\n'
    if anchor not in src:
        print("🔴 找不到 _sf.write 锚点")
        return 1
    decl = ('        # 🔴 第30轮：性别取自音色配置（job["_is_male"]），不是引擎名\n'
            '        _fc_gender = "m" if job.get("_is_male") else "f"\n')
    src = src.replace(anchor, decl + anchor, 1)
    print("✅ 加好：_fc_gender 计算")

    ast.parse(src)
    open(P, "w", encoding="utf-8").write(src)
    print("✅ 语法通过，已写回 " + P)
    return 0


if __name__ == "__main__":
    sys.exit(main())
