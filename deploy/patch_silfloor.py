#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 silfloor 落到链路里 + 修 declick 的过零漏洞。幂等，可重复跑。

① app/silfloor.py           ← 新模块（从 deploy 复制）
② app/declick.py            ← floor 模式改成整段替换（去掉会挖坑的淡入淡出）
③ app/server.py             ← import silfloor + 在 declick 之后调用

跑法：/mnt/tts/projects/qwen3-tts/venv/bin/python patch_silfloor.py
"""
import os
import re
import shutil
import sys

APP = "/mnt/tts/projects/qwen3-tts/app"
SRC = "/mnt/tts/diag/silfloor.py"
BAK = "/mnt/tts/diag/_bak44"
os.makedirs(BAK, exist_ok=True)
changed = []


def backup(p):
    b = os.path.join(BAK, os.path.basename(p) + ".bak")
    if not os.path.exists(b):
        shutil.copyfile(p, b)
        print("  备份 -> %s" % b)


# ---------- ① silfloor.py ----------
dst = os.path.join(APP, "silfloor.py")
shutil.copyfile(SRC, dst)
print("① 写入 %s (%d 字节)" % (dst, os.path.getsize(dst)))
changed.append(dst)

# ---------- ② declick.py ----------
p = os.path.join(APP, "declick.py")
s = open(p, encoding="utf-8").read()
if "第44轮修漏洞" in s:
    print("② declick.py 已修过，跳过")
else:
    backup(p)
    old = """        if mode == "replace":
            out[a:b] = tone
        else:                            # floor：逐样本幅度下限
            seg = out[a:b]
            out[a:b] = np.where(np.abs(tone) > np.abs(seg), tone, seg)"""
    new = """        # 🔴 第44轮修漏洞：原写法 where(|tone| > |seg|, tone, seg) 是「逐样本取较大」，
        #    而 tone 中间会过零、两端又被 env_r 淡到 0 —— |tone| 掉到 |seg| 以下时
        #    数字零就原样留下了（实测成品残留 32 帧 <−100dB，员工听成「断了一下」）。
        #    门限已是 −100dB，段内本来就没有任何可听内容，整段替换才是正确做法。
        #    淡入淡出也必须去掉：淡到 0 = 在段的两端又各补一个数字零。
        out[a:b] = tone"""
    if old not in s:
        print("  !! declick.py 没匹配到目标片段，人工检查")
        sys.exit(1)
    s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)
    print("② 修好 declick.py 的过零漏洞")
    changed.append(p)

# ---------- ③ server.py ----------
p = os.path.join(APP, "server.py")
s = open(p, encoding="utf-8").read()
if "import silfloor" in s:
    print("③ server.py 已接入，跳过")
else:
    backup(p)
    old_imp = "import declick            # 去咔：给数字零的停顿段补环境声（r28）"
    if old_imp not in s:
        m = re.search(r"^import declick.*$", s, re.M)
        if not m:
            print("  !! 找不到 import declick，人工检查")
            sys.exit(1)
        old_imp = m.group(0)
    s = s.replace(old_imp, old_imp + "\n"
                  "import silfloor           # r44：句间停顿地板抬起（治「断句有异声」）",
                  1)
    old_call = "                    _dx = declick.apply(_dx, _dsr)"
    new_call = (old_call + "\n"
                "                    # 🔴 2026-10-16 第44轮 silfloor：停顿「地板抬起」。\n"
                "                    #    我们的停顿穿过 −52dB 继续下沉（−73dB，甚至数字零），\n"
                "                    #    EL 是衰减到 −52dB 就平躺。沉下去后语音再起来落差 50~220dB\n"
                "                    #    → 听感上第一个音节「突然蹦出来」（员工：突然出来短音/额/啊）。\n"
                "                    #    实测：停顿 p5 −73.5→−59.1（EL −60.5）、<−100dB 帧 32→0\n"
                "                    #    （EL 0）、点名三处 −73/−240/−72 → −59/−59/−60；\n"
                "                    #    L1 Σz² 32.2→27.8、平坦度 +3.85σ→+3.22σ（都变好）。\n"
                "                    #    必须紧跟 declick：同一条链、同样两条分支都经过。\n"
                "                    if silfloor.ENABLED:\n"
                "                        _sf0 = silfloor.stats(_dx, _dsr)\n"
                "                        _dx = silfloor.apply(_dx, _dsr)\n"
                "                        _sf1 = silfloor.stats(_dx, _dsr)\n"
                "                        log.info(\"任务 %s 第 %d 块 silfloor: 深谷 %d→%d 段, \"\n"
                "                                 \"<−100dB %d→%d 帧, 停顿 p5 %.1f→%.1f dB\",\n"
                "                                 jid, ci + 1, _sf0[0], _sf1[0], _sf0[1], _sf1[1],\n"
                "                                 _sf0[2], _sf1[2])")
    if old_call not in s:
        print("  !! 找不到 declick.apply 调用点，人工检查")
        sys.exit(1)
    s = s.replace(old_call, new_call, 1)
    open(p, "w", encoding="utf-8").write(s)
    print("③ server.py 接入 silfloor")
    changed.append(p)

print("\n改动的文件：")
for c in changed:
    print("  " + c)
print("\n下一步：python -c \"import app.server\" 语法自检 → 重启 qwen-tts / omni-tts / vox-tts")
print("PATCH_DONE")
