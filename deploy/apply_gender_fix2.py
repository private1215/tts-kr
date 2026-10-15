#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修「性别判定用了还没赋值的变量」（第30轮第二处回归）

## 事故
`apply_gender_fix.py` 把 `--gender` 改成按 `job["_is_male"]` 判，
但 `job["_is_male"]` 在 **1555 行**才赋值，而合成（`_synth_vox/_synth_omni`）
发生在这**之前** → 取到的一定是 `None` → **男声恒被判成女声**。

实测后果（4143 字同一文案，voxcpm 男声）：
| | Σz²(男声基线) | 4k~8k | 时长 | 字/秒 |
|---|---|---|---|---|
| EL 男标杆 | 6.8 | 0.0129 | 509s | 8.14 |
| 老男声（改之前） | **21.2** ✅ | 0.0145 | 495.8s | 8.36 |
| 新男声（我改坏后） | **3519.6** 🔴 | **0.0460(3.6x)** | 550.2s | 7.53 |

550.2s ≈ 4143 / 7.57（女声语速锚）→ 直接证明走的是女声档。
员工反馈：「尖锐，像老播放器加快速度播放，根本不是人能发出来的声音」。

## 修法
性别直接读 `voices[v]["gender"]`（音色自带，模块级随时可用），
拿不到才回退到引擎名。
"""
import ast
import sys

P = "/mnt/tts/projects/qwen3-tts/app/server.py"

HELPER = '''
def _voice_gender(job):
    """🔴 第30轮（修正2）：性别必须从**音色配置**读，不能用 job["_is_male"]。

    `job["_is_male"]` 在 run_job 里到 1555 行才赋值，而 `_synth_vox/_synth_omni`
    在这之前就跑了 —— 用它判性别，男声恒为 None → 被判成女声
    （女声 cps 7.57 / 女声 F0 276 / 女声 EQ），成品变成「尖锐的假声」。
    """
    v = voices.get((job.get("voice") or "").strip()) or {}
    g = (v.get("gender") or "").strip().lower()
    if g == "male":
        return "m"
    if g == "female":
        return "f"
    # 拿不到才回退到引擎名（vox-tts 只有男声模型）
    return "m" if (job.get("engine") or "").strip() == "voxcpm" else "f"


async def run_job(job):'''

OLD_FC = ('        # 🔴 第30轮：性别取自音色配置（job["_is_male"]），不是引擎名\n'
          '        _fc_gender = "m" if job.get("_is_male") else "f"\n')
NEW_FC = ('        _fc_gender = _voice_gender(job)\n')

OLD_OM = ('        _omni_voice = "male" if job.get("_is_male") else "female"\n')
NEW_OM = ('        _omni_voice = "male" if _voice_gender(job) == "m" else "female"\n')


def main():
    src = open(P, encoding="utf-8").read()
    if "_voice_gender(job)" in src and OLD_FC not in src:
        print("· 已修过，跳过")
        return 0

    # 1) 插入模块级 helper（放在 run_job 之前）
    anchor = "\nasync def run_job(job):"
    if anchor not in src:
        print("🔴 找不到 run_job 锚点")
        return 1
    src = src.replace(anchor, HELPER, 1)
    print("✅ 插入 _voice_gender()")

    for old, new, tag in ((OLD_FC, NEW_FC, "_fc_gender"),
                          (OLD_OM, NEW_OM, "_omni_voice")):
        if old not in src:
            print("🔴 找不到：%s" % tag)
            return 1
        src = src.replace(old, new, 1)
        print("✅ 改好：%s" % tag)

    ast.parse(src)
    open(P, "w", encoding="utf-8").write(src)
    print("✅ 语法通过，已写回 " + P)
    return 0


if __name__ == "__main__":
    sys.exit(main())
