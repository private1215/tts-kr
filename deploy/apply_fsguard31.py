#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第31轮：给 `fs` 加空表兜底（幂等，纯防护）

`syl_rates` 已有 `if not out and voiced > 0` 的兜底，但**全静音**时
`voiced == 0` → `out` 仍为空 → `fs = []`。
后面 `bounds`/`segs` 是按 `len(fs)` 切的，空表会让 `segs` 也为空，
`np.concatenate([])` 直接崩，整单失败（而 fix_chain 失败会退回裸音频，
员工拿到的就是没打磨的版本）。

第31轮细化变调窗后又多了一处 `fs[...]` 索引，更需要这个兜底。

## 用法
    ./venv/bin/python deploy/apply_fsguard31.py
"""
import ast
import sys

P = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"

OLD = '''    fs = [(fs[i - 1] + 2 * fs[i] + fs[i + 1]) / 4 if 0 < i < len(fs) - 1 else fs[i]
          for i in range(len(fs))]
    print(f"模式: {mode}")'''

NEW = '''    fs = [(fs[i - 1] + 2 * fs[i] + fs[i + 1]) / 4 if 0 < i < len(fs) - 1 else fs[i]
          for i in range(len(fs))]
    # 🔴 第31轮：空表兜底。`syl_rates` 只在 `voiced > 0` 时补尾巴，
    #    全静音时 `out` 仍为空 → `fs=[]` → 下面 `segs` 也为空 →
    #    `np.concatenate([])` 崩溃，整单退回裸音频。
    if not fs:
        fs = [1.0]
        print("[warn] 速率窗口为空（素材近乎全静音？），按 1.0 处理", flush=True)
    print(f"模式: {mode}")'''


def main():
    s = open(P, encoding="utf-8").read()
    if "速率窗口为空" in s:
        print("已打过补丁，跳过（幂等）")
        return 0
    if OLD not in s:
        print("❌ 未定位到 fs 平滑处")
        return 1
    s = s.replace(OLD, NEW, 1)
    try:
        ast.parse(s)
    except SyntaxError as e:
        print("❌ 语法错误 %s" % e)
        return 1
    open(P, "w", encoding="utf-8").write(s)
    print("✅ 已写入 %s（语法 OK）" % P)
    return 0


if __name__ == "__main__":
    sys.exit(main())
