#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第31轮：每处停顿的填充噪声用**各自的随机种子**（幂等）

## 缺陷
`pause_clean_v3.synth_tone(n, sr, rms, seed=42)` 的默认种子是常量 42，
两处调用（`clean_gaps` 第70行、`scrub_stray` 第145行）**都没传 seed**。

而 `synth_tone` 内部是：
```python
rng = np.random.default_rng(seed)
x = rng.standard_normal(n + pad)     # 同一种子 → 同样的前 N 个数
...
y = y[pad // 2: pad // 2 + n]        # 取固定偏移处的一段
```
→ 长度相近的停顿拿到的是**同一段噪声波形**。

实测长篇（495s）有 **288~290 处停顿**被清洗，也就是说同一段噪声在成品里
重复出现了近 300 次。**重复出现的相同噪声正是耳朵最容易察觉的「合成感」**，
而且它是员工说的「卡顿 / 不像真人」的合理候选来源之一。

## 修法
按停顿的**起始时间**派生种子（确定性、可复现，但每处都不同）：
    seed = 0x9E3779B1 ^ int(a * 1000)
两处调用都改。`scrub_stray` 用碎片起始帧号派生。

## 风控
- 只改随机种子，**不改电平、不改长度、不改谱形**
  → 时长 / 语速 / 停顿节奏 / 频谱指标全部不受影响
- 确定性：同一输入永远得到同一输出（便于复现与回归）

## 用法
    ./venv/bin/python deploy/apply_toneseed31.py
"""
import ast
import sys

P = "/mnt/tts/projects/qwen3-tts/app/pause_clean_v3.py"

OLD1 = "        tone = synth_tone(i1 - i0, sr, target)"
NEW1 = ("        # 🔴 第31轮：每处停顿用**自己的**种子。\n"
        "        #    原来两处调用都不传 seed → 全篇近 300 处停顿填的是\n"
        "        #    **同一段噪声波形**（同一种子 + 固定偏移取片）。\n"
        "        #    重复出现的相同噪声 = 耳朵最容易察觉的合成感。\n"
        "        _sd = (0x9E3779B1 ^ int(a * 1000.0)) & 0xFFFFFFFF\n"
        "        tone = synth_tone(i1 - i0, sr, target, seed=_sd)")

OLD2 = "        tone = synth_tone(i1 - i0, sr, 10 ** (lvl / 20.0))"
NEW2 = ("        _sd = (0x85EBCA6B ^ (a * 2654435761)) & 0xFFFFFFFF\n"
        "        tone = synth_tone(i1 - i0, sr, 10 ** (lvl / 20.0), seed=_sd)")


def main():
    s = open(P, encoding="utf-8").read()
    if "_sd = (0x9E3779B1" in s:
        print("已打过补丁，跳过（幂等）")
        return 0
    n = 0
    for old, new in ((OLD1, NEW1), (OLD2, NEW2)):
        if old in s:
            s = s.replace(old, new, 1)
            n += 1
        else:
            print("❌ 未找到：%s" % old.strip()[:60])
    if n != 2:
        print("❌ 只替换 %d/2，放弃" % n)
        return 1
    try:
        ast.parse(s)
    except SyntaxError as e:
        print("❌ 语法错误 %s" % e)
        return 1
    open(P, "w", encoding="utf-8").write(s)
    print("✅ 已写入 %s（%d 处，语法 OK）" % (P, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
