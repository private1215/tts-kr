#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 server.py 的老后处理段套上 `if _legacy_post:` / `else:`。

新引擎（voxcpm / omnivoice）在 _synth 内部已经跑完 fix_chain
（字/秒锚 + 变调 + 停顿清洗 + EQ + 响度），老后处理链全是给 Qwen/Cosy 的
韵律缺陷调的，再叠一遍只会把已对齐的指标推歪，所以整段跳过。

用**内容锚点**定位，不用行号（行号会随前面的插入漂移）。
"""
import io
import sys

P = "/Users/mac/Desktop/oulian-js/tts-kr/app/server.py"
src = io.open(P, encoding="utf-8").read().split("\n")

START_ANCHOR = "        _pba_ratio = 1.0"
END_ANCHOR = '        mp3_name = f"{base_name}-{ci + 1}.mp3"'

i0 = next(i for i, l in enumerate(src) if l == START_ANCHOR)
i1 = next(i for i, l in enumerate(src) if l == END_ANCHOR)
print("锚点: 老后处理 %d ~ %d（共 %d 行）" % (i0 + 1, i1, i1 - i0))

body = ["    " + l if l.strip() else l for l in src[i0:i1]]

new = (
    src[:i0]
    + [
        "        # ---- 新引擎绕开整套老后处理 ----------------------------------",
        "        # 语速/变调/停顿清洗/频谱/响度已在 fix_chain 里做完（见 _synth_vox/omni）。",
        "        # 老链（PBA/包络/slowdyn/dyn/韵律移植/呼吸/响度归一）是为 Qwen、Cosy 的",
        "        # 韵律缺陷调出来的，叠在新引擎上只会把刚对齐的指标重新推歪。",
        "        _legacy_post = _engine_name not in NEW_ENGINES",
        "        if _legacy_post:",
    ]
    + body
    + [
        "        else:",
        "            # fix_chain 已做语速与响度，这里不要再叠 atempo / 增益",
        "            _atempo = 1.0",
        "            _gain_db = 0.0",
        "            _pba_ratio = 1.0",
        "",
    ]
    + src[i1:]
)

io.open(P, "w", encoding="utf-8").write("\n".join(new))
print("写入完成，行数 %d -> %d" % (len(src), len(new)))
