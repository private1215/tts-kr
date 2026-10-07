# -*- coding: utf-8 -*-
"""把 SFT 检查点变成**可推理的模型目录**（base 目录符号链接 + 只换 llm.pt）。

为什么这样做
------------
模型权重 8 GB+，复制一份既慢又占盘。做法是在新目录里把 base 的其他文件
**符号链接**过去，只有 `llm.pt` 是真实文件（SFT 只训 llm）。
回滚 = 换回原来的 llm.pt，或者干脆把整个目录删掉。

用法：
    venv/bin/python tools/sft/make_model.py \
        --ckpt /mnt/tts/projects/cosy/repo/examples/ko_male/cosyvoice3/exp/sweep_1e-5/epoch_0_step_60.pt \
        --out  /mnt/tts/models/Fun-CosyVoice3-0.5B-sft-male

会把 base 的 llm.pt 先备份成 llm.pt.base（只在首次），所以可以随时回退：
    ln -sf llm.pt.base <out>/llm.pt
"""
import os
import sys
import json
import shutil
import argparse

import torch

# 训练检查点里只比原 llm.pt 多 epoch / step 两个键，直接顶替即可
DROP_KEYS = ()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="/mnt/tts/models/Fun-CosyVoice3-0.5B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if not os.path.isfile(a.ckpt):
        print("🔴 检查点不存在:", a.ckpt)
        return 1
    os.makedirs(a.out, exist_ok=True)

    # 1) 链接 base 里除 llm*.pt 之外的所有条目
    for name in sorted(os.listdir(a.base)):
        if name.startswith("llm") and name.endswith(".pt"):
            continue
        src, dst = os.path.join(a.base, name), os.path.join(a.out, name)
        if os.path.islink(dst) or os.path.exists(dst):
            continue
        os.symlink(src, dst)
        print("  link", name)

    # 2) base 的 llm.pt 备份一次，方便回滚
    base_llm = os.path.join(a.out, "llm.pt.base")
    if not os.path.exists(base_llm):
        shutil.copy2(os.path.join(a.base, "llm.pt"), base_llm)
        print("  备份 base llm.pt -> llm.pt.base")

    # 3) 校验检查点的键，然后顶替
    print("  读检查点 ...", flush=True)
    sd = torch.load(a.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    keys = sorted(sd.keys())
    extra = [k for k in keys if k in ("epoch", "step")]
    print(f"  {len(keys)} 个键，其中 {extra} 是训练元信息")
    dst = os.path.join(a.out, "llm.pt")
    if os.path.exists(dst) and not a.force:
        print("🔴 llm.pt 已存在，加 --force 覆盖")
        return 1
    torch.save(sd, dst)
    print(f"  ✅ 写入 {dst}  ({os.path.getsize(dst)/1024**3:.2f} GB)")

    json.dump({"base": a.base, "ckpt": a.ckpt,
               "keys": len(keys), "extra": extra},
              open(os.path.join(a.out, "sft_meta.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n✅ SFT 模型目录就绪: {a.out}")
    print("   回滚: ln -sf llm.pt.base %s/llm.pt" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
