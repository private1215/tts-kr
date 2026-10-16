# -*- coding: utf-8 -*-
"""patch_gscover48.py —— GS 训练数据两项修正（第48轮体检发现）。

## 发现（tools/gscover48.py 实测）
案例库 36 条 / 293.5 分钟，按声音聚类：SOLA 女 22 条、yohan 男 13 条、elara 1 条。
`gs_prep_group.py --max-minutes 160` 是**按案例 id 排序**逐条累加、到预算就 break：

| 组 | 可用 | 需要 | 实际用 | 损失 |
|---|---|---|---|---|
| SOLA | 21 条（hold 1 条） | **174.4 min** | 160 min | **ea3154b3 整条 8.5 min 被丢** + ce1a3fda 只切 2.9/8.8 min |
| yohan | 11 条（CER 剔 ba24c092） | 91.8 min | 91.8 min | 无（预算有 68 min 富余，但没有更多男声素材） |

🔴 被丢的是 EL 生成的真素材，GS 微调**唯一的养料**，白扔 14 分钟很亏。

## 修正 1：预算 160 → 180 分钟
180 > 174.4，SOLA 21 条全部完整进训练集。
代价：SOLA 夜间 prep+train 从约 144 min → 约 162 min（+12%），
凌晨 3 点起跑仍能在上班前完成。yohan 组不受影响（远低于预算）。

## 修正 2：按质量排序占预算，而不是按 id
原来 `sorted(cids)` = 随机顺序，预算不足时丢谁全看 id 字母。
改成按「该案例最差 segment 的 CER」**升序**：文本与音频对得越准的越先占预算，
万一还是不够，被丢的就是最脏的那条。
（全库除 ba24c092 0.728 外都 ≤0.154，所以这改动是**保险**，不是大修。）

开关：无。改的是训练数据准备，GS 当前不是线上引擎，风险低。
"""
import io
import os
import sys

TOOLS = "/mnt/tts/projects/qwen3-tts/tools"
PREP = os.path.join(TOOLS, "gs_prep_group.py")
AUTO = os.path.join(TOOLS, "gs_auto_train.py")
NIGHT = "/mnt/tts/projects/qwen3-tts/deploy/gs_night_train.sh"

# ---- 修正 1：预算 --------------------------------------------------------
# 用正则，避免空格数对不上（历史上踩过"看着一样其实差一个空格"）
import re  # noqa: E402

RE_MM = re.compile(r"^MAX_MINUTES\s*=\s*160\.0", re.M)
NEW_MM = ("MAX_MINUTES = 180.0       # 训练音频上限（分钟），防训练过久\n"
          "#   🔴 160 -> 180（第48轮体检）：160 分钟装不下 SOLA 组 21 条\n"
          "#   （实际需要 174.4 min），末尾 ea3154b3 整条 8.5 min 被 break 丢掉，\n"
          "#   ce1a3fda 还被硬切成 2.9/8.8 min。180 分钟可全量纳入。")
OLD_NIGHT = "--max-minutes 160 --exclude-cer 0.35 --hold 00b6a45ad6aa"
NEW_NIGHT = "--max-minutes 180 --exclude-cer 0.35 --hold 00b6a45ad6aa"

# ---- 修正 2：质量优先排序 ------------------------------------------------
# cmap 原先只在 exclude_cer>0 分支内赋值，排序代码会 NameError
OLD_CMAP = ("    # ---- CER 质量门槛：文本与音频对不上的案例，整条剔除 "
            "--------------------")
NEW_CMAP = "    cmap = {}\n" + OLD_CMAP

OLD_TAIL = '''        if len(kept) < len(train_cids):
            print("CER 过滤：%d -> %d 条" % (len(train_cids), len(kept)),
                  flush=True)
        train_cids = kept
'''
NEW_TAIL = '''        if len(kept) < len(train_cids):
            print("CER 过滤：%d -> %d 条" % (len(train_cids), len(kept)),
                  flush=True)
        train_cids = kept

    # ---- 排序：质量好的先占预算 ------------------------------------------
    # 🔴 下面的预算循环是 `for cid in train_cids: if got >= budget: break`，
    #    预算不足时**丢掉排序末尾的案例**。原顺序是 sorted(cid)，等于随机，
    #    可能把最准的素材丢掉（第48轮实测：SOLA 组就这么丢了 ea3154b3 整条）。
    #    改成按「该案例最差 segment 的 CER」升序：对得越准越先占预算，
    #    万一还不够，被丢的自然是最脏的那条。
    def _worst_cer(_cid):
        _c = cmap.get(_cid) or {}
        _cs = [s.get("cer") or 0.0 for s in (_c.get("segments") or [])]
        return max(_cs) if _cs else 0.0
    if cmap:
        train_cids = sorted(train_cids, key=lambda c: (_worst_cer(c), c))
        print("按质量排序（最差 segment CER 升序），前 3 条 CER：%s"
              % ", ".join("%s=%.3f" % (c[:8], _worst_cer(c))
                          for c in train_cids[:3]), flush=True)
'''


BAK = ".bak-before-gscover48"


def edit(path, old, new, label):
    """old 可以是 str 或编译好的正则（有 .sub 方法）。"""
    src = io.open(path, encoding="utf-8").read()
    bak = path + BAK
    hit = (old.search(src) if hasattr(old, "search") else (old in src))
    if not hit:
        done = (new in src) or (hasattr(old, "search") and not old.search(src)
                                and "180.0" in src)
        if done:
            print("  已打过（%s）" % label)
            return 0
        print("  ❌ 找不到锚点（%s）" % label)
        return 1
    if not os.path.exists(bak):
        io.open(bak, "w", encoding="utf-8").write(src)
    out = (old.sub(new, src, count=1) if hasattr(old, "sub")
           else src.replace(old, new, 1))
    io.open(path, "w", encoding="utf-8").write(out)
    print("  ✅ %s -> %s" % (label, path))
    return 0


def main():
    rc = 0
    rc |= edit(AUTO, RE_MM, NEW_MM, "MAX_MINUTES 160->180")
    rc |= edit(NIGHT, OLD_NIGHT, NEW_NIGHT, "夜训脚本预算 160->180")
    # cmap 只在 exclude_cer>0 时才被赋值；先初始化，保证排序永远拿得到
    rc |= edit(PREP, OLD_CMAP, NEW_CMAP, "cmap 初始化")
    rc |= edit(PREP, OLD_TAIL, NEW_TAIL, "质量优先排序")
    return rc


if __name__ == "__main__":
    sys.exit(main())
