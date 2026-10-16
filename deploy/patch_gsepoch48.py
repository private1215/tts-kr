# -*- coding: utf-8 -*-
"""patch_gsepoch48.py —— GS 训练 epoch 自适应：让各组看到**相近的训练步数**。

## 为什么
`tools/gs_run_group.sh` 里 `d["train"]["epochs"] = 40` 对所有人写死。
但各组数据量差很多（第48轮 tools/gscover48.py 实测）：

| 组 | 训练段数 | 步/轮（batch=8） | e40 总步数 |
|---|---|---|---|
| SOLA 女 | 1223 | 154 | **6160** |
| yohan 男 | 953 | 119 | **4840**（少 21%） |

男声素材只有 11 条（13 条 − hold 1 − CER 剔除 ba24c092 1），
预算 160 分钟还剩 68 分钟富余但**没有更多男声案例可用**，
所以只能靠多跑几轮把步数补上。

## 触发这个问题的现象（tools/clickpos48.py 实测）
| | click 次数 | 时长 | 次/秒 |
|---|---|---|---|
| 男声 gs48m_post2 | 129 | 182.2s | **0.71** |
| 女声 gs48f_eq | 79 | 458.4s | **0.17** |

同一引擎，男声是女声的 **4 倍**。定位结论：
- 发声中占比 **100%**（不是停顿里的硬切）
- 周期性只有 19%（>60% 才是块边界硬切）→ **不是拼接问题**
→ 按 clickpos48.py 自己的判据，这是**合成器固有**，只能靠重训练。

## 改法
epochs = clamp(round(6160 / (训练段数 // 8)), 40, 80)
- 下限 40：SOLA 这种大组行为**完全不变**（6160/171=36 → 取 40）
- yohan：6160/119 = 51.8 → **52 轮**
- 上限 80：防小数据集过拟合

🔴 这是**假设**，不是定案。今晚夜训跑完，明早用同一条男声文案复测 click，
   没降就把下限锁死回 40（改 GS_EPOCH_MIN 环境变量即可，不用改代码）。
"""
import io
import os
import sys

SH = "/mnt/tts/projects/qwen3-tts/tools/gs_run_group.sh"

OLD = '''d["train"]["batch_size"] = 8
d["train"]["epochs"] = 40
'''

NEW = '''d["train"]["batch_size"] = 8
# 🔴 epoch 自适应（第48轮）：让每个说话人组看到的**训练步数**相近，
#    而不是 epoch 数相同。详见 deploy/patch_gsepoch48.py 文件头。
#    SOLA 1223 段 → 154 步/轮 → e40 = 6160 步；
#    yohan 953 段 → 119 步/轮 → e40 只有 4840 步（少 21%），
#    而男声成品 click 密度是女声的 4 倍（0.71 vs 0.17 次/秒），
#    定位结论为「合成器固有、非拼接」→ 最可能是 yohan 训练不足。
import os as _os
_n = sum(1 for _l in open("$DATA/train.list", encoding="utf-8") if _l.strip())
_spe = max(1, _n // 8)
_lo = int(_os.environ.get("GS_EPOCH_MIN", "40"))
_hi = int(_os.environ.get("GS_EPOCH_MAX", "80"))
d["train"]["epochs"] = max(_lo, min(_hi, round(6160 / _spe)))
print("自适应 epoch：train %d 段 → %d 步/轮 → %d 轮（%d 步）"
      % (_n, _spe, d["train"]["epochs"],
         d["train"]["epochs"] * _spe))
'''


def main():
    src = io.open(SH, encoding="utf-8").read()
    bak = SH + ".bak-before-gsepoch48"
    if NEW in src:
        print("已打过补丁，跳过")
        return 0
    if OLD not in src:
        print("❌ 找不到锚点（epochs = 40）")
        return 1
    if not os.path.exists(bak):
        io.open(bak, "w", encoding="utf-8").write(src)
        print("备份 ->", bak)
    io.open(SH, "w", encoding="utf-8").write(src.replace(OLD, NEW, 1))
    print("✅ 已写入", SH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
