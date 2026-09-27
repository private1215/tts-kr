# 变更记录

## 2026-09-27 修复长文本 CUDA OOM，批量/分段改为自适应

### 背景

上午 7 次任务连续失败，全部报 `CUDA out of memory`。此前已定位到根因：
崩溃点不在主干模型，而在声码器 decoder —— 生成完 codec tokens 后，
整个 batch 的 codes 被一次性 pad 对齐送进上采样网络，中间激活张量过大。

进一步从历史任务中提取出唯一决定因素：**单批 token 预算**，与文案总长度无关。

```
max_new_tokens = min(2048, max_chars * 6.0 + 96)
单批 token 预算 = 实际批段数 × max_new_tokens
```

| 结果 | 单批 tokens | 案例 |
|---|---|---|
| 失败下限 | 7560 | max_chars=110 × 10 段（1084 字） |
| 成功上限 | 7296 | max_chars=60 × 16 段（864 字） |

阈值约 **7300**，10 个任务 100% 命中。这解释了为什么把文案从 4021 字缩到
1084 字仍然崩溃 —— 缩的是段数，每段仍是 110 字 / 756 tokens，单批预算几乎没降。

### 改动

**`app/tts_engine.py`**（385 → 568 行）

- 新增标定常量：`SEG_MAX_CHARS=60` `SEG_MIN_CHARS=20` `MAX_AUTO_BATCH=8`
  `SAFE_BATCH_TOKENS=3650` `REF_HEADROOM_GIB=7.4` `OOM_MAX_RETRY=3` `OOM_THRESHOLD_TOKENS=7300`
- 新增函数 `seg_token_cap()` / `gpu_headroom_gib()` / `is_oom()`
- 新增 `plan_generation(max_chars, batch_size)` —— 把调用方参数收敛到安全区间：
  max_chars 硬夹到 `[20, 60]`；batch_size 按净余量比值动态计算并封顶 8；
  显存越脏批量越小（比值下限 0.30）
- 新增 `_gen_batch_safe(texts, bs, **kw)` —— OOM 时批量减半重试最多 3 次，
  保证返回与入参等长的结果，并回报实际跑通的批量给调用方沿用
- `synth_long` 开头调用 `plan_generation` 覆盖入参；分批改为 while 循环；
  批间 `torch.cuda.empty_cache()`；本批降级后的 batch 沿用给后续批次
- `stats` 新增 `plan` 字段，含 `peak_alloc_gib` / `alloc_now_gib` / `batches` / `batch_tokens`
- `_fix_slow_segments` 的异常处理改为捕获 OOM 并清缓存

**`app/server.py`**（904 行，CRLF 保留）

- 任务完成后把引擎实际生效的 max_chars / batch_size 回写到 job 对象
- 新增一行自适应审计日志：请求值、生效值、批数、单批 tokens、峰值显存、净余量

**`deploy/qwen-tts.service`**

- 新增 `Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8`
  （必须与 max_split_size_mb 互斥，官方文档标记为不可同时设置）

前端 UI **未改动**。员工提交的任何参数都会被后端收敛，无法再构造出 OOM 组合。

### 验收

用上午 7 次 OOM 的原始文案 + 原始参数回归，5 个唯一场景全部通过：

| 原文ID | 字数 | 提交 → 生效 | 段数 | 音频 | 生成 | RTF | 语速字/秒 | 峰值显存 |
|---|---|---|---|---|---|---|---|---|
| ee041ade41a8 | 1084 | 110/16 → 60/8 | 21 | 177.2s | 128.3s | 0.724 | 5.80 | 7.14 GiB |
| d03166ab3d45 | 1496 | 110/16 → 60/8 | 30 | 230.0s | 111.2s | 0.484 | 6.16 | 6.22 GiB |
| 3ff3497bd3dd | 2191 | 110/16 → 60/8 | 45 | 347.9s | 183.0s | 0.526 | 5.98 | 6.40 GiB |
| 0128d44632f1 | 4021 | 50/1000 → 50/8 | 96 | 626.5s | 289.3s | 0.462 | 6.04 | 6.88 GiB |
| a6e8eca89828 | 4021 | 110/16 → 60/8 | 88 | 616.5s | 301.2s | 0.489 | 6.15 | 7.29 GiB |

- **显存零累积**：每例跑完均为 `3.90 / 4.02 GiB`（allocated/reserved），
  修复前服务连续运行 41 小时涨至 11.3 GiB
- **未触发 OOM 重试**：说明 60/8 本身处於安全区，降级逻辑仅作兜底
- **RTF** 由历史均值 1.97 降至 0.48~0.72，性能反而提升
- HTTP 端到端验证通过（job `23a80503896b`，4021 字 + 错误参数 110/16）

原始文件已备份：服务端 `app/*.bak.20260927`，本地 `baseline/`。
