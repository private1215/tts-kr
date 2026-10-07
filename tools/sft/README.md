# CosyVoice3 SFT 流水线（重建版，2026-10-07）

> 🔴 **本目录存在的原因**：上一版脚本全在 `/tmp/sftfiles/`，2026-10-07 服务器
> 重启后**全部丢失**，已经跑出来的 316 条数据和 lr=1e-5 step 70 的检查点都没了。
> 训练脚本是项目资产，必须进仓库。所有中间产物也一律写 `/mnt/tts/**`，不写 /tmp。

## 0. 现在的结论（上一轮实测，别再重踩）

| | CV loss | acc |
|---|---|---|
| 预训练基线（lr=0 实测） | 3.043 | 0.251 |
| **lr=1e-5，step 70（≈ 1 epoch）** | **2.401** ✅ | 0.337 |
| lr=1e-5，epoch 5~9 | 3.41 → 4.53 | 0.297 → 0.279 |

🔑 **SFT 有效，但甜点在 1 个 epoch 之内。** 官方 `max_epoch=40` 会灾难性过拟合，
大约 epoch 5 就比不训还差。**不是 SFT 没用，是训过头了。**

男声同人数据 ≈ 76 分钟（10 个个案，campplus 余弦 0.973~0.987 全是同一人）→ 已过门槛。
女声只有 1 个案例 7.65 分钟，还差约 3 例才能做。

## 1. 流程

```bash
cd /mnt/tts/projects/qwen3-tts

# ① 建数据集（留一验证默认拿最后一个案例）
venv/bin/python tools/sft/prep.py --out /mnt/tts/data/sft/male --gender male
venv/bin/python tools/sft/prep.py --out /mnt/tts/data/sft/male --verify   # CER 抽检

# ② 提特征 → parquet（会调 CosyVoice3 官方脚本，路径坑已在脚本里修）
bash tools/sft/build.sh /mnt/tts/data/sft/male

# ③ 训练（脚本自带限功耗检查 + 日志落 /mnt/tts/diag）
bash tools/sft/train.sh /mnt/tts/data/sft/male

# ④ 检查点 → 可推理模型目录（符号链接 + 只换 llm.pt，可回滚）
venv/bin/python tools/sft/make_model.py --ckpt /mnt/tts/data/sft/exp/.../epoch_0_step_70.pt \
                                        --out /mnt/tts/models/Fun-CosyVoice3-0.5B-sft-male
```

## 2. 环境坑（上一轮踩了 7 个，逐个都有对策）

| # | 坑 | 对策 |
|---|---|---|
| 1 | deepspeed 装不上：0.19.x 要 `Dynamo`（Py3.12 不支持）；0.14.x 要 `CUDA_HOME`（pip 版 torch 不带 nvcc） | 用 `stub/deepspeed/` 最小替身，靠 `PYTHONPATH` 前置注入 |
| 2 | 替身的 `add_config_arguments` 重复注册 `--deepspeed.save_states` → argparse 冲突 | 写成**幂等**（先检查 `parser._actions`）|
| 3 | 缺 pandas | `pip install pandas==2.2.3 pytz python-dateutil tzdata --no-deps`（不要动 numpy 1.26.4）|
| 4 | tensorboard 2.21 要 protobuf 6.x，与运行时 4.25 冲突 | **降到 tensorboard 2.16.2 + protobuf 4.25.0** |
| 5 | 官方 `run.sh` 里 `tools/extract_embedding.py` 路径相对 CWD，实际 tools 在仓库根 | 改 `../../../tools/` |
| 6 | 单卡 `torchrun` 段错误（`c10d_rendezvous_backend.py:113 _call_store`）| 直跑 `python train.py` + 显式 `MASTER_ADDR/PORT/WORLD_SIZE/RANK/LOCAL_RANK` |
| 7 | `KeyError: 'instruct_token'` | 数据集目录里要有 `instruct` 文件，内容固定 `You are a helpful assistant.<\|endofprompt\|>` |

**conf 必须用模型自带的**：`/mnt/tts/models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml`，
不是仓库 examples 里的 `cosyvoice2.yaml`（那是 CosyVoice2 结构，与权重对不上）。

SFT 需要改的几处：
`padding.use_spk_embedding=True`、`lr=1e-5`、`scheduler=constantlr`、
`warmup_steps` 按数据量调小、`save_per_step` 调小（CV 评估跟着 save 走）。

## 3. 🔴 硬件约束：3090 不许满负荷

2026-10-07 训练到第 30 分钟，3090 掉卡：

```
AER: Uncorrectable (Fatal), (Receiver ID), [ 5] SDES   <- Surprise Down
Xid 79: GPU has fallen off the bus   -> Node Reboot Required
```

根因是**供电/接触**，不是 PCIe 链路裕量（详见 `docs/r22-3090掉卡根因与训练降载-2026-10-07.md`）。

训练前必做：
- ✅ `nvidia-smi -i 1 -pl 250`（已做成开机自启 `deploy/gpu-powercap.service`，
  但手动跑训练时**再确认一次**）
- ❌ 训练时**不要**同时跑 cosy 在线推理（叠加电流）
- ✅ `save_per_step` 调小，掉卡不至于白跑
- ✅ 日志/监控写 `/mnt/tts/diag/`，开 `tools/pcie_watch.sh` 留证据
