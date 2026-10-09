# Qwen3-TTS LoRA 微调方案（突破 96.5% 天花板）

> 起草 2026-10-09。目标只有一个：**尽最大可能 1:1 还原案例**。
> 本文只做方案与预算，**未动服务器任何权重**。跑之前需要一次显存冒烟。

---

## 1. 为什么必须走这一步

后处理能修的已经全部修完。五例实测（对标案例口径）均值 **96.07**，剩余 3.93 分：

| 项目 | 状态 | 说明 |
|---|---|---|
| lufs / peak / dur / B | ✅ 已归零 | 响度、峰值、时长、语速波动全部对齐 |
| **tone** | 🔴 8.9 / 14.6 / 7.6 / 10.4 / 6.4 | 音色频谱。18 段 EQ 已重拟合，但 EQ 是**全局静态**滤波，改不了"随内容变化的音色" |
| **E** | 🔴 8.5 / 2.6 / 6.5 / 3.6 / 11.3 | 句尾音高下沉。PSOLA 能改整体轮廓，改不了逐字自然度 |

**这不是"没调好"，是"调不到"**：后处理是事后补救，只能修表面（响度/峰值/停顿/宏观韵律），
修不了底子（模型生成的音色分布、逐字发音习惯）。要让底子对，只能让模型自己生成得像。

---

## 2. 技术方案

### 2.1 为什么是 LoRA 而不是全量 SFT

| | 全量 SFT | **LoRA（选这个）** |
|---|---|---|
| 克隆能力 | 🔴 会丢（见 2.3） | ✅ 保留（speaker_encoder 冻结） |
| 回滚 | 需备份 4.3G 权重 | ✅ 权重不动，只挂适配器，随时摘 |
| 过拟合风险 | 高 | ✅ 低（可训参数少） |
| 显存 | 24G 装不下 | ✅ 可 |

### 2.2 官方已留微调入口（关键前提，已核实）

`qwen_tts/core/models/modeling_qwen3_tts.py`（2299 行）里：

- `Qwen3TTSTalkerForConditionalGeneration.forward(..., labels=None, ...)`
  → 接受 labels，**官方已实现 next-token loss**
- `Qwen3TTSTalkerForConditionalGeneration.forward_sub_talker_finetune(codec_ids, talker_hidden_states)`
  → **函数名直接叫 finetune**，子 talker（16 组码本）的训练通路

所以**不需要自己写 loss**，只要按官方的数据格式喂进去即可。这是本方案最大的可行性支撑。

### 2.3 🔴 必须绕开的坑：不要改成 `custom_voice`

```python
# modeling_qwen3_tts.py:1830
if config.tts_model_type == "base":
    self.speaker_encoder = Qwen3TTSSpeakerEncoder(...)
else:
    self.speaker_encoder = None      # ← 换成 custom_voice 就没了
```

官方 SFT 教程会把 `tts_model_type` 改成 `custom_voice`，代价是：
说话人编码器被删、`tts_model_type` 写死、说话人固定到槽位 3000、只支持单说话人
→ **克隆能力彻底没了**。而本项目克隆是主力链路。

**对策：训练与保存时 `tts_model_type` 全程保持 `"base"`，speaker_encoder 冻结不训。**

### 2.4 训练拓扑

```
案例音频 ──speech_tokenizer.encode──▶ codec_ids (12Hz × 16 组码本)
案例文字 ──tokenizer──▶ input_ids
                    │
                    ▼
        talker (1.7B) ── LoRA r=16 挂在 attention q/v proj
        speaker_encoder ── 冻结（保克隆能力）
        code_predictor ── forward_sub_talker_finetune
                    │
                    ▼
              next-token loss
```

---

## 3. 数据盘点（2026-10-09 实测，`/mnt/tts/data/cases.json`）

| 说话人 | 性别 | 条数 | 时长(分) | 完整文字稿 | 来源 |
|---|---|---|---|---|---|
| SOLA | 女 | 20 | **166.2** | 20/20 ✅ | elevenlabs |
| yohan | 男 | 12 | **100.6** | 12/12 ✅ | elevenlabs |
| SOLA | 男 | 1 | 8.1 | 1/1 | elevenlabs |
| 案例-男声 | 男 | 1 | 8.1 | 0 | elevenlabs |
| 案例-女声 | 女 | 1 | 7.7 | 0 | elevenlabs |
| elara | 女 | 1 | 2.9 | 0 | elevenlabs |
| **合计** | | **36** | **293.5** | | |

**可直接用的对齐数据 = SOLA 166 分 + yohan 101 分 = 267 分钟（4.5 小时），13.8 万字。**

单说话人语音微调的常用门槛是 30 分钟~2 小时对齐数据；267 分钟**远超门槛**，不需要数据增广。

⚠️ 注意：案例本身是 ElevenLabs 生成的，不是真人录音。但我们的目标就是"1:1 还原案例"，
所以训练目标与验收目标**完全一致** —— 让 Qwen 学会这套音色与韵律风格，正是要的。

### 3.1 数据切分（防过拟合）

```
train:  SOLA 17 条 + yohan 10 条
dev:    SOLA  3 条 + yohan  2 条     ← 用于早停，不参与训练
验收:   五例标杆（标杆男/标杆女/대한/세미/리노）   ← 与 dev 不重叠
```

---

## 4. 训练配置（3090 24G）

| 项 | 值 | 理由 |
|---|---|---|
| 显卡 | **3090（UUID GPU-b67eee2c-…）** | 24G；3060 只有 12G 且要跑线上推理 |
| 精度 | bf16 | 3090 支持 |
| LoRA r | 16，alpha 32 | 起步保守；有效再试 r=32 |
| target modules | attention 的 q_proj / v_proj | 标准做法 |
| gradient checkpointing | 开 | 省激活显存 |
| batch / 累积 | 1 × 8 | 长音频单条就很大 |
| 学习率 | 1e-5 起 | Cosy SFT 的教训：甜点在 1 epoch 内 |
| **epoch** | **最多 1 个** | 🔴 epoch5 灾难过拟合（Cosy 实测） |
| checkpoint | 每 50 step 存一次 | 事后逐个体检挑最优 |

### 4.1 时间预算

| 阶段 | 预估 |
|---|---|
| codec 编码 267 分钟音频 | 1~2 小时（一次性，可缓存复用） |
| 1 epoch 训练 | 40~90 分钟 |
| 每 checkpoint 五例评分 | ~20 分钟/个 |
| **首轮出结论** | **约 3~4 小时** |

⚠️ Cosy 服务常驻 3090，训练时需要错峰（或临时停 cosy-tts）。

---

## 5. 验收口径（必须严格遵守，否则白跑）

1. **同一套 `tools/fidelity.py`**，五例标杆，**case 1 口径**（对标案例）
2. 判据：**均值 > 96.07 才算有效**（这是当前基线，等于没进步就不上线）
3. 同时验证**克隆能力没丢**：用训练外的参考音频（elara）生成，音色相似度不应下降
4. **五例方向必须一致**：只有五例全正才算系统性提升，一两条高是噪声（陷阱 #6）
5. 分数**必须带口径标注**（35 条 / 36 条 / case 1 / case 0），跨口径不可比

---

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| 灾难过拟合 | 只训 1 epoch 内；每 50 step 存点；dev loss 抬头就停 |
| 学到"案例的缺点"（如某条案例的口齿不清） | 只用 fb_rating 高的 / 剔除异常条 |
| 克隆能力退化 | speaker_encoder 冻结 + tts_model_type 保持 base + 训练外音色回归测试 |
| 显存不够 | 先跑 1 条冒烟；不够就降 r 到 8、加梯度累积、开 8bit 优化器 |
| 与线上抢卡 | 训练脚本显式 `CUDA_VISIBLE_DEVICES=<3090 UUID>`；训练期间停 cosy-tts |

---

## 7. 执行前检查清单

- [ ] 装 `peft`（当前 venv **没有**，LoRA 必需）
- [ ] 3090 上停 cosy-tts 服务（或确认空闲）
- [ ] 备份 `Qwen3-TTS-12Hz-1.7B-Base`（4.3G；LoRA 其实不改它，但留个底）
- [ ] 冒烟：1 条样本跑通 forward + backward，记录峰值显存
- [ ] 确认 `/mnt/tts` 有 ≥20G 余量放 checkpoint

---

## 8. 第一步该做什么（最小动作）

**不要一上来就训 267 分钟。** 先做这件事：

```
1. 装 peft
2. 取 SOLA 的 1 条案例，编成 codec_ids
3. 跑 1 步 forward + backward，打印 loss 与峰值显存
4. loss 正常下降、显存 < 22G → 方案成立，再铺全量
```

这一步 30 分钟内能出结果，失败也只有 30 分钟代价。
