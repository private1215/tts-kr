# CosyVoice3 双引擎方案

> 目标：员工只选「**情感**」，引擎 / 指令 / 参数全部由后端自动分流。

## 1. 为什么需要第二个引擎

`generate_voice_clone`（Qwen3-TTS Base 克隆模式）**没有 `instruct` / 情感入参**：

- `generate_custom_voice` 有 `instruct`，但一旦传 `ref_audio` 就抛
  `ValueError: ... custom_voice does not support generate_voice_clone`
- CustomVoice 的 speaker 是**跟名字 token 绑定的查表向量**
  （`modeling_qwen3_tts.py:2087-2106`），克隆出来的音色塞不进去当 speaker

→ **Qwen 这条路上情感是调不出来的，只能在引擎层补。**

备选引擎对比：

| 模型 | 韩语 | 克隆 | 情感控制 | 结论 |
|---|---|---|---|---|
| Qwen3-TTS Base（现网） | ✅ | ✅ | ❌ 无 instruct | 保留（最准） |
| IndexTTS-2.5 | ❌ **无韩语** | ✅ | ✅ 8 维情感 | 出局 |
| **CosyVoice3** | ✅ | ✅ | ✅ `instruct` | **选用** |

## 2. 架构：两个独立服务

CosyVoice3 锁 `torch==2.3.1+cu121` / `numpy==1.26.4`，与现网 Qwen 的
`torch 2.14+cu130` **直接冲突，无法同进程**。所以：

```
前端（同一套 UI，只选情感）
        │
        ▼
tts-kr 主服务 :8000  ──── 情感=平稳 ──→ Qwen3-TTS（本地引擎）
        │
        └── HTTP ──→ CosyVoice3 服务 :8100（独立 venv / 独立进程）
                     情感=激昂 / 亲切 / 严肃
```

- 服务：`/mnt/tts/projects/cosy/cosy_server.py`，systemd `cosy-tts`
- 模型：`/mnt/tts/models/Fun-CosyVoice3-0.5B`（9.1G）
- 代码：`/mnt/tts/projects/cosy/repo`（含 Matcha-TTS submodule）

## 3. 显存：不用就不占

现网 Qwen 常驻约 4.25 GiB，CosyVoice3 再常驻约 3.2 GiB 会挤压 Qwen 的生成峰值
（实测峰值 6.97 GiB）。所以做了两层保护：

1. **懒加载** —— 服务启动不读模型，`/health` 返回 `model_not_loaded` / `vram_gib: 0.0`
2. **空闲自动卸载** —— 默认空闲 600s 卸载（`COSY_IDLE_UNLOAD_SEC`），
   也可 `POST /unload` 立即释放

## 4. 情感 → 引擎映射（`server.EMOTION_PRESETS`）

| 情感 | 引擎 | CosyVoice3 指令 |
|---|---|---|
| `standard` 平稳 | **qwen** | — （准确度最高） |
| `excited` 激昂 | cosyvoice3 | `Speak with an excited and passionate tone…` |
| `warm` 亲切 | cosyvoice3 | `Speak in a warm, calm and conversational tone…` |
| `serious` 严肃 | cosyvoice3 | `Speak in a steady, authoritative and serious tone…` |

### 质量闸门
CosyVoice3 出片后仍走 ASR 内容回检；**CER > `TTS_COSY_FALLBACK_CER`（默认 0.15）
就自动回退 Qwen 重出** —— 情绪不能以牺牲「念对字」为代价。

## 5. 踩过的坑

1. **`prompt_text` 必须含 `<|endofprompt|>`**。CosyVoice3 的 zero-shot
   强制这个格式，不带会在 flow 阶段报
   `Kernel size can't be greater than actual input size`（A 组首测就是这样挂的）。
2. **韩语必须 `text_frontend=False`**，否则文本前端会把韩语切坏。
3. **numpy 双版本残留**：`numpy-1.26.4.dist-info` 与 `numpy-2.5.3.dist-info` 并存
   导致 `numpy._core.multiarray failed to import`。修法：
   `pip uninstall numpy numba llvmlite` + 删残留目录 + 重装
   `numpy==1.26.4 numba==0.60.0 llvmlite==0.43.0`。
4. **setuptools ≥81 移除了 `pkg_resources`**，编译老包（pyworld/wetext）会挂。
   装之前先 `pip install "setuptools<81"`，或用 `PIP_CONSTRAINT`。
5. **服务器 GitHub 直连不稳定**（TLS 中断 / 132s 超时）。改为本机 clone
   （含 `--recursive` 拉 Matcha-TTS submodule）打包再传。
6. **分块长度与 Qwen 相反：CosyVoice3 要长块，越短越糟。** 详见 §5.1。

### 5.1 分块长度：CosyVoice3 的最优解与 Qwen 完全相反

最初给 CosyVoice3 设了 `max_chars=80`，理由是**照抄 Qwen 的经验**
（Qwen 用 60 字段，短段能让能量/语速保持一致）。**这是错的，而且代价很大。**

同文本 776 字韩语实测：

| `max_chars` | CER | RTF |
|---|---|---|
| 80 | 0.0892 | 1.276 |
| 160 | 0.0829 | 1.075 |
| 240 | 0.0850 | 1.229 |
| **320** | **0.0727** | **1.057** |

**全文 3699 字**（更能放大差异）：

| 分块方式 | CER | RTF | 备注 |
|---|---|---|---|
| 逐句（平均 ~78 字） | **0.186** | 1.187 | 早期据此判定 CosyVoice3「不可用于韩语」 |
| 320 字块 | **0.0671** | **1.049** | 同一份模型/参考音频/文本 |

一句话：**同一个模型，换个分块方式，CER 从 0.186 变成 0.0671（2.8 倍差距）。**

推测机制：CosyVoice3 的每块都要重新做一次 zero-shot prefill（注入参考音频+
`<|endofprompt|>` 指令）。块越短，重启次数越多，模型每十几秒就被拽回参照点一次，
表现为「念串 / 幻觉 / 漏字」——这些都是 CER 的主要来源。Qwen 恰好相反
（其内部 stanza 自回归在长段上会韵律漂移），所以两个引擎的分块策略必须分开。

→ 默认值已修正为 **320**（`cosy_server.py` 的 `Field` 默认值 +
`cosy_client.py` 的 `TTS_COSY_MAX_CHARS`）。

## 6. ⚠️ 客观指标与主观听感冲突（重要）

同文本 776 字女声实测（CosyVoice3 已修正为 320 字块）：

| | CER | RTF | 句内动态(半音) | 语速CV | 停顿熵 |
|---|---|---|---|---|---|
| **Qwen 平稳（现网）** | **0.0542** | 1.72 | **8.07** | 0.143 | 0.662 |
| **CosyVoice3 / 320 块（修正后）** | 0.0727 | **1.057** | — | **0.156** | **0.810** |
| CosyVoice3 / 80 块（旧的错误配置） | 0.1105 | 1.119 | 7.62 | — | — |
| ElevenLabs（参照） | — | — | 8.20 | 0.137 | 0.749 |

工程口径总结：
- **准确度** Qwen 仍略优（0.054 vs 0.073），但差距从「3.4 倍」缩到 **1.3 倍**
- **速度** CosyVoice3 更快（RTF 1.06 vs 1.72）
- **节奏多样性**（员工最在意的「语速一致」）CosyVoice3 反而更好：
  D 语速CV 0.156 > EL 0.137 > Qwen 0.143；F 停顿熵 0.810 > EL 0.749 > Qwen 0.662
- **情感可控** 只有 CosyVoice3 支持 instruct，这是它能解决「平淡」的唯一理由

**指标上 Qwen 反而更接近 ElevenLabs**，但员工主观反映「Qwen 平淡、有 AI 味」。
这说明「句内音高动态」**抓不到人味**（这也是文档里第二次记这条）。

→ **最终裁决权在人耳。** 情感选项就是给员工这个旋钮，
听完用脚投票；不要拿上表的指标当验收标准。

## 7. 已知遗留

- `engine_select.py`（音色级自动选型：新增音色时自动 A/B 打分绑定引擎）
  已写好但**尚未接入音色创建流程**，当前生效的是静态 `EMOTION_PRESETS` 映射。
- CosyVoice3 未启用 fp16 / TensorRT / vLLM，速度还有优化空间。
- 两个引擎同时常驻时峰值约 7.5 GiB，靠空闲卸载规避；若并发任务增多需复核。
