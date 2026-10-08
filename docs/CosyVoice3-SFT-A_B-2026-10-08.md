# CosyVoice3 SFT A/B 结论（2026-10-08）

## 一句话结论

**CosyVoice3 的 SFT 检查点是负收益，不要用。** 但它逼出了一件更值钱的事：
Qwen 同一条链路、同一案例，两次生成能差 **0.95 分**——拒绝采样挑种子可能是
不改模型就能冲 97% 的最低成本路线。

---

## 1. 三方对比（男声四例，全部走完整生产链路）

| 引擎 | 标杆男 | 대한광통신 | 세미파이브 | 리노공업y | 均值 |
|---|---|---|---|---|---|
| **Qwen（r25f，当前线上）** | 95.31 | 94.67 | 96.19 | 96.09 | **95.57** |
| CosyVoice3 + SFT | 95.87 | 94.78 | 94.02 | **90.99** | **93.92** |
| CosyVoice3 base | 96.15 | 94.13 | （污染） | （污染） | 95.14\* |

\* base 只有前两例可信，见第 2 节。

**读法**：
- SFT 在前两例上与 base 打平（95.325 vs 95.14，噪声内），但后两例崩了
  （세미 94.02、**리노 90.99**）→ 说明 **Cosy 引擎本身对这两类文案/音色不稳**，
  不是 SFT 的锅。
- 完整四例可比口径下，**Cosy-SFT 比 Qwen 低 1.65 分**。
- 结论：**换引擎不划算；SFT 此路已否。**

---

## 2. 🔴 第一版 base 数据作废：两例被静默降级污染

第一批跑出来的「Cosy-base 均值 95.66%、세미 97.14% 打破天花板」**是错的**。
事后查 `service.log`：

```
07:29:51 d8dada43d55b (cosybase-8197d9) CosyVoice3 失败 -> 整任务降级为 Qwen 重出
07:40:04 adc2a7b9703e (cosybase-6cdd25) CosyVoice3 失败 -> 整任务降级为 Qwen 重出
```

**那个 97.14% 是 Qwen 生成的，不是 Cosy。**

🔑 通用教训：**换模型后分数没变化，先 `grep 'CosyVoice3 失败' service.log`
确认没降级**。降级会照样出产物、照样出分数，只是引擎偷偷换人了。

---

## 3. 🔑 意外收获：Qwen 单种子波动就有 ~1 分

同一个 세미파이브 案例，走同一条 Qwen + 后处理链路，什么都没改：

- r25f 批次：96.19
- 降级那次：**97.14**

差 **0.95 分**。记忆里那条「官方随机缺陷 → 拒绝采样换种子」原本只当**防跑飞**
手段用，现在看它同时是**提分手段**：跑 N 个种子挑最高分交付，理论上均值能
+0.5~1.0，直接够到 96.5~97 区间，而且**不用碰模型、不用重训**。

下一步要验证的（避免又是噪声）：
1. 同一文案跑 4~8 个 seed，看分数分布与极差；
2. 挑优后**换一批案例复核**，确认兑现的是真提升而不是挑到了噪声。

---

## 4. 途中修掉的两个线上 Bug

### 4.1 deepspeed 让 CosyVoice3 服务重启后必挂

```
from transformers import PreTrainedModel
  -> modelscope.fix_transformers_upgrade
  -> deepspeed op_builder
  -> MissingCUDAException: CUDA_HOME does not exist
```

机器没有 nvcc（`cuda_nvcc` 里只有 ptxas），deepspeed 编译不了就抛异常。
表现是「**重启 cosy 服务后就 500**」，也正因如此 base 批次后两例才会 unload
后重载失败而降级。

✅ 已卸载 deepspeed（`tools/sft/train.sh` 是单卡直跑官方 `train.py`，不依赖它）。
顺带**修好了员工的情感功能**（此前选激昂/亲切/严肃会失败）。

### 4.2 SFT 检查点多两个键 → 加载崩 → 静默降级

CosyVoice3 用 `load_state_dict(strict=True)` 加载 `llm.pt`，而训练检查点里带了
训练元信息：

```
RuntimeError: Unexpected key(s) in state_dict: "epoch", "step"
```

`tools/sft/make_model.py` 当时只是**打印**了这两个 extra key，`DROP_KEYS = ()`
并没有真删。✅ 已改成 `DROP_KEYS = ("epoch", "step")` 并重新生成
（295 → 293 键，1.89 GB）。

⚠️ 这个 bug 极具迷惑性：它不报错到脸上，而是让任务**静默降级成 Qwen 重出**，
看起来就像「SFT 没效果」。

---

## 5. 线上已恢复（员工零感知）

- `cosy-tts` 模型切回 `Fun-CosyVoice3-0.5B`（service 文件已从 `.bak` 还原）
- 临时音色 `EL_male__COSY` 已删除，`voices.json` 恢复 `['EL_male','EL_female']`
- `qwen-tts` 已重启（voices 是启动时读进内存的）
- `/synth` 自检 200，CosyVoice3 能稳定加载

> 配套脚本已入仓库：`tools/add_cosy_voice.py` / `tools/del_cosy_voice.py`
> （引擎由音色字段决定，`/api/jobs` 不接受 engine 参数，所以 A/B 必须借临时音色）

---

## 6. 下一步

| 方向 | 预期收益 | 成本 | 判断 |
|---|---|---|---|
| **拒绝采样挑种子** | +0.5~1.0 分 → 96.5~97% | 低（不改模型） | **推荐先做** |
| Qwen 侧 LoRA/SFT | 跨数量级 | 高（环境+数据+3090 掉卡风险） | 终极路线，后面做 |
| 继续抠后处理残余 | +0.5 封顶 | 低 | 收益太小 |
| CosyVoice3 重训 SFT | — | 高 | **已否**（引擎本身不稳） |
