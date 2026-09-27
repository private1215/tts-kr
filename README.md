# Qwen3-TTS 韩语语音生成平台

内部 TTS 语音生成平台，主营韩语语音合成。FastAPI + Qwen3-TTS-12Hz-1.7B，
单卡 RTX 3060 12GB 推理，团队 5 人日产 100~200 条韩语语音片段。

## 目录结构

```
.
├── app/                      # 服务端运行代码（对应 /mnt/tts/projects/qwen3-tts/app）
│   ├── server.py             # FastAPI 路由 / 鉴权 / 任务队列 / 用户管理
│   ├── tts_engine.py         # TTSEngine：模型管理、自适应批量、OOM 兜底
│   ├── pipeline.py           # 韩语归一化、断句分段、静音拼接、语速增益
│   ├── ui/index.html         # 单文件前端（无构建流程）
│   ├── bench.py  opt.py      # 性能基准 / 参数寻优脚本
│   ├── sweep.py  realism.py  # 参数扫描 / 自然度评估脚本
│   └── analyze.py            # 任务统计分析
├── deploy/
│   └── qwen-tts.service      # systemd 单元（对应 /etc/systemd/system/）
├── baseline/                 # 2026-09-27 修复前的原版，仅供 diff 对照
│                             #   server.py / tts_engine.py / qwen-tts.service / index.html
├── docs/
│   └── bench_result.json     # 本机 benchmark 结果，参数选型的实测依据
├── requirements.txt          # venv pip freeze（110 个包）
└── CHANGELOG.md
```

> 注意：服务端 `app/` 目录下还存在一份**过期的** `qwen-tts.service`（2026-09-25 版本），
> 真正的生效单元在 `/etc/systemd/system/qwen-tts.service`，已同步到本仓库 `deploy/`。
> 仓库刻意未收录 app/ 下那份，避免混淆。

## 运行环境

| 项 | 值 |
|---|---|
| 服务器 | Ubuntu 24.04.3 LTS，内网 192.168.1.100 / Tailscale 100.96.63.23 |
| CPU / 内存 | Xeon E5-2673 v3 12C/24T @2.4GHz / 62G |
| GPU | RTX 3060 12GB（驱动 595.91，CUDA 13.2，可用 11.63 GiB） |
| 数据盘 | `/mnt/tts` 1.8T NVMe（fstab `nofail`，已在单元中声明挂载依赖） |
| Python | `/mnt/tts/projects/qwen3-tts/venv`（torch 2.14.0+cu130） |
| 服务 | systemd `qwen-tts.service`，root 运行，Restart=always |
| 端口 | FastAPI 直接监听 0.0.0.0:8000（**无 Nginx，无 HTTPS**） |

## 模型

`/mnt/tts/models`，合计约 9.25G，不入仓库：

- `Qwen3-TTS-12Hz-1.7B-CustomVoice` 4.3G —— 9 个预设音色（sohee / eric / aiden / dylan / ryan / serena / vivian / ono_anna / uncle_fu）
- `Qwen3-TTS-12Hz-1.7B-Base` 4.3G —— 声音克隆（clone 模式）
- `Qwen3-TTS-Tokenizer-12Hz` 651M

加载方式为 bfloat16 + sdpa（flash-attn 未安装）。同一时刻仅常驻一个变体，按需热切换。

## 核心机制：自适应批量

2026-09-27 起，分段字数与批量大小**由代码决定，不依赖调用方传参**。

崩溃阈值由历史故障实测标定：单批 token 预算约 **7300** 必 OOM（`max_chars=110` × 10 段 = 7560）。
安全配置取该阈值的 50%：

```
单段 token 数 = min(2048, max_chars * 6.0 + 96)
净余量比值    = clamp((总显存 - 已分配) / 7.4 GiB, 0.30, 1.0)
单批预算      = 3650 × 净余量比值
batch_size    = clamp(预算 // 单段 token 数, 1, 8)
max_chars     = clamp(请求值, 20, 60)
```

RTX 3060 干净显存下收敛到 **max_chars=60 / batch_size=8 / 单批 3648 tokens**。
8 是吞吐最优点（`bench_result.json`：batch_8 RTF 0.49，batch_4 反而退化到 1.07）。

三层保护：
1. `plan_generation()` 把任何不安全入参收敛进安全区间（含 UI 传来的 `batch_size=1000`）
2. 批间 `torch.cuda.empty_cache()`，阻断 reserved 跨批累积
3. `_gen_batch_safe()` 捕获 OOM，批量减半重试最多 3 次

**这句很重要**：不要为了"更保守"去加大 `batch_size` 或减少分段数——
二者都会抬高单批 token 预算，方向恰好相反。

## 同步到服务端

```bash
scp app/server.py      zheng@100.96.63.23:/mnt/tts/projects/qwen3-tts/app/
scp app/tts_engine.py  zheng@100.96.63.23:/mnt/tts/projects/qwen3-tts/app/
scp deploy/qwen-tts.service zheng@100.96.63.23:/tmp/
ssh zheng@100.96.63.23
sudo cp /tmp/qwen-tts.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl restart qwen-tts
```

### 换行符警告

`server.py` 是 **CRLF**，`tts_engine.py` 是 LF。切勿用编辑器整体转换格式——
改动 `server.py` 时若丢失 `\r`，文件在服务端仍可运行，但会与已有版本产生整文件 diff，
后续排查会非常困难。建议用字节级补丁脚本而非交互式编辑。

```bash
# 校验
python3 -c "d=open('app/server.py','rb').read(); \
print('CRLF', d.count(b'\r\n'), 'bare LF', d.count(b'\n')-d.count(b'\r\n'))"
```

## 敏感数据

以下内容**不得入库**，已在 `.gitignore` 屏蔽：

- `data/users.json` —— 含 PBKDF2-SHA256 密码哈希与 salt
- `data/jobs.json` —— 含全部用户文案原文
- `output/` —— 229M 音频产物
- `models/`、`venv/` —— 15.4G

请将仓库设为**私有**。

## 已知待办

- [ ] FastAPI 裸奔 8000 端口，无 Nginx / HTTPS / 并发限制
- [ ] 服务以 root 运行
- [ ] 数据用 JSON 文件持久化，未来建议迁到数据库
- [ ] `service.log` 无轮转策略，前端轮询日志占比高，需按月切割
- [ ] `voices/voices.json` 至今为空 `{}`，尚未沉淀参考音色
- [ ] 前端仍保留「分段字数 / 批量大小」输入框，可考虑移除（后端已兜底，不影响安全）
