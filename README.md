# Qwen3-TTS 韩语语音生成平台

内部 TTS 语音生成平台，主营韩语语音合成。FastAPI + Qwen3-TTS-12Hz-1.7B，
单卡 RTX 3060 12GB 推理，**7 人团队日产 80 条**韩语语音片段（8 小时工作制）。

> 本文档最后核对：2026-09-28（自动裁剪与音色列表上线后）。
> 改动代码后请同步更新本文档与 `CHANGELOG.md`，避免后人按过时信息排查。

## 文档索引

| 文档 | 内容 |
|---|---|
| 本 README | 环境、部署、核心机制速览、同步方式、待办 |
| `CHANGELOG.md` | 按次记录每次改动的背景 / 改动 / **实测验收数据** |
| `docs/声音克隆机制与排障.md` | 克隆链路完整机制、参数表、已踩过的坑与判读方法 |
| `docs/产能与排障分析报告-2026-09-27.md` | 7 人团队产能模型、排队论、错峰规范 |
| `docs/bench_result.json` | batch 选型 benchmark 原始数据 |

## 目录结构

```
.
├── app/                      # 服务端运行代码（对应 /mnt/tts/projects/qwen3-tts/app）
│   ├── server.py             # FastAPI 路由 / 鉴权 / 任务队列 / 用户管理 / 音色录入
│   ├── tts_engine.py         # TTSEngine：模型管理、自适应批量、克隆显存感知、OOM 兜底
│   ├── voice_trim.py         # 参考音频自动裁剪 + 文字稿按有声时长对齐（2026-09-28 新增）
│   ├── pipeline.py           # 韩语归一化、断句分段、静音拼接、语速增益
│   ├── ui/index.html         # 单文件前端（无构建流程，含「已录入音色」面板）
│   ├── bench.py  opt.py      # 性能基准 / 参数寻优脚本
│   ├── sweep.py  realism.py  # 参数扫描 / 自然度评估脚本
│   └── analyze.py            # 任务统计分析
├── deploy/
│   └── qwen-tts.service      # systemd 单元（对应 /etc/systemd/system/）
├── baseline/                 # 2026-09-27 修复前的原版，仅供 diff 对照
│                             #   server.py / tts_engine.py / qwen-tts.service / index.html
├── docs/
│   ├── bench_result.json     # 本机 benchmark 结果，参数选型的实测依据
│   ├── 声音克隆机制与排障.md  # 克隆链路的机制、参数与排障手册
│   └── 产能与排障分析报告-2026-09-27.md
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

### 克隆模式额外一道：前缀 KV 显存感知

克隆时每个样本都要背一份完整 prompt 前缀（`voice_clone_prompt=[prompt] * n`），
**前缀长度按 batch 倍数放大 KV cache**。所以除了 token 预算，还要按显存再算一次上限：

```
KV 单价 = 2(K,V) × 28 层 × 8 KV头 × 128 维 × 2B = 0.109375 MiB/token
前缀 KV 合计 = 单价 × prompt_tokens × batch_size
batch 上限  = min(token 预算上限, 显存余量 × 0.40 ÷ 单样本前缀 KV)
```

用**显存**而不是 token 预算折算：前缀只做一次 prefill，不产生 decode 步和 vocoder 开销。
按 token 折算会把 25 秒参考音频（完全安全）也从 batch 8 压到 4，白白损失吞吐。

实测（free 7.45 GiB）：15s → batch 8；25.74s → batch 8；30s（上限）→ batch 8；
329.33s 旧音频 → batch 5（旧版固定 8，必 OOM）。

## 声音克隆：员工怎么用

**员工只需要做两件事**：上传整段录音 + 粘贴这段录音的完整文字稿。
不需要剪音频、不需要数秒数、不需要对时间轴。
超过 30 秒的音频由服务端自动截取约 15 秒，并把文字稿按**累计有声时长**对齐截出对应片段。

| 参数 | 值 | 位置 |
|---|---|---|
| 最短 / 建议 / 自动截取阈值 | 3s / 15s / >30s 触发 | `server.py` `REF_AUDIO_*` |
| 硬上限（直接拒绝） | 900s（15 分钟） | `REF_AUDIO_HARD_MAX_SEC` |
| 截取窗口范围 | 8~20 秒 | `voice_trim.py` `MIN/MAX_CLIP_SEC` |
| 语速告警区间 | 2.5~10 字/秒 | `server.py` 录入时校验 |

产物：成品片段 `voices/<名>.wav`（≤30s，直接用于克隆）+ 原音频留档
`voices/sources/<名>__full.wav`（换策略时可重算，不用员工重传）。

**两个硬约束**（详见 `docs/声音克隆机制与排障.md`）：

1. `ref_text` **必填**。ICL 模式下留空会直接抛
   `ref_text is required when x_vector_only_mode=False`（模型源码 `qwen3_tts_model.py:437`）。
2. 文字稿必须**逐字对应**音频内容。比例法依赖这个前提；文字明显多于/少于音频
   会让模型从 prompt 里学到错误的语速，表现为拖长音、重复、生成耗时暴涨。
   录入时会给黄色告警，但代码纠正不了内容本身。

## 同步到服务端

```bash
scp app/server.py      zheng@100.96.63.23:/mnt/tts/projects/qwen3-tts/app/
scp app/tts_engine.py  zheng@100.96.63.23:/mnt/tts/projects/qwen3-tts/app/
scp app/voice_trim.py  zheng@100.96.63.23:/mnt/tts/projects/qwen3-tts/app/
scp app/ui/index.html  zheng@100.96.63.23:/mnt/tts/projects/qwen3-tts/app/ui/
scp deploy/qwen-tts.service zheng@100.96.63.23:/tmp/
ssh zheng@100.96.63.23
sudo cp /tmp/qwen-tts.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl restart qwen-tts
```

生效规则（**改错文件会白等**）：

| 文件 | 是否需要重启服务 |
|---|---|
| `app/*.py` | **需要** `systemctl restart qwen-tts`（模型要重新加载，约 45 秒） |
| `app/ui/index.html` | **不需要** —— 前端每次请求读盘，刷新浏览器即可 |
| `deploy/qwen-tts.service` | 需要 `daemon-reload` + `restart` |

### 重启不会弄丢任务（2026-09-28 起）

旧行为：重启把 `running / queued` 一律标成 error「服务重启，任务中断」，
几千字长文案白等，只能手动重试。

现在改为**自动续跑**：启动时把被中断的任务重新入队，按提交时间排序接着跑。
两个保护：

- `resumed` 计数上限 `RESUME_MAX_TIMES=3`，防止"反复重启 → 反复重跑"打满队列
- 原文丢失（`text` 为空）的任务无法续跑，判死并写明原因

任务列表接口会返回 `resumed` 字段（>0 表示这条曾被中断后又接着跑）。

即便如此，**重启仍建议在任务间隙做**：续跑是整任务重跑，已生成的部分不会复用，
一次重启仍会浪费几分钟算力。重启前先看一眼有没有跑着的任务：

```bash
curl -s -u 方式登录后 GET /api/jobs?scope=all | grep -c '"status":"running"'
```

### 换行符警告

`server.py` 与 `ui/index.html` 是 **CRLF**，其余 `.py` 是 LF。切勿用编辑器整体转换格式——
改动 CRLF 文件时若丢失 `\r`，文件在服务端仍可运行，但会与已有版本产生整文件 diff，
后续排查会非常困难。建议用字节级补丁脚本而非交互式编辑。

当前基线（2026-09-28）：`server.py` CRLF 1058、`ui/index.html` CRLF 1572、
`tts_engine.py` / `voice_trim.py` / `pipeline.py` 裸 LF 0。

```bash
# 校验
python3 -c "d=open('app/server.py','rb').read(); \
print('CRLF', d.count(b'\r\n'), 'bare LF', d.count(b'\n')-d.count(b'\r\n'))"
```

### 推送 GitHub

本机环境变量里的 `HTTPS_PROXY` 常常指向一个**没有进程监听**的端口（且端口每次开机都变），
git 走代理必然 502。剥掉代理后直连即可：

```bash
env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy \
    -u ALL_PROXY -u all_proxy -u NO_PROXY -u no_proxy \
    git -c http.proxy= -c https.proxy= push origin main
```

校验远端要用 `git ls-remote origin main`，**不要用 GitHub REST API** ——
私有仓不带 token 会返回 404，看起来像"没推上去"，其实是鉴权问题。

## 敏感数据

以下内容**不得入库**，已在 `.gitignore` 屏蔽：

- `data/users.json` —— 含 PBKDF2-SHA256 密码哈希与 salt
- `data/jobs.json` —— 含全部用户文案原文
- `output/` —— 229M 音频产物
- `models/`、`venv/` —— 15.4G

请将仓库设为**私有**。

## 已知待办

### 安全 / 运维

- [ ] FastAPI 裸奔 8000 端口，无 Nginx / HTTPS / 并发限制
- [ ] 服务以 root 运行
- [ ] 数据用 JSON 文件持久化，未来建议迁到数据库
- [ ] `service.log` 无轮转策略，前端轮询日志占比高，需按月切割
- [x] ~~出厂管理员口令硬编码~~ **已修**（2026-09-28）：改为环境变量
      `TTS_ADMIN_PASSWORD`，未设置则随机生成 12 位并打 WARNING 日志
- [x] ~~服务端 `app/` 备份堆积~~ **已清**（2026-09-28）：7 个 `.bak*` 已删，
      回滚请用本地 git 历史与 `baseline/`

### 功能

- [ ] 参考音频无 ASR 校验：文字稿是否真的对应音频，只能靠录入时的语速告警（2.5~10 字/秒）
      间接判断。若后续发现克隆质量不稳，可考虑接 ASR 做强制对齐
- [ ] 前端「分段字数 / 批量大小」输入框已收归管理员可见（`advBox`），
      但 `POST /api/jobs` 仍不校验 role——员工绕过 UI 调 API 仍可传参；
      后端 `plan_generation()` 会收敛到 60/8，不会 OOM，只是没彻底封堵
- [x] ~~删除音色留下孤儿文件~~ **已修**（2026-09-28）：`DELETE /api/voices/{name}`
      现在连 `sources/` 里的原音频一起删，并校验路径必须落在 `VOICE_DIR` 内
- [x] ~~重启导致任务判死~~ **已修**（2026-09-28）：改为自动续跑，见上文

### 数据侧遗留（需人工确认）

- `voices/sources/요한 쿡_25s.wav`（2.2M）：手工裁剪产物，`voices.json` 无引用，
  但**原 293 秒音频已随音色删除而丢失，这是唯一副本**。已归档到 `sources/` 而非删除——
  确认不需要后可直接 `rm`。
