# tts-kr 速查索引（2026-10-19 晚）

> 细节版见 `.workbuddy/memory/archive-detail.md`（GS 硬坑、CER、已证伪全表、清理链实测、
> click 归因、反馈台出题铁律、v3.1~v3.3 明细）。本文件只留**高频硬规则**。
> 改链路/配置/引擎/权重前先读 `项目规范.md`；与本索引冲突以规范为准。

## 🔴🔴🔴 两条版本轴（用户纠正，别再混）
| 轴 | 格式 | 含义 | 真相源 | 写入口 |
|---|---|---|---|---|
| **项目版本** | `v<主>.<次>` | 链路状态 | `VERSION` 的 `version =` | `release_version.py` |
| **反馈批次** | `z<N>` | 第几批发出去的问题 | `VERSION` 的 `round =` | 同上（`--round`） |
- 题 id = `v<版本>-<z批次>-<维度>`；`ver` 字段**只放项目版本**。
- `/api/version` 三个布尔必须全 true：`version_consistent` / `round_consistent` / `consistent`。
- 🔴 必须 `--set/--bump ... --round ... --items <题库>` **一次给全**（分两次会把 minor 顶两档 v3.3→v3.4）。
- 🔴 事故（已修）：`rewrite_items()` 只换版本前缀不换 `z` 段 → 题库成 `v3.3-z32-*`，`consistent` 仍 true。
  修法 `patch_verround53.py`（脚本加 `new_round`/`RE_ROUND`，`app/relver.py` 加 `round_ok`），
  `patch_audit53_round.py` 给 audit 加 6 条检查项。

## 🔴🔴🔴 EL 标杆口径（推翻多条历史结论）
铁律：**EL 基线一律取 `cases/<id>/merged.wav`**。
- ❌ 女声曾用 `diag/qwen_fullf_EL_female.wav`（非标准）→ 音色记成 0.8172，实际 **0.9204**。
- 🔴🔴 **EL 自身运行间方差极大**（同文本同嗓音两次录音）：click 0.0046 vs 0.0018（**2.56×**）、
  杂音 7.3 vs 1.3（**5.6×**）→ 比值 **<2.6×（click）/ <5.6×（杂音）都在 EL 噪声里，不算真差距**；
  真差距要**同时**低于/高于两个 EL 参考。

## 目标与验收
替代 ElevenLabs（**EL**）：同文案我们 vs EL，员工听不出差别。员工零调参，**验收只看 L3 长篇（≥3~5 分钟）**。
- 🔴 物理差异 ≠ 可闻差异（z44：13dB 改动，9 票仅 1 票指向修后）→ 定根因前先出裁决题。
- 🔴 反向错误（z20）：GS 女声客观全面优于 EL，员工一致判「明显是机器」=0 分
  → click/音色/杂音**只是体检项不是验收项**；换引擎只能由盲听拍板。
- 🔴🔴 **片段口径与全篇口径可以相反** → 只用 L3 下结论。

## 员工两个症状（L3 全篇量化）
| 症状 | 我们 | EL | 倍数 |
|---|---|---|---|
| (a) 句首紧贴碎片（「文案里没的字也发音」） | 6 / 0.64 每分 | 2 / 0.22 | 3× |
| (b) 句末粘连嗯（「吐字后嗯的尾音」） | **56 / 5.99 每分** | 36 / 3.89 | **1.55×** |
🔴 **模版传染已证伪**：omni 参考音频句首 0、句末 ≤1 → 干净模版照样 56 个 = **模型固有**，只能后处理清。
⚠️ 清理目标 = **对齐 EL 密度 3.89/分，严禁清零**。

## 测量口径（不统一就作废）
| 指标 | 铁律 |
|---|---|
| click | **先归一到峰值 0.70** 再判 `|Δ|>=0.30`；🔴 重采样 16k；🔴🔴 **必须测压完 mp3 后的值** |
| 音色 | campplus 16k（cosy venv），同人阈值 0.75；EL-EL 中位 **0.9854(女)/0.9850(男)**；`measure53` 不含 |
| cps | 字数时长取自**同一条 job 记录**，用**含空格**字符数；EL 锚 女 7.45 / 男 8.37 |
| ASR | 🔴🔴 >3min **必须分段**（`asrseg53`，30s 窗）；整篇喂 whisper-small 会**循环复读** |
| 采样率 | 🔴🔴🔴 **成品绝不能 16 kHz**：`fix_chain` 输出就是 16000 Hz，清理全在 16k 跑，最后 `aresample=44100` 压 mp3。libmp3lame：44.1k→mp3 音色 **0.9999**（透明），16k→mp3 只 **0.79**（16k 成品 vs EL **0.6903 ❌**）。旧工具写死 SR=16000 **只能中间处理**；保采样率版 = `clickfix53b`/`ltasmatch53b`/`tailclean53b2`/`53c2`/`eqfix53b` |
- σ 一律 `MAD×1.4826`；必须全篇/中段测。找「多出来的音」别用绝对 dB 阈值（能量分布几乎一致，
  差异在**时序结构**）。

## 现役链路（v3.3）
qwen :8000（**后处理只在这里**）/ cosy :8100 / gs :8200 / omni :8300 / vox :8400。
- 🔴🔴 **男女声都走 omnivoice**，路由看 `voices.json` 每音色 `engine`。**不按引擎名判性别**：
  `_gender_of(job)` 取性别，`_engine_name` 取引擎。
- 🔴🔴 **omni 走极简后处理**（`--no-pitch --no-clean --no-adaptive-hf --eq anull`），守卫必须
  `if _engine_name == "omnivoice":`（曾只剩男声生效，女声白跑一天）。**GS 不走任何后处理**。
- 钩子都是 `run_job()` 内嵌套函数：`_mideq53b_wrap`（窄带 EQ）→ `_clickfix53b_wrap`（click 插值），顺序不能反。
- 🔴 `/api/jobs` 只同源、内部按引擎分流；**必须用 `tools/eval5` 的 login/submit_job/wait_job**。

## ✅ v3.1→v3.3（均已 L3 全篇验收）
| 版本 | 改动 | 结果 |
|---|---|---|
| v3.1 | 男声 clickfix53b（th=0.30） | click 1.21×→**0.05×**、音色零损失、cps→8.27 |
| v3.2 | 男声窄带 EQ 1k:0/1.8k:+2/3k:0 | 中频 0.86→**1.01**、质心 1.02、杂音 0.73、click 0.28×、音色 0.9329 |
| v3.3 | 女声接 EQ+clickfix（男0.30/女0.26） | 中频 0.88→**1.01**、click 1.24×→**0.00×**、质心 0.99、杂音 1.12、音色 0.9204→**0.9227**（六项全改善） |
- 决策原则：**只动「我们比 EL 差」的指标**；已比 EL 好的不动。

## 当前基线（v3.3）
- 男声 click **0.28×** / 质心 1.02 / 中频 1.01 / 杂音 0.73 / cps 8.27 / 音色 **0.9329**。
- 女声（合规标杆 00b6a45ad6aa）click **0.00×** / 质心 0.99 / 中频 1.01 / 杂音 1.12 / cps 1.00 / 音色 **0.9227**。
→ 客观指标已全部对齐 EL。音色 0.92~0.94 距 EL-EL 0.985 差 ~0.05，**男女差距不大**
  → **GS 微调紧迫性大幅下降**（旧「女声 0.8172 最大短板」是错误标杆，已作废）。

## 常用操作坑（最高频）
- 🔴🔴 **补丁自检必须 `compile()`，不能用 `ast.parse()`**（不走符号表，抓不到 `await` 作用域错误）。
  给 `run_job()` 里的嵌套函数插 helper，缩进必须**同为 4 空格**，否则截断外层 async → 服务启动即崩。
- 🔴🔴 **插 helper 后必须跑 L1 实测，不能只信 audit53**（它只查"字符串在不在"，查不到"没被执行"）。
  两招它都全绿：① `return _cf(*_eq(...))` 解包二元组少收参数 → TypeError → 回退 raw；
  ② `if len(_y) != len(audio): return audio` —— **firequalizer 输出比输入长 filter_size-1 样本**
  （16k 长 320 = 20ms）→ 每次退回。修：helper **只返回 audio**；长度差**居中裁剪**，差 >0.5s 才真退回。
- 🔴🔴 **改完 `voices.json` 必须重启 qwen-tts**（任务结束时把内存音色表写回磁盘，冲掉新加音色）。
- 🔴🔴 **app/ 下扁平导入**：`from app import xxx` 启动即连崩 7 次。
- 🔴🔴 **服务日志在 `service.log`，不在 journalctl**；`sudo -n journalctl` 需密码时**静默返回空、不报错**。
- 🔴🔴 **产物目录会加 `-N` 后缀**：`find_out` 取 mtime 最新；`glob(title+"*")` 会抓旧目录。
- 🔴🔴 **faster_whisper 必须 `local_files_only=True`**（无外网 → 静默挂死 4 分钟）；**不能传文件路径**
  （av 不兼容）→ ffmpeg 降 16k，soundfile 读 ndarray 再 `transcribe`。
- 🔴 `voices/sources/` 是 root 属主 → 切片放 `voices/<name>.wav`。`voices.json` 是**扁平字典**（键=音色名），
  写入前照抄完整字段：path/label/ref_text/sr/dur/engine/gender/bench_case/origin/src_case/src_speaker/src_dur/clip/cps_lock/cps。
- 🔴 反馈台 `items.json` 必须 `{"items":[...]}`（裸 list 打不开）；前端选项**单选**；
  **每个有 file 的 option 都渲染播放器** → 只让第一个带 file。
- 🔴 `pkill -f "[x]xx"`；缓存名用**全路径 hash**（男女标杆都叫 merged.wav）。
- 🔴🔴 **永远不整目录 rsync**（本地是旧快照），只能按文件 scp；韩文目录名 ssh 双引号里常失败 → 先 cp 到 ASCII。
- 🔴 ssh 后台任务：`cd X && setsid nohup <绝对路径 python> ... & disown`（`python` 不存在）。

## 🖥️ 服务器 / 队列
- **GPU0 = 3060(12G) qwen**；GPU1 = 3090(24G) cosy/gs/vox/omni（两张卡都塞不下 qwen+omni）。
- 🔴🔴 **队列串行**：一条卡住 → 后面全「排队中」。判真假挂：`systemctl show qwen-tts -p NRestarts`（0=没崩只是慢）。
- 🔴 耗时放大链：OOM → 反复初始化 → 语速崩 → 重试 ×3 + CER ×3 ≈ **9 倍**（已关两个 retry 环境变量）。
  ✅ 修复后基准 **4142 字 / 515.9s / RTF 0.493**。`expandable_segments` 会报假性 OOM，用
  `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8`。

## 操作速查
`ssh yuyin`（zheng，可 sudo；密码见用户级记忆）。代码 `/mnt/tts/projects/qwen3-tts`；
venv `…/qwen3-tts/venv/bin/python`；cosy venv `/mnt/tts/projects/cosy/venv`。时区 UTC（+8=北京）；
判活用根路径 200（`/health` 不存在，404 正常）。
- 🔴 一次 ssh 里多个 sudo 只有第一个吃到密码 → `echo "$PW" | sudo -S -p '' bash -c 'a; b; c'`。
- zheng 可写 /mnt/tts/{data,projects,fb,diag}；**root 属主写不进**：`data/tone-ab`、`/etc/systemd/system/*.service.d/`
  （常不存在先 `mkdir -p`；改后 daemon-reload + restart）。git push 间歇 502，重试 2~4 次。

## 工具索引
**现役**：`clickfix53b`(click 插值) / `eqfix53b`(窄带 EQ) / `measure53`(全量指标) / `spkcmp53`(音色 campplus) /
`asrseg53`(>3min 分段 ASR) / `alignchk48`(文本对齐) / `audit53`(服务端体检，含两条版本轴) /
`release_version`(版本号唯一写入口) / `fbcheck53`(素材体检) / `tailclean53b2`+`53c2`(清理，**未上生产**)。
**deploy**：`patch_clickfix53b`(v3.1) / `patch_mideq53b`(v3.2) / `patch_female53`+`patch_mideqlog53`(v3.3) /
`patch_verround53` / `patch_audit53_round`。
**diag**：`pipe_1019h`(clickfix 音色 vs 阈值七变体) / `pipe_f53`(女声 EQ+cf 同源) / `l3f53_verify`(女声 L3 验收) /
`l1_female53`(L1 日志取证) / `gen_z33f`(补女声题+响度统一)。
