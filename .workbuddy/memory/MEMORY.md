# tts-kr 速查索引（2026-10-21，v3.5）

> 细节版见 `.workbuddy/memory/archive-detail.md`（GS 硬坑、CER、已证伪全表、清理链实测、
> click 归因、反馈台出题铁律、v3.1~v3.4 明细）。本文件只留**高频硬规则**。
> 改链路/配置/引擎/权重前先读 `项目规范.md`；与本索引冲突以规范为准。

## 目标与验收
替代 ElevenLabs（**EL**）：同文案我们 vs EL，员工听不出差别。员工零调参，**验收只看 L3 长篇（≥3~5 分钟）**。
- 🔴 物理差异 ≠ 可闻差异（z44：13dB 改动，9 票仅 1 票指向修后）→ 定根因前先出裁决题。
- 🔴 反向错误（z20）：GS 女声客观全面优于 EL，员工一致判「明显是机器」=0 分
  → click/音色/杂音**只是体检项不是验收项**；换引擎只能由盲听拍板。
- 🔴🔴 **片段口径与全篇口径可以相反** → 只用 L3 下结论。

## 🔴🔴 员工反馈核验法
员工报的点**先客观复核再动手**，别直接当缺陷修。
1. 🔴 **ASR diff 必须两边做数字归一化**：韩语数词串（`[일이삼사오육칠팔구십백천만억조]{2,}`）→`0`、
   阿拉伯数字串→`0`。不归一化会有 146 条假增删（사십사만→44만、오백원→500원、백퍼센트→100%）。
2. 🔴 **建 char_t 必须和归一化同步**（否则时间整体漂移）；delete 的时间用**夹住空隙的 ASR 字符**。
3. **局部语速判多字/漏字**：窗内 ASR 音节数 ÷ 净发声时长，与全局 cps 比（>1.35× 多字 / <0.65× 漏字）。
4. **头尾结构对比**：10ms 包络 + 「起音后是否 >150ms 近静音断口」，和 EL 标杆同口径比。
5. >3min 必须 10s 窗 + word_timestamps（`diag/verify_0902e.py` / `local_cps.py` / `head_check.py`）。
6. ⚠️ 能量峰数当音节数**不可靠**（±40% 误差，`count_syl.py` 已弃）。

## 🔴🔴🔴 文件头孤立段（v3.4 修 → v3.5 收紧）
omni 女声**外语专有名词开头**（로보티즈 类）起音抛 100-200ms 垃圾碎片 → 200-600ms 近静音 → 才进正文。
**文案触发**（女+로보티즈 3/3；女+반갑습니다/아모텍 0~1/3）→ 这解释 cases（아모텍 开头）0% 复现。
- 碎片 = **幻觉级垃圾**；🔴 **静音对照：-50dB 底噪即出「고맙습니다」**。
  whisper 近静音幻觉清单：고맙습니다 / 기상캐스터입니다 / MBC 뉴스 이덕영입니다 / 끝까지 시청해주셔서 감사합니다。
- 修法（`app/fix_chain.py` `_headfix_cut`，`TTS_HEADFIX=0` 回退）：
  🔴 插在 **load16 之后、任何增益之前**（阈值=原始 dBFS 口径）；切完把 `a.raw` 重指到切过的文件。
  四条件：持续语音起点 S / S 前间隙 / **碎片组恰好 1 组**且起点 <0.30s / 切点后 400ms 安全闸；5ms 淡入。
- 形态：A 起音即碎片（可切）/ B 词内断裂（切了吃字）/ #7 多组崩坏（保守不切，占 9%）。
- 🔴🔴 **v3.5 阈值（13 样本无一批判错）**：**gap ≥250ms**、碎片 **≤200ms**（v3.4 是 150/350，会误切实词「한」）。
  真碎片 gap 最小 260ms，真首词最大 190ms。**碎片时长单独用会误伤**；**间隙底噪 dB 无区分度**（都 ~-52dB）。
  ⚠️ whisper 把「한 퍼센트」规范化成「**1%**」→ 别据此判丢词，用**时长差**判（`TTS_HEADFIX=0` 对照重跑）。

## 🔴🔴 三问结论（详版 `out/三问深挖-202610190902.md`）
1. **模版比成品脏 = 测量伪影**：模版原生 24kHz click 0 处，重采样 16k 后出 7 处（2:3 非整数比造假 click）。
2. **数字念读 48 个数值 40 个全对**；员工报的「吃字」全是 whisper 数字伪影（干净单句同样听错 = 定罪）。
   **唯一真问题 = 文案病态数字** `한 사쩜사퍼센트` → 改文案即可。
3. **句末「嗯」= 浊音自然衰减残留，我们比 EL 还干净**（严 2.15 vs EL 3.40 / 分；宽 6.24 vs 10.32）。
   🔴 旧结论「我们 5.99 vs EL 3.89 = 1.55×」口径 bug **已作废**。能清但不该清：不可闻 + EL 更密 + 严禁清零。
   → 员工听到的「嗯」另有所源（疑文件头孤立段）。

## 🔴🔴🔴 两条版本轴（用户纠正，别再混）
| 轴 | 格式 | 含义 | 真相源 | 写入口 |
|---|---|---|---|---|
| **项目版本** | `v<主>.<次>` | 链路状态 | `VERSION` 的 `version =` | `release_version.py` |
| **反馈批次** | `z<N>` | 第几批发出去的题 | `VERSION` 的 `round =` | 同上（`--round`） |
- 题 id = `v<版本>-<z批次>-<维度>`；`ver` 字段**只放项目版本**。
- `/api/version` 三个布尔必须全 true；🔴 必须 `--set/--bump ... --round ... --items <题库>` **一次给全**
  （分两次会把 minor 顶两档）。`rewrite_items()` 同时换版本前缀和 `z` 段。

## 🔴🔴 EL 标杆口径
铁律：**EL 基线一律取 `cases/<id>/merged.wav`**（曾用非标准文件 → 女声音色错记 0.8172，实际 **0.9204**）。
- 🔴🔴 **EL 自身运行间方差极大**：click 2.56×、杂音 5.6× → 比值 <2.6×/<5.6× **都在 EL 噪声里，不算真差距**。

## 测量口径（不统一就作废）
| 指标 | 铁律 |
|---|---|
| click | **先归一到峰值 0.70** 再判 `|Δ|>=0.30`；重采样 16k；🔴🔴 **必须测压完 mp3 后的值** |
| 音色 | campplus 16k（cosy venv），同人阈值 0.75；EL-EL 中位 **0.9854(女)/0.9850(男)** |
| cps | 字数时长取自**同一条 job 记录**，用**含空格**字符数；EL 锚 女 7.45 / 男 8.37 |
| ASR | 🔴🔴 >3min **必须分段**（`asrseg53`，30s 窗）；整篇喂 whisper-small 会**循环复读** |
| 采样率 | 🔴🔴🔴 **成品绝不能 16 kHz**：清理全在 16k 跑，最后 `aresample=44100` 压 mp3（44.1k→mp3 音色 0.9999，16k→mp3 只 0.79）。旧工具写死 SR=16000 **只能中间处理**；保采样率版 = `clickfix53b`/`ltasmatch53b`/`tailclean53b2`/`53c2`/`eqfix53b` |
- σ 一律 `MAD×1.4826`；找「多出来的音」别用绝对 dB 阈值（差异在**时序结构**）。

## 现役链路（v3.5）
qwen :8000（**后处理只在这里**）/ cosy :8100 / gs :8200 / omni :8300 / vox :8400。
- 🔴🔴 **男女声都走 omnivoice**，路由看 `voices.json` 每音色 `engine`。**不按引擎名判性别**。
- 🔴🔴 **omni 走极简后处理**（`--no-pitch --no-clean --no-adaptive-hf --eq anull`），守卫必须
  `if _engine_name == "omnivoice":`。**GS 不走任何后处理**。
- 钩子是 `run_job()` 内嵌套函数：`_mideq53b_wrap`（窄带 EQ）→ `_clickfix53b_wrap`（click 插值），顺序不能反。
- 🔴 `/api/jobs` 只同源；**必须用 `tools/eval5` 的 login/submit_job/wait_job**。

## ✅ 已上线改动
| 版本 | 改动 | 结果 |
|---|---|---|
| v3.1 | 男声 clickfix53b（th=0.30） | click 1.21×→0.05×、音色零损失 |
| v3.2 | 男声窄带 EQ 1k:0/1.8k:+2/3k:0 | 中频 0.86→1.01、click 0.28×、音色 0.9329 |
| v3.3 | 女声接 EQ+clickfix（男0.30/女0.26） | 中频→1.01、click 1.24×→0.00×、音色 0.9227（六项全改善） |
| v3.4 | 文件头孤立段切除 headfix53 | 碎片 60%→0；音色 0.9994 / click 0→0 / ASR 无损 |
| v3.5 | headfix 判据收紧 + numcheck53 标注 | 修误切实词「한」：D_han 0/3 切除，真碎片仍 2/3 切除 |
- 决策原则：**只动「我们比 EL 差」的指标**。
- **当前基线**：男声 click 0.28×（回归实测 0.09×）/ 中频 1.01 / 杂音 0.73 / cps 8.27 / 音色 0.9329；
  女声 click 0.00× / 中频 1.01 / 杂音 1.12 / cps 1.00 / 音色 0.9227 → **客观指标已全部对齐 EL**。
  音色距 EL-EL 0.985 差 ~0.05，男女差距不大 → GS 微调紧迫性下降。

## 🔴 病态数字（numcheck53，v3.5 起生产只标注）
- `tools/numcheck53.py`：R1 native+Sino 混排 / R1b native+阿拉伯 / R2 native+大单位 /
  R3 小数点残缺·双点·小数带大单位 / R4 百分号倒置 / R5 native+百分号(WARN) / R7 双数混排 / R8 小数空格拆开(WARN)。
  测试 `tools/test_numcheck53.py`（解析 29/29 + 检出 40/40 + 写实 22/22 全绿）。
- 生产：`run_job()` 分块前扫 `job["text"]` → `job["num_warnings"]` + log.warning，
  **不改文案/不拦任务/不动音频**。模块 `app/numcheck53.py`（扁平导入）。开关 `TTS_NUMCHECK=0`。
- 🔴 **不是起音怪音触发源**（15 单对照：病态 0/3、合法 0/3、干净基线 1/3）。
  但它是**发音错误**源（TTS 忠实念「한 사쩜사 퍼센트」= 韩语不通）→ 标注给人改文案，别动链路。
- 防误报四闸门：native+原生量词合法（두 달）；이 主语词尾（数 token 须 ≥2 字符）；
  **한 冠形词尾**（`상승한 2,450선`）；名词闸门（`한 일이 있어서요`）。

## ⚠️ eval5 总分不可跨版本比
r20(10/6) 94.24% vs v3.5 88.73% **不可比**（中间隔 v3.1~v3.3）。比**指标表**才有效。

## 常用操作坑（最高频）
- 🔴🔴 **补丁自检必须 `compile()`，不能用 `ast.parse()`**；给 `run_job()` 里嵌套函数插 helper，
  缩进必须**同为 4 空格**，否则截断外层 async → 启动即崩。
- 🔴🔴 **插 helper 后必须跑 L1 实测，不能只信 audit53**（只查"字符串在不在"，查不到"没被执行"）。
  经典坑：firequalizer 输出比输入长 filter_size-1 样本（16k 长 320=20ms）→ 每次退回。
  修：helper **只返回 audio**；长度差**居中裁剪**，差 >0.5s 才真退回。
- 🔴🔴 **app/ 下扁平导入**：`from app import xxx` 启动即崩。
- 🔴🔴 **服务日志在 `service.log`**；`sudo -n journalctl` 需密码时**静默返回空、不报错**。
- 🔴🔴 **产物目录会加 `-N` 后缀**：`find_out` 取 mtime 最新；`glob(title+"*")` 会抓旧目录。
- 🔴🔴 **faster_whisper 必须 `local_files_only=True`**（无外网 → 挂死 4 分钟）；**不能传文件路径**
  → ffmpeg 降 16k，soundfile 读 ndarray 再 `transcribe`。
- 🔴🔴 **voices.json 外部改动会被冲掉**（旧 bug：任务结束整表写回内存快照）。已修
  `deploy/patch_voices53.py`（`refresh_voices_if_changed` 热重载 + `patch_voice` 单音色原子写）
  + `patch_voices53b.py`（写回前 chown/chmod 继承，否则变 root:root 就再也改不动）。
- 🔴 `voices/sources/` 是 root 属主 → 切片放 `voices/<name>.wav`。`voices.json` 是**扁平字典**（键=音色名），
  写入前照抄完整字段：path/label/ref_text/sr/dur/engine/gender/bench_case/origin/src_case/src_speaker/src_dur/clip/cps_lock/cps。
- 🔴 反馈台 `items.json` 必须 `{"items":[...]}`；前端选项**单选**；
  **每个有 file 的 option 都渲染播放器** → 只让第一个带 file。
- 🔴 `pkill -f "[x]xx"`；缓存名用**全路径 hash**（男女标杆都叫 merged.wav）。
- 🔴🔴 **永远不整目录 rsync**（本地是旧快照），只能按文件 scp；韩文目录名 ssh 双引号里常失败 → 先 cp 到 ASCII。
- 🔴🔴🔴 **本地 `VERSION` = 版本文档（~800 行），服务器 `VERSION` = 凭证文件（10 行），同名不同物！**
  本地文档只手工更新「当前版本行 + 变更表」，服务器凭证只由 `release_version.py` 写。
- 🔴 ssh 后台任务：`cd X && setsid nohup <绝对路径 python> ... & disown`。

## 🖥️ 服务器 / 队列
- **GPU0 = 3060(12G) qwen**；GPU1 = 3090(24G) cosy/gs/vox/omni（两卡都塞不下 qwen+omni）。
- 🔴🔴 **队列串行**：一条卡住 → 后面全排队。判真假挂：`systemctl show qwen-tts -p NRestarts`（0=没崩只是慢）。
- 耗时放大链：OOM → 反复初始化 → 语速崩 → 重试 ×3 ≈ 9 倍。✅ 修复后基准 **4142 字 / 515.9s / RTF 0.493**。
  `expandable_segments` 会报假性 OOM，用 `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8`。

## 操作速查
`ssh yuyin`（zheng，可 sudo；密码见用户级记忆）。代码 `/mnt/tts/projects/qwen3-tts`；
venv `…/qwen3-tts/venv/bin/python`；cosy venv `/mnt/tts/projects/cosy/venv`。时区 UTC（+8=北京）；
判活用根路径 200（`/health` 返回 404 是正常的）。
- 🔴 一次 ssh 里多个 sudo 只有第一个吃到密码 → `echo "$PW" | sudo -S -p '' bash -c 'a; b; c'`。
- zheng 可写 /mnt/tts/{data,projects,fb,diag}；**root 属主写不进**：`data/tone-ab`、
  `/etc/systemd/system/*.service.d/`（先 `mkdir -p`；改后 daemon-reload + restart）。git push 间歇 502，重试 2~4 次。

## 工具索引
**现役**：`clickfix53b`(click) / `eqfix53b`(窄带 EQ) / `measure53`(全量指标) / `spkcmp53`(音色 campplus) /
`asrseg53`(>3min 分段 ASR) / `alignchk48`(文本对齐) / `audit53`(服务端体检) / `release_version`(版本唯一写入口) /
`fbcheck53`(素材体检) / `tailclean53b2`+`53c2`(清理，**未上生产**) / `numcheck53`(病态数字) + `test_numcheck53`。
**deploy**：`patch_clickfix53b`(v3.1) / `patch_mideq53b`(v3.2) / `patch_female53`+`patch_mideqlog53`(v3.3) /
`patch_headfix53`(v3.4) / `patch_headfix53b`+`patch_numcheck53`(v3.5) / `patch_voices53`+`patch_voices53b`(引擎字段重置) /
`patch_verround53` / `patch_audit53_round`。
**diag**：`pipe_1019h` / `pipe_f53` / `l3f53_verify` / `l1_female53` / `gen_z33f` / `headfix_verify53` /
`headtype53`(头型普查) / `numtrigger53`(病态×怪音对照) / `voices53_test`。
