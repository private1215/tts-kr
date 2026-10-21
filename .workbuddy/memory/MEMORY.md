# tts-kr 速查索引（2026-10-21，v3.7 / 批次 z37）

> 细节版见 `.workbuddy/memory/archive-detail.md` 与当日日志；改链路前先读 `项目规范.md`，冲突以规范为准。

## 目标与验收
替代 EL：同文案我们 vs EL 员工听不出差别。员工零调参，**验收只看 L3 长篇**。
- 🔴 物理差≠可闻差（z44：13dB 改动 9 票仅 1 票）→ 定根因前先出裁决题。
- 🔴 z20：GS 女声客观全优仍被判「机器」=0 分 → 客观指标只是体检，换引擎只能盲听拍板。
- 🔴 片段口径与全篇口径可以相反 → 只用 L3 下结论。

## 🔴 数字零填充（v3.6，2026-10-21 上线）—— 本轮最大发现
员工逐秒报「0.1秒没音 / 额·嗯·啊 / 电磁音全程很多 / 碰碰声」→ 根因是**模型输出的停顿是数字真零**（|y|<1e-9），不是安静。取证 `fback53d`：报的 26 个时间点 **24 个**命中 ≥20ms 全零段（男 52 / 女 44 处，EL 标杆 **0** 处）。
- **根因代码**：`roomtone`/`declick` 只挂在 `_legacy_post = _engine_name not in NEW_ENGINES`，而 `NEW_ENGINES=("voxcpm","omnivoice")` → **v3.0 切引擎后 omni 一次都没跑过**（`git log -S declick -- app/server.py` 零匹配）。
- **修复**：`app/fix_chain.py` 链尾（响度二次校正后）调 `declick.apply(fill_db=-80, quiet_db=-80, mode="floor")`；开关 `TTS_ZEROFILL=0`；备份 `.bak-before-zerofill53`。
- 🔴 **必须 `sf.write(..., subtype="FLOAT")`**：ffmpeg 默认 pcm_s16le，16bit 下 −80dB 只剩 1.6 LSB = 白填。
- 🔴 **`fix_chain.py` 是 subprocess 调用，改完立即生效、不用重启**（区别于 app/ 下被 import 的模块）。
- 🔴 **与规范第 7 节封存的「抬停顿底噪 −56/−50」不是一回事**：那边是 z44 证明听不出的「做房间感」，这边是填深坑消硬切，填完**静音中位不变**（男 −61 vs EL −59、女 −54 vs EL −54）。别混为一谈。
- 定标：语音样本改动 0.0000%、时长零变化、p25 不变。L3：零段 45/39→**0**，陡降 161/46→**55/2**，p5 −139/−84→**−97/−81**（EL −91/−76）。
- 🔴 **z36 盲听：员工说「两段都有问题」**（不是听不出）→ **数字零不是主因，停止在这个方向加码**。v3.6 仍保留（客观全面优于 v3.5）。

## 🔴🔴 16k 带宽瓶颈（v3.7，2026-10-21）—— 「发闷 / 不如 EL」的真凶
**现役链路有效带宽只有 8 kHz**：`app/fix_chain.py` 全链路走 `load16()`（`ffmpeg -ar 16000`），omni 原生 24k（`omni_server.py:44 SR=24000`）一进链就被砍掉 8k 以上，成品贴 44.1k 标签却只有 8k 内容。
- **取证**（`docs/r53-16k带宽瓶颈根因-2026-10-21.md`，同文案 109.1s 男 / 120.1s 女）：

  | | 8-10k | 10-12k |
  |---|---|---|
  | omni 原始 24k | 1.32e-3 | 2.28e-4 |
  | **16k 链路（现役）** | 1.96e-5（÷67） | **1.04e-8 地板** |
  | **24k 链路（v3.7）** | **1.32e-3** | **2.28e-4** |
  | EL 男 / 女 | 8.24e-4 / 1.05e-3 | 1.10e-4 / 1.11e-3 |

  音色余弦 vs EL：男 **0.9429→0.9523**、女 0.9307→0.9313（EL 上限 0.9628）。时长/LUFS/停顿数/cps 全不变。
- ❌ **已推翻两个假设**：①「模型/声码器不给高频」——omni 原始 8-12k 比 EL 还高 1.6~2.1×；②「mp3 砍高频」——192k CBR 前后 8-12k **小数点都一样**，libmp3lame 无辜。
- 🔴🔴 **为什么客观指标全绿却听出差**：所有指标（campplus/whisper）都在 **16k 口径**下测，这次降采样对它们**完全隐形**（`server.py:1474` 原话）。→ **量 8k 以上必须出 44.1k 成品再量，绝不能拿 16k 文件量**（`server.py:454` 早有此警告）。
- **修复**：`FIX_SR = int(os.environ.get("TTS_FIX_SR","24000"))`，`load16` 用 `str(FIX_SR)`；备份 `.bak-before-fixsr53`。
- ✅ **兼容性已逐项排除**：`mideq53b`/`clickfix53b` 都「保采样率」（clickfix53b 检测在 16k 域、改样本映射回原生 sr）；`pause_clean_v3` 用 silencedetect（电平阈值）；`f0_median` 是 16k 口径但 omni 走 `--no-pitch` 不调用。**唯一硬编码点就是 load16**。
- ⚠️ **遗留风险**：24k 后男声 10-12k 达 EL 的 **2.07×**、女声 1.33×，可能偏亮/齿音重 → **必须盲听确认**；若刺耳用 `AIR_EQ` 的 `equalizer=f=10000:t=h:w=3200:g=-6` 压。
- 🔴 `AIR`（`aexciter=amount=5:freq=11200:ceil=13000`，补 11-13.5k）挂在 `if _legacy_post and AIR:`，**omni 从未跑过** —— 与数字零同一个坑。但**先修 16k 才有意义**（16k 下补了也会被砍），故排在 v3.7 之后。

## z36 轮（2026-10-21 发布，反馈已回收）
6 题：男/女 × 杂音/卡顿（v3.6 vs v3.5 对照）+ 对 EL + 控制题。
结果：`m_noise C / m_gap C / f_noise C / f_gap C / m_el B / ctrl C`；
用户澄清 = **只有 EL 那段没问题，其余 5 段都有问题**。
🔴 **出题规范补充（血的教训）**：A/B「有没有」题的选项必须**同时**给
「①第1段有问题 ②第2段有问题 ③两段都有 ④两段都没有」。z36 只给了
「两段都干净」= 光有"都没有"、漏了"都有"，员工只能把"两段都有问题"
挤进 C，被我误读成"听不出区别"。**下次出题先查这个。**
🔴 z35 其实有人答（admin 00:38，ctrl C 有效）：**4 题全选 EL**，留言「第五题有电磁音」；
디아 那份 ctrl 答错作废。→ 电磁音是跨版本稳定症状。

## headfix 文件头孤立段（#7 已结案）
`app/fix_chain.py _headfix_cut`（`TTS_HEADFIX=0`）。A_single 才切：**恰好 1 组 + ≤200ms + 起点<0.30s + 间隙≥250ms + 切点后 400ms 安全闸**。
- 🔴🔴 **#7（≥3 组，占 9%）不能切——头部就是真词**（`shorthead53`+`headasr53`，11 样本零误切）。f_head 选②→只能生成侧解决。

## 版本轴 & 发布（别混）
项目版本 `v<主>.<次>` / 反馈批次 `z<N>`，真相源都是服务器 `VERSION`，唯一写入口 `release_version.py`（在服务器 `tools/` 下，不是项目根）。
- 题 id=`v<版本>-<z批次>-<维度>`；`/api/version` 三布尔必须全 true；`--set/--bump … --round … --items` **一次给全**。
- 🔴 **`--items` 只改 id 前缀，不安装**——必须再 `cp items_zN.json /mnt/tts/fb/items.json`（z35、z36 都踩过）。本地 `VERSION`=文档，服务器=凭证，同名不同物。
- 反馈台 `/feedback`（`/fb` 404）；items.json 必须 `{"items":[…]}`；盲听题**不许出现身份词**（dim 也查，它同样显示在卡片上）。

素材 `L3z36-*`(492.3/454.9s) vs `L3v35c-*`(492.6/458.2s)；取样位置**自动探测 B 侧零段最密集的 20s 窗**（规范 7）。
⚠️ 客观（4-8k 1.00×、静音4-8k 0.50×、click 0.25×）显示我们比 EL 还"干净"却听出脏 → **该维度客观指标失效，只能盲听**。

## 测量口径（不统一就作废）
click：归一到峰值 0.70 判 `|Δ|>=0.30`，16k，**测压完 mp3 后的值**。音色：campplus 16k 多窗，阈值 0.75。cps：同一条 job 记录含空格字数；EL 锚 女 7.45/男 8.37。
**数字零三件套**：≥20ms 全零段(|y|<1e-9)、30ms 内陡降≥25dB、静音段样本级 p5/中位——须样本级同口径，与早期帧级不可混。
🔴🔴 **链路绝不能跑 16k**（v3.7 已修，此前 v3.6 及更早全程 16k = 有效带宽只有 8k）：主链在 **24k** 跑，最后 `aresample=44100` 压 mp3。EL 基线一律 `cases/<id>/merged.wav`（**也是 24k**，所以 8-12k 是公平对比）；EL 运行间方差 click 2.56×/杂音 5.6×。
🔴 **量 8k 以上必须出 44.1k 成品再量**（`server.py:454`）：拿 24k/16k 文件量会看到"砖墙"假象。

## 现役链路（v3.7）
qwen :8000（后处理只在这）/ cosy :8100 / gs :8200 / omni :8300 / vox :8400。男女都走 omnivoice（路由看 `voices.json` 每音色 engine）；omni 极简后处理（`--no-pitch --no-clean --no-adaptive-hf --eq anull`）+ **链尾数字零填充** + **全链路 24k（FIX_SR）**。GS 无后处理。钩子顺序 `_mideq53b_wrap`(EQ)→`_clickfix53b_wrap`(click)，不能反。`/api/jobs` 用 `tools/eval5`。
已上线：v3.1 男 clickfix / v3.2 男 EQ / v3.3 女声接管 / v3.4 headfix / v3.5 headfix 收紧+numcheck53 / v3.6 数字零填充 / **v3.7 链路 16k→24k（带宽 8k→12k）**。

## 病态数字 numcheck53（只标注不拦截）
8 规则 43/43 全绿；`run_job()` 分块前扫 text→`num_warnings`，不动音频；开关 `TTS_NUMCHECK=0`。防误报闸门含 `_glued_in_word`。
- 🔴🔴 **部署铁律：改完 app/ 下模块，服务重启时间必须 > 文件 mtime**（Python 导入缓存）。
- ⚠️ 疑似误报待核：`영쩜삼일퍼센트`(0.31%) 被报 `쩜삼일` INVALID。

## 常用操作坑
- 🔴🔴 补丁自检必须 `compile()`；插后必须跑 L1 实测（audit53 查不到"没被执行"）。
- 🔴🔴 app/ 扁平导入 `from app import xxx` 启动即崩。产物目录加 `-N` 后缀→取 mtime 最新。
- 🔴🔴 faster_whisper 必须 `local_files_only=True`；传 ndarray 不传路径。
- 🔴 Python 字符串坑：相邻字面量后紧跟 `(...)` = 函数调用；`%` 优先级高于 `+`，`"a"+DEC%(x)` 会把参数喂给无格式符的 DEC → 整串先拼好再一次格式化（`mk_z36.dec_note` 就是为此）。
- 🔴 whisper 个别 20s 窗解码抽风 → 出题脚本设 ASR 字数下限 + 位置回退。
- 🔴 服务器 app/ 是真相，本地是旧快照 → **按文件 scp，永不整目录 rsync**（先拉服务器版同步再改）。
- voices.json 外部改动会被冲掉。`ssh yuyin`（sudo 密码见用户级记忆），时区 UTC。git push 间歇 502 重试 2~4 次。

## 工具索引
`clickfix53b`/`eqfix53b`/`measure53`/`spkcmp53`/`asrseg53`/`audit53`/`release_version`/`fbcheck53`/`numcheck53`+test/`mk_z35`/**`mk_z36`**(出题)/`gen_z35_long`/**`gen_z36_long`**(PREFIX 切版本)/`longcov53`/`cmp_l3v35`/`shorthead53`+`headasr53`/**`fback53d`**(三栏数字零)/**`zerofix53b`**(填充电平网格)/**`l3check53`**(L3验收)/**`l1_zerofill53`**(L1+off对照)/**`z36audit`**(盲听结果审查：频段/杂音/click 对照)。

## 产品铁律：员工零调参（已核实，别再当缺口）
员工只做：输标题+输文案+选男女→等结果。emotion 恒 standard。仍开：TTS_SLOWDYN=1 / TTS_BREATH_SYN=1 / TTS_PEAK_FIT=1 / TTS_LOUD_TOL=0.15 / PROSO=0。
