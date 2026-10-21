# tts-kr 速查索引（2026-10-21，v3.6 / 批次 z36）

> 细节版见 `.workbuddy/memory/archive-detail.md` 与当日日志；改链路前先读 `项目规范.md`，冲突以规范为准。

## 目标与验收
替代 EL：同文案我们 vs EL 员工听不出差别。员工零调参，**验收只看 L3 长篇**。
- 🔴 物理差≠可闻差（z44：13dB 改动 9 票仅 1 票）→ 定根因前先出裁决题。
- 🔴 z20 反向错误：GS 女声客观全优仍被判「机器」=0 分 → 客观指标只是体检，换引擎只能盲听拍板。
- 🔴 片段口径与全篇口径可以相反 → 只用 L3 下结论。

## 🔴 数字零填充（v3.6，2026-10-21 上线）—— 本轮最大发现
员工逐秒报「0.1秒没音 / 额·嗯·啊 / 电磁音全程很多 / 碰碰声」→ 根因是**模型输出的停顿是数字真零**（|y|<1e-9），不是安静。取证 `fback53d`：报的 26 个时间点 **24 个**命中 ≥20ms 全零段（男 52 / 女 44 处，EL 标杆 **0** 处）。
- **根因代码**：`roomtone`/`declick` 只挂在 `_legacy_post = _engine_name not in NEW_ENGINES`，而 `NEW_ENGINES=("voxcpm","omnivoice")` → **v3.0 切引擎后 omni 一次都没跑过**（`git log -S declick -- app/server.py` 零匹配）。
- **修复**：`app/fix_chain.py` 链尾（响度二次校正后）调 `declick.apply(fill_db=-80, quiet_db=-80, mode="floor")`；开关 `TTS_ZEROFILL=0`；备份 `.bak-before-zerofill53`。
- 🔴 **必须 `sf.write(..., subtype="FLOAT")`**：ffmpeg 默认 pcm_s16le，16bit 下 −80dB 只剩 1.6 LSB = 白填。
- 🔴 **`fix_chain.py` 是 subprocess 调用，改完立即生效、不用重启**（区别于 app/ 下被 import 的模块）。
- 🔴 **与规范第 7 节封存的「抬停顿底噪 −56/−50」不是一回事**：那边是 z44 证明听不出的「做房间感」，这边是填深坑消硬切，填完**静音中位不变**（男 −61 vs EL −59、女 −54 vs EL −54）。别混为一谈。
- 定标：语音样本改动 0.0000%、时长零变化、p25 不变。L3：零段 45/39→**0**，陡降 161/46→**55/2**，p5 −139/−84→**−97/−81**（EL −91/−76）。

## headfix 文件头孤立段（#7 已结案）
`app/fix_chain.py _headfix_cut`（`TTS_HEADFIX=0`）：load16 后、任何增益前，切完 `a.raw` 重指。A_single 才切：**恰好 1 组 + ≤200ms + 起点<0.30s + 间隙≥250ms + 切点后 400ms 安全闸**。
- 🔴🔴 **#7（≥3 组，占 9%）不能切——头部就是真词**（`shorthead53`+`headasr53`，11 样本零误切）。f_head 选②→只能生成侧解决。

## 版本轴 & 发布（别混）
项目版本 `v<主>.<次>` / 反馈批次 `z<N>`，真相源都是服务器 `VERSION`，唯一写入口 `release_version.py`（在服务器 `tools/` 下，不是项目根）。
- 题 id=`v<版本>-<z批次>-<维度>`；`/api/version` 三布尔必须全 true；`--set/--bump … --round … --items` **一次给全**。
- 🔴 **`--items` 只改 id 前缀，不安装**——必须再 `cp items_zN.json /mnt/tts/fb/items.json`（z35、z36 都踩过）。本地 `VERSION`=文档，服务器=凭证，同名不同物。
- 反馈台 `/feedback`（`/fb` 404）；items.json 必须 `{"items":[…]}`；盲听题**不许出现身份词**（dim 也查，它同样显示在卡片上）。

## z36 轮（2026-10-21 发布，v3.6）
6 题：男杂音 / 男卡顿 / 女杂音 / 女卡顿（**v3.6 vs v3.5 同文案同 seed 对照**）+ 男声对标杆 + 控制题；ASR 对齐 0.878~0.972，响度 −26.1 LUFS。
- 🔴 取样位置是**自动探测 B 侧（v3.5）零段密集区**（男 100s/260s 各 3 处、女 0s/60s 各 2 处）——规范 7 禁止在无现象素材上验证。
- 映射：选①(v3.6)→保留发布；选②(v3.5)→回退重定标；选③→按 z44 封存；ctrl 选①/②→整轮作废。
- 素材：`L3z36-*` 492.3s/454.9s（v3.6）与 `L3v35c-*` 492.6s/458.2s（v3.5 对照）。

## 测量口径（不统一就作废）
click：归一到峰值 0.70 判 `|Δ|>=0.30`，16k，**测压完 mp3 后的值**。音色：campplus 16k 多窗，阈值 0.75。cps：同一条 job 记录含空格字数；EL 锚 女 7.45/男 8.37。
**数字零三件套**（新增口径）：≥20ms 全零段(|y|<1e-9)、30ms 内陡降 ≥25dB、静音段(<-45dB) 样本级 p5/中位 —— 必须**样本级同口径**，与早期帧级 dB 分位不可混。
🔴🔴 成品绝不能 16k：清理在 16k 跑，最后 `aresample=44100` 压 mp3。EL 基线一律 `cases/<id>/merged.wav`；EL 运行间方差 click 2.56×/杂音 5.6×。

## 现役链路（v3.6）
qwen :8000（后处理只在这）/ cosy :8100 / gs :8200 / omni :8300 / vox :8400。男女都走 omnivoice（路由看 `voices.json` 每音色 engine）；omni 极简后处理（`--no-pitch --no-clean --no-adaptive-hf --eq anull`）+ **链尾数字零填充**。GS 无后处理。钩子顺序 `_mideq53b_wrap`(EQ)→`_clickfix53b_wrap`(click)，不能反。`/api/jobs` 用 `tools/eval5`。
已上线：v3.1 男 clickfix / v3.2 男 EQ / v3.3 女声接管 / v3.4 headfix / v3.5 headfix 收紧+numcheck53 / **v3.6 数字零填充**。

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
- voices.json 外部改动会被冲掉；`voices/sources/` root 属主。`ssh yuyin`（sudo 密码见用户级记忆），时区 UTC。git push 间歇 502 重试 2~4 次。

## 工具索引
`clickfix53b`/`eqfix53b`/`measure53`/`spkcmp53`/`asrseg53`/`audit53`/`release_version`/`fbcheck53`/`numcheck53`+test/`mk_z35`/**`mk_z36`**(出题)/`gen_z35_long`/**`gen_z36_long`**(PREFIX 切版本)/`longcov53`/`cmp_l3v35`/`shorthead53`+`headasr53`/**`fback53d`**(三栏数字零)/**`zerofix53b`**(填充电平网格)/**`l3check53`**(L3验收)/**`l1_zerofill53`**(L1+off对照)。

## 产品铁律：员工零调参（已核实，别再当缺口）
员工只做：输标题+输文案+选男女→等结果。`admin-only` 隐藏高级项；emotion 恒 standard；情绪来源=案例参考音频复刻。仍开：TTS_SLOWDYN=1 / TTS_BREATH_SYN=1 / TTS_PEAK_FIT=1 / TTS_LOUD_TOL=0.15 / PROSO=0。
