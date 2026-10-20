# tts-kr 速查索引（2026-10-20 晚，v3.5 / 批次 z35）

> 细节版见 `.workbuddy/memory/archive-detail.md` 与当日日志；改链路前先读 `项目规范.md`，冲突以规范为准。

## 目标与验收
替代 EL：同文案我们 vs EL 员工听不出差别。员工零调参，**验收只看 L3 长篇**。
- 🔴 物理差≠可闻差（z44：13dB 改动 9 票仅 1 票）→ 定根因前先出裁决题。
- 🔴 z20 反向错误：GS 女声客观全优仍被判「机器」=0 分 → 客观指标只是体检，换引擎只能盲听拍板。
- 🔴 片段口径与全篇口径可以相反 → 只用 L3 下结论。

## headfix 文件头孤立段（#7 已结案）
`app/fix_chain.py _headfix_cut`（`TTS_HEADFIX=0` 回退）：插在 load16 后、任何增益前，切完 `a.raw` 重指；切点=持续语音起点。A_single 才切：**恰好 1 组 + ≤200ms + 起点<0.30s + 间隙≥250ms + 切点后 400ms 安全闸**。
- 🔴🔴 **#7（≥3 组，占 9%）不能切——头部就是真词**：로보티즈 被压碎成「로봇이쥐쥐」、반갑습니다 完好，全篇 ASR 首词来自头部（`shorthead53`+`headasr53`，11 样本零误切）。旧「幻觉垃圾」结论作废。f_head 选②→只能生成侧解决。
- v3.5 阈值定标（13 样本）：真碎片 gap≥260ms vs 真首词≤190ms；whisper 把「한 퍼센t」规范化成「1%」→ 判丢词用时长差。

## 员工反馈核验法（先复核再动手）
ASR diff 两边数字归一化（数词串→0）；char_t 同步归一化；局部语速=窗内音节÷净时长 vs 全局 cps（>1.35×多字/<0.65×漏字）；>3min 必须 `asrseg53` 30s 窗（整篇喂 whisper 会循环复读）。

## 版本轴 & 发布（别混）
项目版本 `v<主>.<次>`（链路状态）/ 反馈批次 `z<N>`，真相源都是服务器 `VERSION`，唯一写入口 `release_version.py`。
- 题 id=`v<版本>-<z批次>-<维度>`；`/api/version` 三布尔必须全 true；`--set/--bump … --round … --items` **一次给全**（分两次把 minor 顶两档）。
- 🔴 **`--items` 只改 id 前缀，不安装**——必须再 `cp items_zN.json /mnt/tts/fb/items.json`（z35 漏这步导致不一致）。本地 `VERSION`=800 行文档，服务器=10 行凭证，同名不同物。
- 反馈台入口 `/feedback`（`/fb` 404）；items.json 必须 `{"items":[…]}`；选项单选只让第一个带 file。

## z35 轮（2026-10-20 晚已发布，链路仍 v3.5）
v3.1~v3.5 从未盲听验收；z34 出题后员工没反馈，按用户指示继续优化再出新版。
- 5 题=男中段@390 / 男前段@100 / 女中段@390 / 女起音@0(8s) / 控制题；A=我们 B=EL，ASR 对齐 0.90~0.98；响度统一 −26.1 LUFS。
- 新取样长语音 `L3z35-male` 489.2s / `L3z35-female` 456.8s；完整性覆盖率 0.941/0.956、复读 0、循环 0。
- vs EL：click 0.27×/0.00×、质心中频 1.00×/0.98×、杂音噪声内、cps 0.99×/1.01×、音色 0.9407/0.9163(阈值0.75)；男停顿 +0.6%（旧取样 +4.5%=取样波动，按 z44 接受不停）。**最大剩余差距=音色 0.05~0.06（引擎上限，只能盲听定）**。
- 映射：选②→回 S1；选③→验收通过；f_head 选②→#7 生成侧挂账；ctrl 选①/②→整轮作废。

## 测量口径（不统一就作废）
click：归一到峰值 0.70 判 `|Δ|>=0.30`，16k，**测压完 mp3 后的值**。音色：campplus 16k 多窗（cosy venv），阈值 0.75，EL-EL≈0.985。cps：同一条 job 记录含空格字数；EL 锚 女 7.45/男 8.37。
🔴🔴 成品绝不能 16k：清理在 16k 跑，最后 `aresample=44100` 压 mp3（44.1k→mp3 音色 0.9999，16k→mp3 只 0.79）；保采样率版=`clickfix53b`/`ltasmatch53b`/`tailclean53b2`/`53c2`/`eqfix53b`。EL 基线一律 `cases/<id>/merged.wav`；EL 运行间方差 click 2.56×/杂音 5.6×，低于此不算差距。

## 现役链路（v3.5）
qwen :8000（后处理只在这）/ cosy :8100 / gs :8200 / omni :8300 / vox :8400。男女声都走 omnivoice（路由看 `voices.json` 每音色 engine，**不按引擎名判性别**）；omni 极简后处理（`--no-pitch --no-clean --no-adaptive-hf --eq anull`，守卫 `if _engine_name=="omnivoice":`）；GS 无后处理。钩子顺序：`_mideq53b_wrap`(EQ)→`_clickfix53b_wrap`(click)，不能反。`/api/jobs` 只同源，用 `tools/eval5`。
已上线：v3.1 男 clickfix / v3.2 男 EQ / v3.3 女声接管 / v3.4 headfix / v3.5 headfix 收紧(250ms/200ms)+numcheck53。

## 病态数字 numcheck53（只标注不拦截）
R1 native+Sino 混排等 8 规则，43/43 全绿；`run_job()` 分块前扫 text→`num_warnings`+log.warning，不动音频；开关 `TTS_NUMCHECK=0`。防误报五闸门含**数词嵌在词里 `_glued_in_word`**（修「매도세만」类）。不是起音怪音源但是发音错误源→标注给人改文案。
- 🔴🔴 **部署铁律：改完 app/ 下模块，服务重启时间必须 > 文件 mtime**（`stat -c %y` vs `systemctl show qwen-tts -p ActiveEnterTimestamp`）——2026-10-20 第五闸门磁盘有、进程没加载（Python 导入缓存），重启才生效。
- ⚠️ eval5 总分不可跨版本比（隔了 v3.1~3.3），只比指标表。

## 常用操作坑
- 🔴🔴 补丁自检必须 `compile()`；嵌套函数插 helper 缩进同 4 空格；插后必须跑 L1 实测（audit53 查不到"没被执行"）；helper 只返回 audio，长度差居中裁剪。
- 🔴🔴 app/ 扁平导入：`from app import xxx` 启动即崩。服务日志 `service.log`。产物目录加 `-N` 后缀→取 mtime 最新。
- 🔴🔴 faster_whisper 必须 `local_files_only=True`；不能传文件路径（ffmpeg 降 16k→soundfile 读 ndarray→transcribe）。
- 🔴 **whisper 个别 20s 窗口解码抽风**（150s/20s 只出尾 8 字、25s 完整）→ 出题脚本对"我们侧 ASR 字数"设下限+位置回退（`mk_z35` 已带）。
- voices.json 外部改动会被冲掉（已修 `patch_voices53+53b`）；`voices/sources/` root 属主。永远不整目录 rsync；韩文目录名先 cp 到 ASCII。ssh 后台：`setsid nohup <绝对路径> … & disown`。
- 服务器：GPU0=3060 qwen、GPU1=3090 其余；队列串行，`NRestarts`=0 只是慢。`ssh yuyin`（sudo 密码见用户级记忆），时区 UTC，判活根路径 200。git push 间歇 502 重试 2~4 次。

## 工具索引
`clickfix53b`(click)/`eqfix53b`(EQ)/`measure53`(全量)/`spkcmp53`(音色)/`asrseg53`(分段ASR)/`audit53`(体检)/`release_version`(版本写入口)/`fbcheck53`(素材)/`numcheck53`+test(43/43)/**`mk_z35`**(出题)/`gen_z35_long`(长语音)/`longcov53`(完整性P0)/`cmp_l3v35`(对比)/`shorthead53`+`headasr53`(#7证据链)。deploy：`patch_clickfix53b`/`patch_mideq53b`/`patch_female53`/`patch_headfix53(+53b)`/`patch_numcheck53`/`patch_voices53(+53b)`/`patch_vlib_collapse`。

## 产品铁律：员工零调参（已实现且已核实，别再当缺口）
员工只做：输标题+输文案+选男女→等结果。`admin-only` 隐藏高级项（`#voiceLibraryCard` 已默认折叠）；emotion 下拉早已下架（提交恒 standard，`server.py:1288` 只对管理员 A/B 生效）；情绪来源=案例参考音频复刻。仍开着：TTS_SLOWDYN=1 / TTS_BREATH_SYN=1 / TTS_PEAK_FIT=1 / TTS_LOUD_TOL=0.15 / PROSO=0(关闭态=已验收基线)。
