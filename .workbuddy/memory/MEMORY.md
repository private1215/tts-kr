# tts-kr 速查索引（2026-10-23，**生产 v4.4 链路 + r65 底噪 + r67 啊额音/电子音修复**）

> 历史细节看`.workbuddy/memory/YYYY-MM-DD.md`（10-15~10-22 全在）+ `archive-detail.md`；改链路前先读 `项目规范.md`，冲突以规范为准。

## 🔴🔴 协作约定（用户明确要求，逐条照做）
1. **非必要不发反馈批次**。每批 1~2 票、控制题常失效（±1 档）。发之前先问：这批能定什么案？
2. **代码改动必须及时入库**。`/mnt/tts/projects/qwen3-tts` **不是 git 仓库**，不同步就永久丢失。
   → 收工前跑 `tools/sync_from_server.sh`（幂等，比对 205 文件）→ 提交 → push（重试≤4 次，常 502）。
   ⚠️ 只按文件 scp，**绝不整目录 rsync**；systemd drop-in 手动归档到 `deploy/`。
3. 汇报要分清「**生产真的改了**」和「**只是发了批次**」。
4. 遇到问题先自己解决（sudo 密码 `a19910119`），别停下来问。

## 目标与验收
替代 EL：同文案我们 vs EL，员工听不出差别。员工零调参，**验收只看 L3 长篇**。
- 🔴 物理差 ≠ 可闻差 → 定根因前先出裁决题。
- 🔴 客观指标只是体检；换引擎/调参只能盲听拍板。
- 🔴 片段口径与全篇口径可相反 → 只用 L3 下结论。
- 🔴 **题里不放人声标杆**（员工听 EL 没意义）→ 用**重复控制题**校准；代价：只能批内排序。
- 🔴 题面文案必须用案例原始 `seg*.txt`。
- 🔴 **跨批次 A 票统计**：z42（L3 长篇切片）A 占 50%，z40/41/43/44（25s 短片段）只有 0~25%
  → **短片段系统性低估 → 后续一律 L3 长篇切片**。

## 现役链路
qwen :8000（后处理只在这，也跑 fix_chain）/ cosy :8100 / gs :8200 / omni :8300 / vox :8400。男女都走 omnivoice。
omni 后处理 `--no-pitch --no-clean --no-adaptive-hf --eq anull` + 链尾零填充 + 全链 24k。
钩子顺序 `_mideq53b_wrap`→`_clickfix53b_wrap` 不能反。已上线：v3.1~v3.9 + **v4.1 num_step=48** + **r65 底噪** + **r67 伪声/电子音**。

## 🔴🔴 r67（2026-10-23）：「啊额音 / 闷嗯 / 电子音」一次做透（详见 10-23 日志）
- **为什么 r66 反复修不好**：EDGEFIX 用频谱分类（mid≥0.45 且 hi≤0.15），38 边界只中 13 个
  —— 段终点按「第一个 >-14dB 帧」找，已切进真实语音起振斜坡，谐波把 hi 顶到 0.16~0.34 → 拒掉。
  **判据建立在自己算出来的边界上 = 靠运气**。且只 ×0.15 → -17dB 伪声压完还有 -33dB = 「闷嗯」。
- **r67 三改**：① EDGEFIX 结构化（pad 数字零为锚，零段→持续语音起点整段压到 -100dB，
  由 declick 整段换房间声，**不留残响**），覆盖率 13/38→37/38；② `_rumble` 独立扫低频隆隆
  （低频>85% 且中频<15% 且 >-40dB），命中 8 处含文件头 -22.7dB/97%低频 启动瞬态；
  ③ 修 `declick._pink` 真 bug：`rfftfreq(m,1.0)` 是**归一化频率**，阈值却按 Hz 写 →
  退化成 +12dB/oct 高通，填充噪声 **87% 能量在 8-12kHz** = 「电子音」根因（r30 起潜伏）。
- **验收数据**：伪声区峰值 -23.9→**-61.4dB**、高于-50dB 的帧 1686→**0**；
  填充谱形 8-12kHz 86.8%→**1.3%**；CER 0.0749→**0.0652**（更准，伪声原被 ASR 误识别）；
  e2e 女 460s/CER 0.041、男 413s/CER 0.0389。git b70eb73。
- 🔴 教训：**分类判据不能建立在自己算出来的边界上**；**衰减 ≠ 消除**（×0.15 留残响=新毛病）。
- 🔴 omni **完全绕过文本前端**（`server.py:1375` `NEW_ENGINES → chunks=[job["text"]]`）→ `normalize_korean()`/
  `TTS_KO_NUM` 对 omni 零作用。数字读法实测正确（`1,830억원→천팔백삼십억원`）。
- 🔴 `_legacy_post = engine not in ("voxcpm","omnivoice")` → omni 整条**跳过老后处理，包括 roomtone**（`server.py:2333`）。
- per-voice 开关 `TTS_OMNI_PERVOICE=1` 在 **qwen-tts** 进程；参考表 `/mnt/tts/data/voices/omni_prompts.json`
  （`_load_prompt_overrides()` **只在启动时读**，改完必须重启 omni-tts）。
- omni 跑 `/mnt/tts/projects/omnivoice/omni_server.py`（**不是** qwen3-tts 里的 deploy 副本，那个已旧）。
  解码覆盖 `_GEN_KEYS`：denoise/cfg/class_t/num_step/t_shift + r65 新增 chunk/pad/fade；空=行为不变。日志 `gen=` 可验。

## 🔴🔴 r65（2026-10-23）：员工秒级反馈 → 定位并修好「卡音」
员工原话：`2，9秒卡音 23秒卡音 / 3，11秒 0.1秒突然弱音 / 4，3秒卡音… / 6，6秒卡音 14秒弱音 19秒吃字 / 7，1秒卡音 11秒卡音 26秒没音 / 8、9 整体电子音感`。
- **报点对上了**：8/10 命中 Y_f 的数字真零断点（±1.5s），其余 ≤3s。
- **机制**：`fade_and_pad_audio()`（omnivoice `utils/audio.py:236`）给每个分块首尾插 `pad_duration=0.1` 秒纯零
  + 0.1s 淡入淡出 → 块边界是 100ms 真空硬切；叠加 omni 跳过 roomtone → 断点间隔中位 **12.46s ≈ 分块周期**。
  女声 38 处 ≥40ms 真零，**EL 女声 0 处**；男声 169，EL 男声 83。
- **根因不是分块**：A/B 实测 `pad_duration=0` 只把 9→7，`chunk=30` 9→5 → 主体是**模型直出的数字真零**，
  而我们的填充电平错在 **-80dB 比 EL 安静 18dB**。
- ✅ **已上线**：ZEROFILL 填充电平按性别可配，女 **-65dB** / 男 回落 -80（drop-in `deploy/systemd/30-zerofill65.conf`，
  加在 **qwen-tts** 上；回滚=删文件+daemon-reload+restart）。实测断点 **38→0**、p0.1 -80.5→-65.6（EL -62.6）、
  中位/p75 不变、语音段（|x|>0.01，68.9%）**字节级零改动**。
  🔴 **男声不能抬**：案例 1a6c20a60481 静音底部 p1=-86/p0.1=-90.5，任何抬升都让 L1 变差。
  ⚠️ 历史冲突已核对：r29 试 -66 被否是按当年案例（静音 p5=-89）判的，当前女声案例是 -62.6，反对理由不成立。
- ⚠️ **测量口径大坑**：`ffmpeg` 转 16bit 会把 -80dB 读成 -90dB（截断偏置）→ 量底噪必须 `sf.read` **直接读浮点**。
- 备选未上：`TTS_OMNI_CHUNK=30`（断点再 -44%，间隔 12.4→26.8s，但动模型分块有风险）。
- 🔴🔴 **37 案例普查 + 留出验证（12:10）**：13 男案例静音底部 p1=-82~-92（83~148 个断点），
  23 女案例 p1=-53~-64（0~9 个断点）→ **录音棚级干净 vs 有房间声是案例属性，与性别强相关**。
  抽两个没见过的案例各自当参考做留出验证：女 `557b83ff98c8` 上 -65 让断点 8→0（L1 30.1→12.3，
  理论最优 -60 时 L1=3.5）；男 `8197d9ac8a35` 上 -85 略优于 -80（19.5→17.9）。
  → **r65 的 -65 在留出案例上同样成立；理论最优 ≈ 该参考案例自身的 p0.1。**
  ⚠️ **"按性别一个固定值"只因为现役两条参考正好各居一端。新增音色前必须按新案例重算**——
     女声 23 例里有 16 例 p0.1≈-90 但 p1≈-60，对它们填 -65 会把合法深静音抬 25dB = r29 失败模式。
  男声保持 -80 不改成 -85/-88：EL 男声的 -90 是录音实录（无量化噪声），我们的合成粉噪填太轻
  会掉进 libmp3lame 比特分配极少的区间 → 有"滋滋"风险。宁响勿噪。

## 🔴 已排除清单（别再测）
16k 带宽（v3.7 已修但不可闻）、数字零（v3.6 已修，非主因）、韵律（F0std/p90-10/包络）、HNR、LTAS 1/6 倍频程、
jitter/shimmer、音节级时长、11.2k 窄带电磁音（EL 也有）、omni ref_text 配比、15s 接缝深静音、`language` 参数、
`normalize_text`（伤韩语）、CosyVoice3 SFT、omni 换声码器、同说话人 LoRA 加自然度。
**三条轴被耳朵判平/判死**：解码参数（z40/41/43 步48/64/ct0.3/cfg3/denoise_off 全平）、
参考选段（z44：15s/句对齐10s/8s 无赢家）、节奏起伏（z45：--rate-gain 2 与现役 15:17 打平）。
- 🔴 z43 一票定案：**解码参数侧扫到头**；步64 另 CER 女 0.201 退化。
- 🔴 参考音频判据：干净>长度；**5~10s 甜点，15s 天花板**；句子要完整；情绪中性。女声现役参考 = `00b6a45ad6aa@366.5s`，
  **起点前 150ms 能量=段内 90.9% → 切在句中**（已知缺陷，但选段轴已死）。`EL_male.wav` 不在案例库，
  campplus 确认与 `1a6c20a60481` 同一人（0.9624/0.9147）。
- 🔴 稳定规律：**输出停顿密度 ≈ 参考停顿密度**；**raw 与成品停顿数几乎相同** → 停顿是模型直出，不是后处理吃掉。

## 关键结论（别重复踩）
- **控制题在 4 个听者上系统性失效**（差 1~2 档）→ 绝对分不可信，**只看组间相对排序**。
- 🔴 曾误报「男声停顿 -57% 是大缺口」：实为 3 vs 7 个 ≥0.3s 长停顿（差 1.86s/8min）→ 小样本放大，作废。
- 🔴 `--target-rate 0` 会被 `if a.target_rate else` 判为未设置 → 回默认 8.9；要"不变速"必须传**负值**。
- 🔴 `fix_chain --target-cps` 的 `fs=clamp(f*(r/mean),1.0,1.35)` 只放慢不加快 → 把长篇节奏往匀速压；
  `--rate-gain`/`--min-fs` 可放大起伏（默认 0/1.0=不变）。**短素材是单窗=空操作**。
- r59：OmniVoice 是**掩码扩散模型**（arXiv 2604.00688），非自回归。现役高频细节（4-12k/0.2-3k）比 EL 少 19%；
  `t_shift=1.0` 对韩语是灾难（CER 0.38/0.83）→ **必须保持 0.1**；`position_temperature` 保持 5.0。
- r55：qwen 侧 `_get_clone_prompt` 整段 ref_text 会复读（CER 0.431），**语速体检抓不到，必须 ASR 复核**。

## 版本轴 & 发布
版本 `v<主>.<次>` / 批次 `z<N>`；真相源服务端 `VERSION`，唯一写入口 `tools/release_version.py`。
- `--bump` 要带 `v` 前缀（`--bump minor` 会报格式错）→ 用 `--set v4.5`。
- 只改 note 用 `--set <当前版本>`（不破 consistent）；`--items` 只改前缀，必须再 `cp` 到 `/mnt/tts/fb/items.json`。
- 反馈台 `/feedback`；票存 `/mnt/tts/fb/votes.jsonl`；题面/选项/note **不许出现身份词**，`ref:null`。

## 测量口径（不统一就作废）
click：峰值 0.70 归一判 |Δ|≥0.30，16k，测压完 mp3 后。音色 campplus 16k 多窗阈值 0.75，跑间噪声 ±0.005 男/±0.011 女。
L3 ASR 相似度跑间波动 ≈0.12 → <0.12 不算证据。cps 含空格；EL 锚 女 7.45/男 8.37。
链路 24k，最后 `aresample=44100` 压 mp3；**量 8k 以上必须 44.1k 成品**。EL 基线 `cases/<id>/merged.wav`（24k）。
🔴 量底噪/停顿电平**必须 sf.read 直读浮点**，ffmpeg 转 16bit 会压低下限 10dB。

## 常用坑
- 🔴 改 `app/` 后重启时间必须 > mtime；补丁自检 `compile()` + L1 实测。
- 🔴 `from app import xxx` 扁平导入启动即崩；faster_whisper `local_files_only=True` + ndarray + cpu/int8（无 libcublas）。
- 🔴 服务器 `app/` 是真相；`ssh yuyin`（用户 zheng，**UTC 时区**）；`/mnt/tts/data` 等只有 zheng 能写，
  `/etc/systemd`、`systemctl` 必须 sudo；`/mnt/tts/data/tone-ab` 属主 root。
- 🔴 判活别用 `/health`（不存在，404 是正常的）。
- 数字零填充必须 `sf.write(..., subtype="FLOAT")`；`fix_chain.py` 是 subprocess，改完即生效。

## 工具索引
`sync_from_server.sh`（收工前必跑）/ `release_version.py` / `clickfix53b` / `measure53` / `spkcmp53` / `asrseg53` /
`audit53` / `numcheck53` / `l3check53`；本地 `/tmp`：`droppos.py`（断点定位）/ `r65grid2.py`（填充电平网格）/
`envshape.py`（边界包络）/ `ratevar.py`（节奏起伏）/ `mk_r65.py`（分块 A/B）。

## 产品铁律：员工零调参
输标题+文案+选男女 → 等结果。emotion 恒 standard。TTS_SLOWDYN/BREATH_SYN/PEAK_FIT=1、LOUD_TOL=0.15、PROSO=0。

## 🔑 认证 / 账号掉线（2026-10-23 定位）
员工账号授权"过段时间就掉"的根因（非 bug，是设计）：
- **登录态存进程内存，不落盘**：`SESSIONS = {}`（`server.py:787`），`new_session` 写
  `SESSIONS[tok]={user, exp:now+SESSION_TTL}`（`:955-957`）；全代码只有读写和过期 pop，
  **无 save/load** → **服务一重启，全员立即掉线**（cookie 还在但服务端查不到 → 401）。
  本项目 qwen-tts 本就易重启（OOM/部署）→ 这是"突然掉"的主因。
- **SESSION_TTL = 7 天**（`server.py:96`）= 即使不重启，一周也自动过期（"定期掉"的原因）。
- 已排除多 worker 随机掉线：`ExecStart = python -u server.py`，**单进程单 worker**，SESSIONS 唯一。
- 校验：`require_auth`（`server.py:914-944`）取 cookie `tts_session` 或 `Authorization: Bearer`，
  命中 `SESSIONS` 且未过期方放行；过期项访问时才 pop（无后台定时清理，无害）。
- 修复方向（C 级）：A) session 落盘 `data/sessions.json`（启动 load / 登录存）；
  B) 改 secret 签名长效 token（JWT/安全 cookie，无状态）。临时缓解可把 SESSION_TTL 调长
  （但 96 行硬编码，改需动代码）。
