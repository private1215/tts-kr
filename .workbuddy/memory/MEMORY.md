# tts-kr 速查索引（2026-10-22，**v4.4 / z45 已回收：9:9 打平，生产仍停在 v4.1 链路**）

> 细节版 `.workbuddy/memory/archive-detail.md` + 当日日志；改链路前先读 `项目规范.md`，冲突以规范为准。

## 🔴🔴 协作约定（用户 2026-10-22 明确要求）
1. **非必要不要频繁发反馈批次**。每批只有 1~2 票、且控制题常失效（±1 档），
   连续发批次等于消耗员工耐心换低置信度结论。发之前先问：这一批能定什么案？定不了就别发。
2. **代码改动必须及时提交入库**。服务端 `/mnt/tts/projects/qwen3-tts` **不是 git 仓库**，
   改完不主动同步就永远丢失（2026-10-22 才发现 25 个文件与仓库不一致、`omnivoice/` 整个未入库）。
   → 收工前：比对 `md5sum app/*.py tools/*.py`（服务端 vs 本地）→ scp 回本地 → 提交 → push（重试 ≤4 次）。
   ✅ **已固化为 `tools/sync_from_server.sh`（幂等，比对 205 个文件）+ 项目规范 S7.1**（2026-10-22）。
   ⚠️ 只按文件 scp，**绝不整目录 rsync**；systemd drop-in / 参考表要手动归档到 `deploy/`。
3. 汇报要分清「**生产真的改了**」和「**只是发了批次**」，别把发批次说成优化。

## 目标与验收
替代 EL：同文案我们 vs EL 员工听不出差别。员工零调参，**验收只看 L3 长篇**。
- 🔴 物理差 ≠ 可闻差（z44：13dB 9票仅1票；z37：8-10k ×58 不可闻）→ 定根因前先出裁决题。
- 🔴 客观指标只是体检；换引擎/调参只能盲听拍板（z20 教训）。
- 🔴 片段口径与全篇口径可相反 → 只用 L3 下结论。
- 出题规范：A/B「有没有」题四选项全给；控制题失效信号 = 在相同音频间瞎挑（选 A/B）。
- 🔴 **题里不放人声标杆**（用户 2026-10-22 明确：员工听 EL 没意义，只听本项目方案）；
  校准改用**重复控制题**（同段音频出现两次）。代价：只能做批内排序，不能跨批校绝对分。
- 🔴 题面文案必须用案例原始 `seg*.txt`，不能用自己的 ASR 输出。

## 🔴 已排除清单（2026-10-21 为止，别再测）
全部与 EL 同口径持平或已修：16k 带宽（已修 v3.7，但不可闻）、数字零（已修 v3.6，非主因）、
韵律（F0std/p90-p10/能量包络）、HNR、LTAS 1/6倍频程、**jitter/shimmer**（男1.40/1.35%、女0.92/0.91%，Praat）、
**音节级时长**（syl mean/cv/p10-p90 全平，medium-whisper 词时间戳）、**11.2k 窄带电磁音**（EL 也有，
el_m 在 10k 有突起 25.7dB 的峰，比我们 4.9dB 还大）、**omni ref_text 配比**（126/112 是含空格口径，
去空格 95/86 vs ASR 89/83，差 3.6~6.7% = whisper 正常漏字，**配比是准的**）、15s 接缝深静音（男 0 处落 14~16.5s 带）、
`language` 参数（空操作）、`normalize_text`（对韩语有害）、CosyVoice3 SFT、omni 换声码器（codec LM 封死）、
同说话人 LoRA 加自然度（学的是音色不是自然度）。
→ **物理层已测无可测，差距只在人耳。有效手段：盲听定位（z39）+ 解码参数（z40）。**

## 现役链路 v3.9（= v3.8 链 + per-voice 参考）
qwen :8000（后处理只在这）/ cosy :8100 / gs :8200 / omni :8300 / vox :8400。男女都走 omnivoice。
omni 后处理（`--no-pitch --no-clean --no-adaptive-hf --eq anull`）+ 链尾零填充 + 全链 24k。
钩子顺序 `_mideq53b_wrap`→`_clickfix53b_wrap` 不能反。`_mideq53b` 对 omni 默认 +2dB@1.8k（男女同值）。
已上线：v3.1 clickfix / v3.2 EQ / v3.3 女声接管 / v3.4-3.5 headfix+numcheck / v3.6 零填充 / v3.7 24k / v3.8 批次 / **v3.9 per-voice 参考**。
🔴 **omni 完全绕过文本前端**：`server.py:1375` `NEW_ENGINES → chunks=[job["text"]]`，不走 `pipeline.build_segments()`
→ **不调 `normalize_korean()`，`TTS_KO_NUM=1` 的 konum 对 omni 零作用**。暂无需修（数字读法实测正确），
但 konum 本身也不完整（`67`/`11시 5분`/`1,830억원` 都不转）。
🔴 **数字读法实测正确**：`67→육십칠`、`11시 5분→열한시 오분`、`1,240만주→천이백사십만주`、`1,830억원→천팔백삼십억원`。
  但 **生产 ASR 复核查不出读错数字**——whisper 把听到的数字写成阿拉伯数字，CER 恒过。
  要看真实读法必须用 `initial_prompt` 强制 whisper 输出韩文。

## 🔴🔴 z38 结论（2026-10-22 回收，两票）
**音频指纹实证**（`who55.py` 滑动相关全 1.000）：m1=qwen(xvec-only)/m2=EL男/m3=omni、f1=omni/f2=qwen(xvec)/f3=EL女、ctrl=m2。
🔴 **`/tmp/mk_z38.log` 尾部映射是旧跑残留、与 `z38_map.json` 矛盾 → 批次映射只信音频指纹，别信日志。**

| | m1 qwen | m2 EL | m3 omni | f1 omni | f2 qwen | f3 EL | ctrl |
|---|---|---|---|---|---|---|---|
| admin | **A** | A | B | B | B | A | A |
| 디아 | B | B | B | B | B | A | A |

- EL 9 票 8A；**omni 0A/4B；qwen-xvec 1A/3B**（男声那个 A 是我们唯一的 A）→ qwen-xvec 男声值得再验。
- 디아 控制题自相矛盾（同一段 EL 男声 m2=B / ctrl=A）→ ±1 档噪声 + 位置效应；admin 自洽。
- 🔴🔴 **员工听的不是现役版**：omni.log `07:47 synth voice=male 154字`（旧格式、无括号解析）= 旧代码 + 10s 写死 prompt；
  新代码记录 `voice=EL_male(EL_male)`（09:11 起），per-voice 是 omni-tts **09:30 重启**后才生效。
  **日志格式判别新旧代码：`voice=X(Y)`=新，`voice=X`=旧。**
- 员工文字：甲「没有啊额音了」= filler 音已解决（真实进步），「有几个是机械音感觉」；
  乙「1-5 像AI 6-7 可以」**不可采信**——#2 是 EL 真人标杆、#7 是同一段音频却判相反。

## 🔴🔴 v4.4 / z45 在收票：长篇节奏（链路 num_step=48、参考音频全不动）
**z44 已回收（디아，控制题又失效 p01=B/p09=C）**：a现役15s=4 / t句对齐10s=4 / s短8s=3 / **r句对齐10s=2（男女双C）**
→ **参考音频选段这条轴也扫完了**，无一超过现役（r 的参考文本含数字+问句且 ASR 带错字，两头不干净）。
🔴🔴 **跨批次 A 票统计：z42（L3 长篇 30s 切片）A 占比 50%，而 z40/41/43/44 的 25s 短片段只有 0~25%**
→ **短片段批次系统性低估，而验收只看 L3 → 后续批次一律走 L3 长篇切片**。
🔴🔴 **长篇节奏被压成匀速（本轮最实的客观缺口）**：逐窗速率 CV/极差
现役 男 2.1%/7.7%、女 2.5%/6.7% ｜ raw不变速 男 2.6%/8.1% ｜ **EL 真人 男 4.2%/15.5%、女 3.9%/13.2%**
→ 真人起伏是现役两倍（联网：匀速 metronomic 是合成语音第一破绽）；
  但 raw 也只有 2.6% → **光去掉变速补不回，必须主动放大**。
`fix_chain` 新增 `--rate-gain g`（默认 **0.0**=现状，`fs_i=f*(r/mean)^(1-g)`）与 `--min-fs`（默认 **1.0**=只放慢）；
**Z = --rate-gain 2 --min-fs 0.9** → 男 CV 2.1→3.2%、女 2.5→4.6%（贴到/略超 EL）。
z45 = Y(现役) vs Z(起伏x2)，同一长稿 30s 起 / 210s 起两窗 × 2 性别 = 8 + dup = 9；
**Y 占 p02/p03/p06/p07（与 z42 的 B/步48 同题位）→ 可跨批直比**。
回滚 = 用 `.bak-before-rategain64` 覆盖 `app/fix_chain.py`（即时生效，无需重启）。
⚠️ 坑：`--target-rate 0` 会被判为未设置而回落默认 8.9，**拿不到"不变速"**；要传负值。

## 🔴🔴 v4.4 / z45 已回收（9:9 打平 → 不上线，生产仍是 v4.1 链路）
z45 票（디아 1 票，控制题又失效 p01=B/p09=A）：Y(现役)=9 / Z(起伏x2)=9 **完全打平** → 节奏放大不可闻，不上线。
✅ **本批 8 题 C 票 = 0**（全 B/A）；z43/z44 短片段批次 C 票占一半 → 长篇素材被评得明显更好。
⚠️ 我曾把「男声停顿 −57%」误报成大缺口：**实际是 3 个 vs 7 个 ≥0.3s 长停顿（差 1.86s/8min）**，
短停顿(0.1-0.3s) 我们 150 vs EL 140 反而略多 → 小样本放大的相对差，别再当结论用。
✅ 稳定规律：**输出停顿密度 ≈ 参考停顿密度**（男 参考100.1→输出88.8/分，EL 83.8；女 31.8→35.0，EL 31.2）；
**raw 与成品停顿数几乎相同**（男 0.386 vs 0.372）→ **停顿是模型直出，不是后处理吃掉的**。
🔴 **阶段结论：解码参数(z40/41/43)、参考选段(z44)、节奏起伏(z45) 三条轴全被耳朵判平或判死；
已排除清单之外没找到新的可闻杠杆。生产保持 v4.1 不动。** 用户要求「非必要不频繁发批」→ 停发。
若要再推进，优先**补听者**（z45 素材已在，再请一人听即可破 9:9），不要新造素材。

## 🔴🔴 v4.3 / z44 已回收（参考轴无赢家）
**z43 一票定案**（디아，控制题自洽 p01=B/p09=B）：a=步48 / s=步64 / t=步48+ct0.3 / u=步64+ct0.3
**四候选全部 3 分打平**，无一超过现役 → **解码参数侧扫到头**（步64 另有 CER 女声 0.201 退化 + 男声唯一 C）。
**参考音频选段判据（联网两轮核对）**：① 干净 > 长度（参考里的底噪会**逐句复制**）；
② **句子必须完整，不能切在词中/句尾戛然而止**（否则尾语调复制到每一句）；
③ **5–10s 甜点，15s 是天花板**（MOSS-TTS：15s 后 artifacts creep in）；④ 多句略优；⑤ 采样率无关。
🔴 **已查实的现役缺陷**：女声参考 = 案例 00b6a45ad6aa @366.5s（非 252s），
**起点前 150ms 能量 = 段内 90.9% → 切在句中**；长度 15s 正好卡天花板。
男声 `EL_male.wav` **不在案例库**（相关 0.03），但已用 campplus 确认与 1a6c20a60481 同一人（0.9624/0.9147）。
**z44 候选**：a=现役15s(跨批锚) / r=句对齐10s / s=短8s / t=另一处句对齐10s；4×2=8 + dup(a_m)=9，
a_m/a_f 仍在 p01/p05 可跨批直比。候选注册为**新 voice key**，现役 EL_male/EL_female **没动**。
客观预筛坐实的机制：**参考段韵律动态会 1:1 传到输出**（男 116→115 / 158→132 / 159→143）；
但 HF 细节三候选互有高低、男女反向 → **噪声，不能拍板**。
🔴 **方法论缺口（待处理）**：生产对 omni 走 `--target-cps`（男 8.27 / 女 7.57，按性别），
`fs=clamp(f*(r/mean_r),1.0,1.35)` 只放慢不加快、按窗分配 → 客观上把节奏往匀速压。
**短素材是单窗 → fs 全 1.0 = 空操作**（实测男 f=0.975 / 女 f=0.953），所以 z40~z43 结论仍有效；
🔴 **但 L3 长篇是多窗会被真的变速，而验收只看 L3** → 盲区。z42 是唯一长篇批次。
长篇若仍被判「机械」，优先查这里的 atempo。

## 🔴🔴 v4.2 / z43（链路仍 num_step=48，未变）
**z42 一票定案**（이민정）：A(base 步32) 合计 7（1个A） vs **B(步48) 合计 10（3个A）**，
且**控制题首次自洽**（p01=A/p09=A）→ 步48 坐实，不再回退。디아 未投 z42。
**z43 候选**：a=步48(现役/跨批锚) / s=步64 / t=步48+class_temperature0.3 / u=步64+ct0.3；
a_m/a_f 放在 z41 里 s_m/s_f 的题位（p01/p05）→ 与 z41 的步48 成绩可跨批直比。
🔴 **联网核对（r62）**：社区文档 num_step 4~64、**64=最佳质量**；`denoise` 建议 True（与我们盲听一致）；
**参考音频建议 3~15s → 「加到 30s」作废**。DiSTAR：离散解码「贪心=最大稳定性，**随机采样=增多样性**」，
韵律研究「**均值分布陷阱**」=贪心导致过平滑发闷 → **ct 从 0 抬起对症**。
🔴 反向警告：解码顺序随机性越大 MCD 变好但 **UTMOS 自然度变差** → `position_temperature` 保持 5.0 别上调；
`layer_penalty_factor` 不用动。
CER 体检：步48 与 base32 持平（男0.162/女0.175）；**步64 女声 0.201（+0.026，退化风险）**；
步48+ct0.3 中性偏优（男0.169/女0.169）。步64 仍进盲听由耳朵裁决。

## 🔴🔴 v4.1 已上线：num_step 32 → 48（2026-10-22 13:32）
drop-in `/etc/systemd/system/omni-tts.service.d/20-steps48.conf`（`TTS_OMNI_STEPS=48`）；
回滚 = 删文件 + daemon-reload + restart。日志 `gen={'num_step': 48}` 可验。代价 RTF +40%（L3 223→262s）。
**依据 = z41 两票**（이민정/디아）：s(步48) 合计 9 > c(cfg30) 8 > a(base) 7 > **p(关后处理) 5**；
p 组男声两人全 C → **淘汰**（且证明「后处理削静音」不是机械音成因）。
L3 同文本 A/B 全链实测：男 cps 8.22→8.20 / 音色 0.9681→0.9668，女 7.53→7.47 / 0.9504→0.9475
→ **客观等价无回归**（差在噪声带内）。
🔴 **控制题在三个听者上全部失效**（admin/이민정/디아，差 1~2 档）→ **系统性锚定/位置效应**，
绝对分不可信，**只看组间相对排序**。新长篇也没有 L3pv55b 的「第1窗独快」，佐证那是单件现象。

## 🔴🔴 v4.0 / z41 已发布（2026-10-22 12:05，**链路参数未变**）
v4.0 = v3.9 链路 + z41 批次；bump 版本只为题面 `ver` 对得上服务端（否则 consistent=false）。
**z41 候选**：a=base / s=num_step=48（扩散步数直击过平滑）/ p=postprocess_output=False（保自然静音）/
c=cfg3.0（**跨批锚**）。顺序 s_m, p_f, a_m, c_f, s_f, a_f, c_m, p_m, dup(a_m)；答案 `_z41/answer.json`。
🔴 **a_m/a_f/c_f/c_m 与 z40 同位（p03/p06/p04/p07）→ 4 题可跨批直比**，弥补无 EL 锚点只能批内排序。
客观体检（6 候选×男女 hf）全落 0.0145~0.0175 = 跑间噪声内 → **分辨不出，只能盲听拍板**。
🔴 **per-voice 开关在哪**：`TTS_OMNI_PERVOICE=1` 在 **qwen-tts** 进程（查 omni 进程查不到会误判失效）；
omni 侧靠 `/mnt/tts/data/voices/omni_prompts.json`（默认路径、/voices 写回、重启不丢，**不需要 drop-in**）。
当前参考 EL_male 14.98s / EL_female 15.08s @24k。omni 跑带解码覆盖的 omni_server.py（日志 `gen=` 可验），
全留空 = v3.9 行为。

## 反馈批次状态（v4.3 已发布 / z44 在收票）
- **v3.9 已发布**（10-22）：per-voice omni 参考（P0）+ 批次 z39，`/api/version` consistent=true。
  P0 内容：`TTS_OMNI_PROMPTS` + `_resolve_voice()`（先判 female）+ `TTS_OMNI_PERVOICE=1`（drop-in `pervoice55.conf`）。
  音色余弦 男 0.9518→0.968/0.970、女 0.9195→0.955/0.951。驳回：删 drop-in + daemon-reload + 重启。
- **z39 已备好但被 z40 顶掉（未回收）**：14 题 10s 片段定位盲听（男6+女6 连续切片 60~120s + EL 校准 + dup 控制=m4），
  素材取自 **09:45 的 per-voice 现役产物 L3pv55b**。需要时可 `release_version.py --round z39 --items items_z39.json` 装回。
- z37 已回收：我们 4 段全「明显机器」、EL 全「像真人」；24k 改动不可闻，方向封存。

## r55 克隆参考文本泄漏（qwen 侧，未上线）
`_get_clone_prompt` 整段 ref_text → 复读（男 CER 0.431）。**xvec-only 0.083 但音色 −0.08**。
甜点 = 配比 7.3 字/秒（男 110 字）。语速体检抓不到复读，必须 ASR 复核。P2-b 已修缓存 key（指纹含音频+ref_text）。

## 🔴🔴 r59：OmniVoice 是**掩码扩散模型**（不是自回归 codec LM）
论文 arXiv 2604.00688 *OmniVoice: Towards Omnilingual Zero-Shot TTS with Diffusion Language Models*
→ 非自回归、全码本随机 masking。之前「codec LM 封死、只能换引擎」要收窄：**解码参数全可调，而现役一直用默认**。
现役默认：num_step=32 / cfg=2.0 / t_shift=0.1 / denoise=True / class_temperature=0.0 / postprocess=True。
- `denoise` = 在 style 插 `<|denoise|>` token（`omnivoice.py:1223`），**不是滤波器**；参考是干净录音却让模型去噪 → 过度平滑。
- `class_temperature=0.0` → `log_probs.argmax()`（`:1448`）**纯贪心**；>0 才走 top-k(0.1)+Gumbel。
🔴 **现役高频细节（4-12k/0.2-3k）比 EL 少 19%**（0.0142 vs EL 0.0176）=「闷/机械」的可测对应量。
  cfg=3.0 → 0.0173（−2%，回到 EL）；denoise_off → 0.0153；ct06+dnoff → 0.0166。
🔴 **t_shift=1.0 对韩语是灾难**（CER 男 0.383 / 女 0.831）→ **t_shift 必须保持 0.1**，别照搬社区 ComfyUI 默认值
  （那个包默认 cfg 3.0 / t_shift 1.0 / denoise off，一半对一半错）。
- **服务端已支持解码参数覆盖**（备份 `.bak-before-genpar55`）：优先级 请求体 `gen` >
  环境 `TTS_OMNI_DENOISE/CFG/CLASS_T/STEPS/TSHIFT` > 默认；**全留空 = 行为与 v3.9 一致**。日志加 `gen=` 字段。
  L1 已验（默认 `gen=-`、带参传 dict、两次 md5 不同）。上线 = drop-in conf + daemon-reload + 重启 omni-tts。
- **z40 已安装**（10-22，替代 z39）：**按用户要求不含人声标杆**——员工听 EL 没意义，只比本项目方案。
  9 题 = 4 候选（a=现役base / b=denoise_off / c=cfg30 / d=ct06+dnoff）× 男女 + dup 控制（=a_m，排第 9）。
  顺序 d_m, b_f, a_m, c_f, d_f, a_f, c_m, b_m, dup。全走生产路径（/synth per-voice + fix_chain），
  −20 LUFS、23s(男)/26s(女)、md5 已验（dup==p03，其余唯一）。答案 `/mnt/tts/diag/_z40/answer.json`（不可给员工）。
  三选项与 z38 同尺度 → 可直接和 z38 里现役的 B 分对比。判读：有候选到 A → 上线；全 B → 回到 codec 层。
  ⚠️ 无 EL 锚点意味着本批**只能做内部排序**，跨批校准靠 dup 控制题。

## 🔴🔴 r60：z40 首票 + 两个症状定位（2026-10-22）
**首票**（admin 02:51，控制题失效 p03=A/dup=B → ±1 档噪声降权）：
男 a=A c=A d=B b=C；女 a=B b=B c=C d=C。→ **base 是唯一男女都 ≥B 的配置**；cfg=3.0 男 A 女 C；
**b 组（denoise_off）出局**。行动：**不改参数**，z40 还在收票，**别装新批次顶掉**。
- **「说着说着变慢」= 单件现象，非系统**：L3pv55b-male 第 1 窗独快 ~6%（音节核 8.97 vs 后段 8.4，之后贴锚），
  6 件历史 L3 全平稳，EL 自身波动 ±8%。文本难度已排除（early/late 独立生成同速）。
  ASR 口径的「单调下滑 −9%」受数字漏字干扰不可信。🔴 曾疑 fix_chain「只放慢」放大下滑——
  同参数跑新 raw 产物平稳，**假设推翻，fix_chain 不改**。观察点：下件 L3 生成后 syl_rates 查第 1 窗。
- **「第 8 题卡音」= b_m**：click/频谱突变/能量凹陷/词级对齐/音节重复**五种口径全测不到**，
  但票面 b_m=C+单独点名卡音 → 互证 denoise_off 有可闻瑕疵，**不再追，b 组已死**。
- 工具：`/tmp/sylrate55.py`（音节核逐窗，import fix_chain.syl_rates）、`/tmp/pos55.py`、`/tmp/rawlong55.py`。
- 二期优化点：`postprocess_output=False`（去静音会削自然停顿）、参考音频 15s→30s。

## omni LoRA（跑通，不上线）
dev loss 150 步 3.7782 见顶 → 300 步 3.9012 过拟合。campplus：男无效、女 +0.010（噪声 ±0.005/±0.011）。
scale 默认 2.0 最好。**2 个说话人 → 学音色不学自然度**；要自然度必须换多样说话人语料。
坑：数据 id 唯一+seg 序号；`CUDA_VISIBLE_DEVICES=0`=3090；batch_tokens=2048/max_bs=4/accum=8；
认 LoRA 层用 `hasattr(mod,'lora_A')`。checkpoint-150/300 各 2.7G 待清理。

## 版本轴 & 发布
版本 `v<主>.<次>` / 批次 `z<N>`；真相源服务器 `VERSION`，唯一写入口 `tools/release_version.py`。
`--items` 只改前缀不安装 → 必须再 `cp` 到 `items.json`。反馈台 `/feedback`；`{"items":[…]}`；
**题面/选项/dim/note 不许出现身份词**。票存 `/mnt/tts/fb/votes.jsonl`。

## 测量口径（不统一就作废）
click：峰值 0.70 归一判 `|Δ|≥0.30`，16k，测压完 mp3 后。音色 campplus 16k 多窗阈值 0.75，跑间噪声 ±0.005 男/±0.011 女。
**L3 ASR 相似度跑间波动 ≈0.12（omni 无 seed）→ <0.12 不算证据**。cps 含空格；EL 锚 女 7.45/男 8.37。
链路 24k，最后 `aresample=44100` 压 mp3；**量 8k 以上必须 44.1k 成品**。EL 基线 `cases/<id>/merged.wav`（24k）。
跨容器比高频先统一容器。EL 运行间方差 click 2.56×/杂音 5.6×。

## 常用坑
- 🔴🔴 改 `app/` 模块后重启时间必须 > mtime；补丁自检 `compile()` + L1 实测。
- 🔴🔴 `from app import xxx` 扁平导入启动即崩；faster_whisper `local_files_only=True` + ndarray。
- 🔴 服务器 `app/` 是真相，按文件 scp 永不整目录 rsync；`ssh yuyin`（UTC 时区）。
- 🔴 `%` 优先于 `+`；相邻字面量后紧跟 `(...)` 是函数调用。
- 数字零填充必须 `sf.write(..., subtype="FLOAT")`；`fix_chain.py` 是 subprocess 改完即生效。

## 工具索引
`clickfix53b`/`measure53`/`spkcmp53`/`asrseg53`/`audit53`/`release_version`/`numcheck53`/`spkclus53`/
`fback53d`/`l3check53`；本轮新增 `/tmp/seam55.py`（静音块统计）、`/tmp/deep55.py`（深静音块）、
`/tmp/syl55.py`（音节时长）、`/tmp/jit55.py`（Praat jitter/shimmer）、`/tmp/mkz39.py`（z39 切片）。

## 产品铁律：员工零调参
输标题+文案+选男女 → 等结果。emotion 恒 standard。开 TTS_SLOWDYN/BREATH_SYN/PEAK_FIT=1、LOUD_TOL=0.15、PROSO=0。
