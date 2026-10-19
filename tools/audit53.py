# -*- coding: utf-8 -*-
"""audit53.py —— 服务端完整性体检（只读）。

检查项：
  1. voices.json 每个音色的 engine / gender 是否符合现役路由
  2. server.py 关键补丁是否都在（女声极简分支 / _gs_ref_pair / TTS_GS_REF_MAP）
  3. tools/ 下优化脚本是否齐备
  4. VERSION / release_version.py 版本凭证链是否完整
  5. /api/version consistent 是否 true

用法: python3 tools/audit53.py
"""
import json
import os
import re
import sys

ROOT = "/mnt/tts/projects/qwen3-tts"
VOICES = "/mnt/tts/data/voices/voices.json"

# 现役期望路由（2026-10-19 修复后）
EXPECT_ENGINE = {
    "EL_male": "omnivoice",
    "EL_female": "omnivoice",
    "yohan_gs10": "gptsovits",
    "yohan_gs_f8": "gptsovits",
    "SOLA_gs10": "gptsovits",
    "yohan_male": "qwen",
}
VALID = {"qwen", "cosyvoice3", "gptsovits", "voxcpm", "omnivoice"}

ok = True


def chk(cond, msg):
    global ok
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        ok = False


print("=== 1. voices.json ===")
try:
    v = json.load(open(VOICES, encoding="utf-8"))
except Exception as e:
    chk(False, "读取失败 %r" % e)
    v = None
if v is not None:
    if isinstance(v, dict):
        vs = [dict(x, name=k) for k, x in v.items()]
    else:
        vs = v
    chk(len(vs) >= 7, "音色数 = %d（2026-10-19 恢复后应为 7）" % len(vs))
    for x in vs:
        nm = x.get("name")
        eng = x.get("engine")
        chk(eng in VALID, "%s -> %s（合法引擎）" % (nm, eng))
        if nm in EXPECT_ENGINE:
            chk(eng == EXPECT_ENGINE[nm],
                "%s -> %s（期望 %s）" % (nm, eng, EXPECT_ENGINE[nm]))
    # 参考音频必须真的在盘上
    for x in vs:
        p = x.get("path")
        if p:
            chk(os.path.exists(p), "%s 参考音频存在 %s" % (x.get("name"), p))

print("=== 2. server.py 补丁 ===")
sp = os.path.join(ROOT, "app", "server.py")
src = open(sp, encoding="utf-8").read()
lines = src.splitlines()
# 找到极简后处理的开关行，往上找最近的 if 守卫
gi = next((i for i, l in enumerate(lines) if "--no-pitch" in l), None)
guard = ""
for l in reversed(lines[max(0, gi - 8):gi]):
    s = l.strip()
    if s.startswith("if ") or s.startswith("elif "):
        guard = s
        break
chk(guard.startswith('if _engine_name == "omnivoice":'),
    "极简分支守卫 = %r（不能带 and _g == \"m\"，否则女声漏掉）" % guard)
# 男声 cps 锚点 8.27 仍然必须存在（这是合法的性别分支，不是漏补丁）
chk('_cps = 8.27' in src, "男声 cps 锚点 8.27 仍在（这个性别分支是必需的）")
chk(src.count("_gs_ref_pair") >= 2,
    "_gs_ref_pair 定义+调用都在（GS 参考音频映射）")
chk("TTS_GS_REF_MAP" in src, "TTS_GS_REF_MAP 被消费")
chk("_run_fix_chain" in src and "_synth_gs" in src,
    "fix_chain 只在 vox/omni 分支；_synth_gs 不走后处理")
# 🔴 v3.2 起：保采样率窄带中频 EQ（patch_mideq53b.py）
# 🔴 v3.3 起：EQ + clickfix 扩到女声（patch_female53.py），阈值分性别
chk("_mideq53b_wrap" in src, "_mideq53b_wrap 定义+调用都在（窄带中频 EQ）")
chk("TTS_MID_EQ_MALE_DB" in src, "TTS_MID_EQ_MALE_DB 被消费（可关）")
chk("TTS_MID_EQ_FEMALE_DB" in src, "TTS_MID_EQ_FEMALE_DB 被消费（v3.3 女声 EQ）")
chk('"f": "TTS_MID_EQ_FEMALE_DB"' in src, "EQ 按性别选环境变量（m/f 分流）")
chk("CLICKFIX_FEMALE_TH" in src, "CLICKFIX_FEMALE_TH 已定义（v3.3 女声 clickfix）")
chk('elif _gd == "f":' in src, "clickfix 按性别分流（男 0.30 / 女 0.26）")
chk('"TTS_CLICKFIX_FEMALE_TH", "0.26"' in src,
    "女声 clickfix 默认阈值 0.26（pipe_f53 阈值扫描定）")
chk('log.info("mideq53b 完成' in src,
    "🔴 EQ 成功有 info 日志（此前成功路径零日志 = 审计盲区，"
    "前两轮补丁失效都发现不了）")
chk("_mideq53b_wrap(y2, sr2, _g, _engine_name)," in src,
    "EQ 插在 fix_chain 与 clickfix53b 之间（🔴 不能写成 *解包，会少收 "
    "gender/engine_name 两个参数导致整块回退）")
chk("*_mideq53b_wrap" not in src, "没有 *解包 误用（那会让 EQ+clickfix 全不跑）")
chk("_extra = len(_y) - len(audio)" in src and "_y[_o:_o + len(audio)]" in src,
    "FIR 输出变长用**居中裁剪**（🔴 用 != 判等会让 EQ 永远退回）")
chk('"firequalizer=gain_entry=\'entry(1000,0);' in src,
    "用 firequalizer（🔴 本机 equalizer 滤镜实测失效）")
# 补丁脚本本身要在（防回滚丢失）
for f in ("patch_clickfix53b.py", "patch_mideq53b.py", "patch_gsref50.py",
          "patch_female53.py", "patch_mideqlog53.py"):
    chk(os.path.exists(os.path.join(ROOT, "deploy", f)),
        "deploy/%s 存在（补丁可重打）" % f)
# 语法自检
try:
    compile(src, sp, "exec")
    chk(True, "server.py 语法通过")
except SyntaxError as e:
    chk(False, "server.py 语法错误 %r" % e)

print("=== 3. tools/ 优化脚本 ===")
for f in ("clickfix53.py", "tailclean53b.py", "tailclean53c.py",
          "ltasmatch53.py", "measure53.py", "spkcmp53.py",
          "pipeline53.py", "asrseg53.py", "fixchk53.py",
          "clickfix53b.py", "tailclean53b2.py", "tailclean53c2.py",
          "eqfix53b.py", "ltasmatch53b.py", "tailattr53.py",
          "cut_voice53b.py"):
    chk(os.path.exists(os.path.join(ROOT, "tools", f)), f)

print("=== 4. 版本凭证链 ===")
vp = os.path.join(ROOT, "VERSION")
chk(os.path.exists(vp), "VERSION 真相源存在")
if os.path.exists(vp):
    src_ver = open(vp, encoding="utf-8").read()
    m = re.search(r"^version\s*=\s*(\S+)", src_ver, re.M)
    mr = re.search(r"^round\s*=\s*(\S+)", src_ver, re.M)
    print("       当前版本 = %s   当前批次 = %s"
          % (m.group(1) if m else "?", mr.group(1) if mr else "?"))
    chk(bool(mr), "\U0001f534 VERSION 里标了 round（反馈批次轴）")
rv = os.path.join(ROOT, "tools", "release_version.py")
chk(os.path.exists(rv), "release_version.py 存在（唯一版本号写入口）")
if os.path.exists(rv):
    _rv = open(rv, encoding="utf-8").read()
    # \U0001f534 静态确认批次轴真的实现着，被回滚会立刻显形
    chk("new_round" in _rv, "\U0001f534 rewrite_items 支持同步批次（new_round）")
    chk("RE_ROUND" in _rv, "\U0001f534 release_version.py 能识别 z 批次段（RE_ROUND）")
_rl = os.path.join(ROOT, "app", "relver.py")
chk(os.path.exists(_rl), "app/relver.py 存在（凭证快照）")
if os.path.exists(_rl):
    _rls = open(_rl, encoding="utf-8").read()
    chk("round_ok" in _rls, "\U0001f534 relver.snapshot 校验批次（round_ok）")
    chk("round_consistent" in _rls, "\U0001f534 relver 输出 round_consistent 字段")

print("=== 5. /api/version ===")
try:
    import urllib.request
    j = json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/version",
                                         timeout=6))
    chk(j.get("consistent") is True, "/api/version consistent = %s"
        % j.get("consistent"))
    chk(j.get("version_consistent") is True, "/api/version version_consistent = %s"
        % j.get("version_consistent"))
    chk(j.get("round_consistent") is True,
        "\U0001f534 /api/version round_consistent = %s（false = 批次漂移："
        "题库 id 第二段和 VERSION 的 round 不一致）" % j.get("round_consistent"))
    _fb = j.get("fb") or {}
    chk(not _fb.get("round") or _fb.get("round") == j.get("round"),
        "题库自报批次 = %s，真相源 %s" % (_fb.get("round"), j.get("round")))
    chk(not j.get("fb", {}).get("missing"), "反馈台素材无缺失：%s"
        % j.get("fb", {}).get("missing"))
    print("       version=%s round=%s" % (j.get("version"), j.get("round")))
except Exception as e:
    chk(False, "无法读取 /api/version：%r" % e)

print()
print("总计: " + ("全部通过 ✅" if ok else "存在 FAIL ❌"))
sys.exit(0 if ok else 1)
