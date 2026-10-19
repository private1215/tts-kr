# -*- coding: utf-8 -*-
"""给 audit53.py 增加 v3.3（女声接 EQ+clickfix）的检查项。

只做**追加**，不改动原有任何检查，避免把在用的检查弄坏。
用法: python3 tools/audit53_female.py --apply
"""
import io
import os
import shutil
import sys
import time

TARGET = "/mnt/tts/projects/qwen3-tts/tools/audit53.py"

OLD_BLOCK = """# \U0001f534 v3.2 起：保采样率窄带中频 EQ（patch_mideq53b.py，只 omni+男声）
chk("_mideq53b_wrap" in src, "_mideq53b_wrap 定义+调用都在（窄带中频 EQ）")
chk("TTS_MID_EQ_MALE_DB" in src, "TTS_MID_EQ_MALE_DB 被消费（可关）")"""

NEW_BLOCK = """# \U0001f534 v3.2 起：保采样率窄带中频 EQ（patch_mideq53b.py）
# \U0001f534 v3.3 起：EQ + clickfix 扩到女声（patch_female53.py），阈值分性别
chk("_mideq53b_wrap" in src, "_mideq53b_wrap 定义+调用都在（窄带中频 EQ）")
chk("TTS_MID_EQ_MALE_DB" in src, "TTS_MID_EQ_MALE_DB 被消费（可关）")
chk("TTS_MID_EQ_FEMALE_DB" in src, "TTS_MID_EQ_FEMALE_DB 被消费（v3.3 女声 EQ）")
chk('"f": "TTS_MID_EQ_FEMALE_DB"' in src, "EQ 按性别选环境变量（m/f 分流）")
chk("CLICKFIX_FEMALE_TH" in src, "CLICKFIX_FEMALE_TH 已定义（v3.3 女声 clickfix）")
chk('elif _gd == "f":' in src, "clickfix 按性别分流（男 0.30 / 女 0.26）")
chk('"TTS_CLICKFIX_FEMALE_TH", "0.26"' in src,
    "女声 clickfix 默认阈值 0.26（pipe_f53 阈值扫描定）")
chk('log.info("mideq53b 完成' in src,
    "\U0001f534 EQ 成功有 info 日志（此前成功路径零日志 = 审计盲区，"
    "前两轮补丁失效都发现不了）")"""

OLD_DEPLOY = """for f in ("patch_clickfix53b.py", "patch_mideq53b.py", "patch_gsref50.py"):"""

NEW_DEPLOY = """for f in ("patch_clickfix53b.py", "patch_mideq53b.py", "patch_gsref50.py",
          "patch_female53.py", "patch_mideqlog53.py"):"""


def main():
    apply = "--apply" in sys.argv
    if not os.path.exists(TARGET):
        raise SystemExit("找不到 %s" % TARGET)
    src = io.open(TARGET, encoding="utf-8").read()
    if "TTS_MID_EQ_FEMALE_DB" in src:
        print("已打过，跳过")
        return 0
    for old, new, what in ((OLD_BLOCK, NEW_BLOCK, "EQ/clickfix 检查项"),
                           (OLD_DEPLOY, NEW_DEPLOY, "deploy 补丁清单")):
        if src.count(old) != 1:
            raise SystemExit("❌ %s 匹配 %d 次（要 1 次）" % (what, src.count(old)))
        src = src.replace(old, new)
        print("替换 %s" % what)
    compile(src, TARGET, "exec")
    print("compile() 通过")
    if not apply:
        print("（未加 --apply）")
        return 0
    bak = "%s.bak-before-auditfemale-%s" % (TARGET, time.strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(TARGET, bak)
    with io.open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)
    compile(io.open(TARGET, encoding="utf-8").read(), TARGET, "exec")
    print("✅ 写入 %s  备份 %s" % (TARGET, bak))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
