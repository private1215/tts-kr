# -*- coding: utf-8 -*-
"""l1_female53.py —— L1 验证女声 EQ + clickfix **真的在执行**。

## 为什么必须实测
audit53 只查「字符串在不在」，**查不到「没被执行」**。第53轮已经连中两招：
  ① `return _clickfix53b_wrap(*_mideq53b_wrap(...))` 少收参数 → TypeError
     → 整块回退 raw，EQ 和 clickfix 全都没跑，audit53 全绿；
  ② `if len(_y) != len(audio): return audio` 判等 → firequalizer 输出长 320 样本
     → 每次都退回。
→ 打补丁后**必须**跑一条真单，从日志里找执行证据，不能只信自检。

## 判据
1. journalctl 里出现 `clickfix53b 完成 th=0.26`（女声阈值）→ 女声 clickfix 真跑了；
2. 同一条单里还能看到 `th=0.30` 的男声记录（回归：男声没被改坏）；
3. 产物中频/click 与离线预期同向。

用法: venv/bin/python -u /mnt/tts/diag/l1_female53.py
"""
import glob
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from tools.eval5 import login, submit_job, wait_job  # noqa: E402

PY = "/mnt/tts/projects/qwen3-tts/venv/bin/python"
T = "/mnt/tts/projects/qwen3-tts/tools"
ELF = "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"
SHORT_F = "/mnt/tts/diag/short_f.txt"
SHORT_M = "/mnt/tts/diag/male_1500.txt"


def journal(since="6 min ago"):
    r = subprocess.run(
        ["sudo", "-n", "journalctl", "-u", "qwen-tts", "--since", since,
         "--no-pager"], capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


def find_out(title):
    hits = sorted(glob.glob("/mnt/tts/output/" + title + "*"),
                  key=os.path.getmtime)
    return hits[-1] if hits else None


def one(tag, text_path, voice, title, chars):
    txt = open(text_path, encoding="utf-8").read().strip()
    print("\n===== %s（%s，%d 字）=====" % (tag, voice, len(txt)), flush=True)
    j = submit_job(txt, voice, title)
    s = wait_job(j.get("id"), timeout 	= 3600)
    print("status=%s engine=%s cps=%s" % (
        s.get("status"), s.get("engine"),
        (s.get("stats") or {}).get("cps")), flush=True)
    d = find_out(title)
    if not d:
        print("❌ 找不到产物 %s" % title)
        return None
    fs = sorted(glob.glob(os.path.join(d, "*.mp3")))
    if not fs:
        print("❌ 目录里没有 mp3")
        return None
    mp3 = fs[-1]
    print("产物：%s" % mp3, flush=True)
    subprocess.run([PY, "-u", T + "/measure53.py", mp3, ELF, str(chars)])
    return mp3


def main():
    login()
    t0 = time.strftime("%Y-%m-%d %H:%M:%S")
    one("女声 L1", SHORT_F, "EL_female", "l1f53", 300)
    one("男声 L1（回归）", SHORT_M, "EL_male", "l1m53", 1500)

    log = journal("10 min ago")
    print("\n===== 执行证据（journalctl）=====", flush=True)
    pats = [r"clickfix53b 完成 th=(\S+)",
            r"mideq53b|firequalizer|EQ",
            r"clickfix53b 失败", r"clickfix53b 异常",
            r"clickfix53b 不存在", r"clickfix53b 产物过短"]
    for p in pats:
        hits = re.findall(p, log)
        print("  %-34s → %d 次 %s" % (p, len(hits), hits[:6]))
    ths = re.findall(r"clickfix53b 完成 th=(\S+)", log)
    print("\n  ✅ 判据1 女声 th=0.26 出现：%s"
          % ("是" if "0.26" in ths else "❌ 否"))
    print("  ✅ 判据2 男声 th=0.30 出现：%s"
          % ("是" if "0.30" in ths else "❌ 否"))
    bad = re.findall(r"clickfix53b (?:失败|异常|产物过短)", log)
    print("  ✅ 判据3 无失败/异常/过短：%s"
          % ("是" if not bad else "❌ %s" % bad[:3]))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
