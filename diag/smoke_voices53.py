# -*- coding: utf-8 -*-
"""smoke_voices53.py —— 冒烟：跑一个任务，确认服务写回后音色数仍是 9。

🔴 铁律（2026-10-19 踩过）：任务结束时（音色语速自校准）服务会把**内存音色表**
写回磁盘，把磁盘上后加的音色冲掉（当时 SOLA_female 7→6）。
→ 加完音色必须：重启 → 跑冒烟任务 → 再数一次。

用法: venv/bin/python -u /mnt/tts/diag/smoke_voices53.py
"""
import json
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")

from eval5 import login, submit_job, wait_job  # noqa: E402

V = "/mnt/tts/data/voices/voices.json"


def count():
    d = json.load(open(V))
    return len(d), sorted(d.keys())


def main():
    n0, k0 = count()
    print("任务前：%d 个 %s" % (n0, k0), flush=True)
    for name in ("elara", "yo han"):
        if name not in k0:
            print("❌ %s 不在表里，先修 voices.json" % name)
            return 1
    login()   # 🔴 鉴权是 cookie，必须先 Form 登录，否则提交会 401
    j = submit_job("안녕하세요. 오늘도 좋은 하루 되세요.", "elara",
                   "smoke_elara53")
    print("job =", j, flush=True)
    try:
        r = wait_job(j, timeout=180)
        print("结果 =", str(r)[:300], flush=True)
    except Exception as e:
        print("wait_job 异常（不影响音色表检查）：%s" % e, flush=True)
    n1, k1 = count()
    print("任务后：%d 个 %s" % (n1, k1), flush=True)
    if n1 == n0 and "elara" in k1 and "yo han" in k1:
        print("✅ 写回没冲掉新音色")
        return 0
    print("❌ 音色表被冲掉了（%d → %d）" % (n0, n1))
    return 1


if __name__ == "__main__":
    sys.exit(main())
