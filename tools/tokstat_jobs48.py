# -*- coding: utf-8 -*-
"""tokstat_jobs48.py —— 扫 **真实 job 原文**（不是 full.txt）里的拉丁/数字占比。

为什么必须看 job：案例库的 full.txt 疑似已做过数字归一化（33 条 13 万字符
里阿拉伯数字 0 个，财经文案不可能没有数字）。真正喂给 TTS 的是员工上传的
`job["text"]`，CER 也是拿它当参考文本。所以要看 job。

判据：CER 重掷阈值 0.35。若真实 job 里拉丁/数字占比仍是 0，
那「영문약어 假错」就是个不存在的靶子 —— 做了收益 0，且有双向映射误伤风险。
"""
import json
import os
import re
import sys

LAT = re.compile(r"[A-Za-z][A-Za-z0-9&\.\-]{0,11}")
DIG = re.compile(r"[0-9][0-9,\.]*")


def main():
    d = json.load(open("/mnt/tts/data/jobs.json", encoding="utf-8"))
    it = d.values() if isinstance(d, dict) else d
    rows = []
    for v in it:
        if not isinstance(v, dict):
            continue
        t = v.get("text") or ""
        if len(t) < 200:            # 忽略短测试
            continue
        n = max(len(t), 1)
        lat = LAT.findall(t)
        dig = DIG.findall(t)
        rows.append((v.get("title") or v.get("id", "")[:12],
                     v.get("owner"), n,
                     sum(len(x) for x in lat) / n,
                     sum(len(x) for x in dig) / n,
                     len(lat), len(dig),
                     sorted(set(lat))[:12], sorted(set(dig))[:12]))
    rows.sort(key=lambda r: -(r[3] + r[4]))
    print("job 数（≥200 字）= %d" % len(rows))
    print("%-22s %-8s %7s %8s %8s %6s %6s" %
          ("title", "owner", "字数", "拉丁%", "数字%", "拉数", "数数"))
    for t, o, n, lp, dp, ln, dn, ls, ds in rows[:25]:
        print("%-22s %-8s %7d %8.3f %8.3f %6d %6d"
              % (str(t)[:22], str(o)[:8], n, lp * 100, dp * 100, ln, dn))
        if ls:
            print("        拉丁样例: %s" % ls)
        if ds:
            print("        数字样例: %s" % ds)
    tot = [r[3] + r[4] for r in rows]
    if tot:
        print()
        print("拉丁+数字占比：最大 %.3f%%  均值 %.3f%%"
              % (max(tot) * 100, sum(tot) / len(tot) * 100))
        print("🔴 重掷阈值 35%%；最坏上界（每个 token 错成 3 倍长）%.3f%%"
              % (max(tot) * 3 * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
