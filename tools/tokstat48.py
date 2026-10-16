# -*- coding: utf-8 -*-
"""tokstat48.py —— 统计原文里「拉丁字母串」和「数字串」的占比，
用来判断 CER 里的 영문약어 假错**最多**能占多少（上界）。

思路：不转写也能定上界。若原文里拉丁字母总共只有 0.5%，
那就算 ASR 把每个字母都念成三音节韩语，假错也不可能超过这个量级，
离 0.35 的重掷阈值差得远 → 这个优化就不值得做。

用法： python tokstat48.py --job <id>   或   --txt <file>
"""
import argparse
import collections
import json
import os
import re
import sys
import urllib.request

LAT = re.compile(r"[A-Za-z][A-Za-z0-9&\.\-]{0,11}")
DIG = re.compile(r"[0-9][0-9,\.]*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="")
    ap.add_argument("--txt", default="")
    a = ap.parse_args()
    if a.job:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8000/api/jobs/" + a.job,
            headers={"Authorization": "Bearer " + os.environ.get(
                "TTS_DEBUG_TOKEN", "smoke-8f3a2c91")}), timeout=60).read())
        t = r.get("text") or ""
    else:
        t = open(a.txt, encoding="utf-8").read()
    n = max(len(t), 1)
    lat = LAT.findall(t)
    dig = DIG.findall(t)
    lc = sum(len(x) for x in lat)
    dc = sum(len(x) for x in dig)
    print("总字符 %d" % len(t))
    print("拉丁 token %d 个，共 %d 字符，占原文 %.4f（%.2f%%）"
          % (len(lat), lc, lc / n, lc / n * 100))
    print("  最常见的：%s" % collections.Counter(lat).most_common(25))
    print("数字 token %d 个，共 %d 字符，占原文 %.4f（%.2f%%）"
          % (len(dig), dc, dc / n, dc / n * 100))
    print("  样例：%s" % dig[:25])
    print()
    print("🔴 判据：CER 重掷阈值 %.2f。拉丁+数字合计占原文 %.4f。"
          % (0.35, (lc + dc) / n))
    print("   即使全部被误判成错（每个 token 错成 3 倍长度），"
          "上界也才 %.4f —— 差 %.1f 倍才够触发重掷。"
          % ((lc + dc) * 3 / n, 0.35 / max((lc + dc) * 3 / n, 1e-9)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
