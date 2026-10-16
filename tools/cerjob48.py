# -*- coding: utf-8 -*-
"""cerjob48.py —— 数字密集型 job 的 CER / 耗时体检。

## 为什么要查
`tokstat_jobs48.py` 实测：真实 job 原文里阿拉伯数字占比最高 **15.49%**
（"1,052,749" "3.72" "64,200"），而案例库 full.txt **33 条 13 万字符里 0 个数字**
（疑似已归一化）。
→ v1.2「数字归一化补丁收益 0.0000」是在**没有数字的文案**上量出来的，
  **结论作废**：真实施工文案上它可能省下大量 CER 重掷（重掷 = 慢 3 倍的真因）。

## 看什么
数字占比高的 job 是否 CER 高、是否触发重掷、耗时多少。
"""
import json
import re
import sys

DIG = re.compile(r"[0-9][0-9,\.]*")


def main():
    d = json.load(open("/mnt/tts/data/jobs.json", encoding="utf-8"))
    it = d.values() if isinstance(d, dict) else d
    rows = []
    for v in it:
        if not isinstance(v, dict):
            continue
        t = v.get("text") or ""
        if len(t) < 200:
            continue
        n = max(len(t), 1)
        dr = sum(len(x) for x in DIG.findall(t)) / n
        fs = v.get("files") or []
        cer = fs[0].get("cer") if fs else None
        st = v.get("stats") or {}
        secs = st.get("audio_seconds") or 0.0
        chars = st.get("chars") or n
        dur = (v.get("finished_at") or 0) - (v.get("started_at") or 0)
        rows.append((dr, v.get("title") or "", v.get("owner"), n, cer,
                     secs, chars, dur, st.get("chunks_total") or 1))
    rows.sort(key=lambda r: -r[0])
    print("数字占比 Top 20（≥200 字的 job，共 %d 条）" % len(rows))
    print("%-20s %-7s %6s %7s %8s %7s %7s %6s %6s" %
          ("title", "owner", "数字%", "CER", "音频s", "字数", "耗时s",
           "块数", "字/s"))
    for dr, t, o, n, cer, secs, chars, dur, ck in rows[:20]:
        print("%-20s %-7s %6.2f %7s %8.1f %7d %7.0f %6d %6.2f" % (
            str(t)[:20], str(o)[:7], dr * 100,
            ("%.3f" % cer) if isinstance(cer, (int, float)) else "n/a",
            secs or 0.0, chars, dur, ck,
            (chars / secs) if secs else 0.0))
    hi = [r for r in rows if r[0] >= 0.03]
    print()
    print("数字占比 ≥3%% 的 job：%d 条" % len(hi))
    cs = [r[4] for r in hi if isinstance(r[4], (int, float))]
    if cs:
        print("  其中 CER：中位 %.4f  最大 %.4f  （阈值 0.35）"
              % (sorted(cs)[len(cs) // 2], max(cs)))
        print("  触发重掷的条数：%d / %d"
              % (len([c for c in cs if c > 0.35]), len(cs)))
    owners = {}
    for r in rows:
        owners[r[2]] = owners.get(r[2], 0) + 1
    print()
    print("owner 分布：%s" % owners)
    return 0


if __name__ == "__main__":
    sys.exit(main())
