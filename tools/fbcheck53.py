# -*- coding: utf-8 -*-
"""fbcheck53.py —— 反馈台素材体检：题目、音频是否存在、**采样率**。

2026-10-19 加的采样率检查：发现 ltasmatch53 / tailclean53b / tailclean53c /
clickfix53 全部把产物降到 16 kHz，而 16k 音频压 mp3 会明显伤音色
（0.9999@44.1k vs 0.7900@16k）。盲听素材如果是 16k 的，员工听的是降质版，
投票结论不可信 —— 必须先查这一项。

用法: venv/bin/python tools/fbcheck53.py
"""
import json
import os
import subprocess

FB = "/mnt/tts/fb"


def probe(p):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=sample_rate,bit_rate", "-of", "default=nw=1:nk=1", p],
            capture_output=True, text=True, timeout=30).stdout.split()
        return "%s Hz / %s bps" % (out[0], out[1] if len(out) > 1 else "?")
    except Exception as e:                                  # noqa: BLE001
        return "探测失败 %r" % (e,)


def main():
    ip = os.path.join(FB, "items.json")
    j = json.load(open(ip, encoding="utf-8"))
    items = j.get("items") or []
    print("题库 %s：%d 题" % (ip, len(items)))
    bad = []
    for it in items:
        print("\n%s  ver=%s" % (it.get("id"), it.get("ver")))
        print("  %s" % (it.get("title") or it.get("question") or "")[:70])
        for o in it.get("options") or []:
            f = o.get("file")
            p = os.path.join(FB, f) if f else None
            if not f:
                print("    (无音频) %s" % (o.get("label") or o.get("text")))
                continue
            ok = os.path.exists(p)
            info = probe(p) if ok else "缺失"
            print("    %-28s %-22s %s" % (f, info, o.get("label") or ""))
            if ok:
                sr = int(info.split()[0])
                if sr < 22050:
                    bad.append((f, sr))
    print("\n" + "=" * 50)
    if bad:
        print("⚠️  以下素材采样率 < 22.05k，压 mp3 后音色会受损，"
              "投票结论不可信：")
        for f, sr in bad:
            print("   %s  %d Hz" % (f, sr))
    else:
        print("✅ 所有素材采样率 >= 22.05k")


if __name__ == "__main__":
    main()
