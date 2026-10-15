#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""server.py 里给 roomtone.apply 传「频谱斜率」（女声 -6 dB/oct）。

离线实验（tools/rt_slope.py）结果：RMS 不变的前提下把底噪斜率从 -3 改到 -6，
女声 flat 从 +4.54σ 掉到 +0.65σ，voiced 几乎不动（-2.44 -> -2.48）。
"""
import io
import shutil
import sys

P = "/mnt/tts/projects/qwen3-tts/app/server.py"

OLD1 = ('            _rt_db = None\n'
        '            if _legacy_post and roomtone.ENABLED:\n'
        '                _rt_db = roomtone.level_for(\n'
        '                    "male" if job.get("_is_male") else\n'
        '                    "female" if job.get("_is_male") is False else "")\n')
NEW1 = ('            _rt_db = None\n'
        '            # 底噪斜率（dB/oct）与电平是两个正交自由度：\n'
        '            #   电平决定「停顿空不空」(voiced)，斜率决定「频谱平不平」(flat)。\n'
        '            _rt_slope = roomtone.ROOMTONE_SLOPE_MALE\n'
        '            if _legacy_post and roomtone.ENABLED:\n'
        '                _rt_gender = ("male" if job.get("_is_male") else\n'
        '                              "female" if job.get("_is_male") is False\n'
        '                              else "")\n'
        '                _rt_db = roomtone.level_for(_rt_gender)\n'
        '                _rt_slope = roomtone.slope_for(_rt_gender)\n')

OLD2 = '                _nx = roomtone.apply(_nx, _nsr, _rt_db)\n'
NEW2 = '                _nx = roomtone.apply(_nx, _nsr, _rt_db, slope=_rt_slope)\n'


def main():
    with io.open(P, encoding="utf-8") as fh:
        s = fh.read()
    if OLD1 not in s:
        print("MISS: 锚点1 未找到，可能已改过")
        return 1
    if OLD2 not in s:
        print("MISS: 锚点2 未找到")
        return 1
    if NEW2 in s:
        print("ALREADY: 已改过，跳过")
        return 0
    shutil.copy2(P, P + ".bak-before-rtslope")
    s = s.replace(OLD1, NEW1).replace(OLD2, NEW2)
    with io.open(P, "w", encoding="utf-8") as fh:
        fh.write(s)
    print("OK: server.py 已接上底噪斜率")
    return 0


if __name__ == "__main__":
    sys.exit(main())
