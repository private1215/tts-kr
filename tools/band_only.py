# -*- coding: utf-8 -*-
"""只量频段+flat（跳过共振峰，快），用于沿处理链逐段定位是谁引入了偏差。

用法：
    python band_only.py vox_male_raw.wav _hb_rate.wav _hb_spec.wav m_hb_1.045.wav
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eq_search6 as E  # noqa: E402

E.FORMANT_W = 0.0          # 关掉共振峰，省掉 Praat 的大头
KEYS = [k for k in E.BASE if k != "flat"]

paths = sys.argv[1:] or ["vox_male_raw.wav"]
rows = []
for p in paths:
    p = p if os.path.isabs(p) else os.path.join(HERE, p)
    rows.append((os.path.basename(p), E.measure(p)))

hdr = "%-24s %6s | " % ("文件", "flat")
hdr += " ".join("%7s" % k.replace("b", "").replace("_", "~") for k in KEYS)
hdr += " | %7s" % "J频段"
print(hdr)
print("-" * len(hdr))
for name, m in rows:
    mm, ss = E.BASE["flat"]
    j = sum(((m[k] - E.BASE[k][0]) / E.BASE[k][1]) ** 2 for k in KEYS)
    print("%-24s %+6.2f | " % (name, (m["flat"] - mm) / ss)
          + " ".join("%+7.2f" % ((m[k] - E.BASE[k][0]) / E.BASE[k][1])
                     for k in KEYS)
          + " | %7.1f" % j)
