# -*- coding: utf-8 -*-
"""对若干候选 wav 量一遍 eq_search6 的全套指标（含共振峰 z），用于横向对比。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eq_search6 as E  # noqa: E402

for p in sys.argv[1:]:
    p = p if os.path.isabs(p) else os.path.join(HERE, p)
    m = E.measure(p)
    print("=" * 72)
    print("%s   J=%.1f" % (os.path.basename(p), E.cost(m)))
    for k in E.BASE:
        v, (mm, ss) = m[k], E.BASE[k]
        z = (v - mm) / ss
        flag = "[X]" if abs(z) > 2 else ("[!]" if abs(z) > 1.5 else "[ok]")
        print("  %-4s %-11s %9.4f  EL %9.4f  z=%+6.2f"
              % (flag, k, v, mm, z))
    for i, nm in enumerate(("F1", "F2")):
        v = m.get(nm)
        if not v:
            continue
        z = (v - E.FORMANT_TARGET[i]) / E.FORMANT_SD[i]
        flag = "[X]" if abs(z) > 2 else ("[!]" if abs(z) > 1.5 else "[ok]")
        print("  %-4s %-11s %9.1f  EL %9.1f  (%+5.1f%%)  z=%+6.2f"
              % (flag, nm, v, E.FORMANT_TARGET[i],
                 100 * (v / E.FORMANT_TARGET[i] - 1), z))
