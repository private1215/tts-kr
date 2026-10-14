# -*- coding: utf-8 -*-
"""补齐 el_feats.json 里缺失的 F0（pyin / 16k，与 el_baseline.py 同口径）。

背景：el_baseline2.py 当年为了省 20 分钟，F0 是从 elbase.log 里解析回来的，
而那份日志**只有男声**，所以女声 21 条的 f0_med 全是空的
-> 女声建不出 F0 基线。这里老老实实算一遍并写回缓存。

口径必须一致：librosa.pyin / 16k / fmin=60 / fmax=400，取中位数。
"""
import json
import os
import sys

import librosa
import numpy as np

CACHE = "/mnt/tts/diag/x8197/el_feats.json"
CASE_DIR = "/mnt/tts/data/cases"
ONLY_GENDER = sys.argv[1] if len(sys.argv) > 1 else "female"
# 🔴 --force：覆盖已有的 f0_med。
#    必须做一次：男声的 f0_med 是当年从 elbase.log 解析回来的**旧口径**，
#    实测比 pyin 口径低 7~13Hz（EL 案例用 pyin 量是 200.7/204.2/207.7，
#    而缓存里写的是 189.4~200.7）。两套口径混着比会得出「我们 F0 偏高 1.9σ」
#    这种假结论。统一到 pyin 后男女才可比。
FORCE = "--force" in sys.argv


def f0_median(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr,
                                 frame_length=2048, hop_length=256)
    v = f0[voiced] if voiced is not None else f0
    v = v[~np.isnan(v)] if v is not None else np.array([])
    if len(v) < 100:
        return None, None
    return float(np.median(v)), float(np.percentile(v, 90) - np.percentile(v, 10))


def main():
    cache = json.load(open(CACHE, encoding="utf-8"))
    todo = [(k, v) for k, v in cache.items()
            if (v.get("gender") or "") == ONLY_GENDER
            and (FORCE or not v.get("f0_med"))]
    print("待算 %d 条（%s）" % (len(todo), ONLY_GENDER), flush=True)
    for i, (k, v) in enumerate(todo, 1):
        p = os.path.join(CASE_DIR, k, "merged.wav")
        if not os.path.exists(p):
            print("  %d 缺文件 %s" % (i, k), flush=True)
            continue
        try:
            med, rng = f0_median(p)
        except Exception as e:
            print("  %d FAIL %s %r" % (i, k, repr(e)[:70]), flush=True)
            continue
        if med is None:
            print("  %d 测不出 %s" % (i, k), flush=True)
            continue
        # 🔴 只写 f0_med，不写 f0_range：
        #    男声 elbase.log 里的 f0_range 均值只有 17.3，明显是**半音**口径；
        #    我这里算的是 Hz 极差（女声实测 170~180），两者定义不同。
        #    写进同一个字段会把基线混成两代口径，宁可先留空。
        v["f0_med"] = med
        v.pop("f0_range", None)
        cache[k] = v
        print("  %d/%d %s F0=%.1f" % (i, len(todo), k, med), flush=True)
        json.dump(cache, open(CACHE, "w"), ensure_ascii=False, indent=1)
    print("DONE -> %s" % CACHE)


if __name__ == "__main__":
    main()
