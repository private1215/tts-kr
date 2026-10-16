#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pause47.py —— 「语速慢」到底慢在哪：念得慢，还是塞满了停顿？

gs47m3（正确 yohan 权重）1500 字出了 669.4s，而 EL 3416 字才 509s。
要么每个字拖长，要么段间塞了大段静音。本脚本把两者分开：

  · 发声占比 voiced%   = 高于阈值的帧比例（阈值 = 中位 − 30dB）
  · 停顿总时长 / 段数  = 连续低于阈值的游程（≥200ms 才算停顿）
  · 最长停顿
  · 「念字段」内的语速 = 只算发声部分的字/秒（排除停顿稀释）

用法：python3 pause47.py --spec spec.json
"""
import argparse
import json
import subprocess

import numpy as np

SR = 16000


def load(p):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ar", str(SR),
                        "-ac", "1", "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def db(v):
    return 20.0 * np.log10(max(v, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args()
    items = json.load(open(a.spec, encoding="utf-8"))

    print(f"{'样本':<28}{'时长':>8}{'发声%':>8}{'停顿s':>8}"
          f"{'停顿段':>8}{'最长s':>8}{'净字/秒':>9}")
    print("-" * 78)
    for it in items:
        x = load(it["path"])
        dur = len(x) / SR
        n = int(SR * 0.01)                       # 10ms 帧
        m = (len(x) // n) * n
        rms = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
        thr = np.median(rms) * 10 ** (-30 / 20)
        voiced = rms > thr
        vpct = voiced.mean() * 100
        # 停顿游程
        gaps, cur = [], 0
        for v in voiced:
            if not v:
                cur += 1
            else:
                if cur * 0.01 >= 0.20:
                    gaps.append(cur * 0.01)
                cur = 0
        if cur * 0.01 >= 0.20:
            gaps.append(cur * 0.01)
        gsum = sum(gaps)
        gmax = max(gaps) if gaps else 0.0
        net = (it.get("chars") or 0) / max(dur - gsum, 1e-6)
        print(f"{it['name']:<28}{dur:>8.1f}{vpct:>8.1f}{gsum:>8.1f}"
              f"{len(gaps):>8}{gmax:>8.1f}{net:>9.2f}")


if __name__ == "__main__":
    main()
