#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""konum 单测：用例取自人工盲听反馈里给出的对照表，必须全对。

    python tools/test_konum.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.konum import normalize_numbers, ko_number  # noqa: E402

# (输入, 期望输出)
CASES = [
    ("35,000원", "삼만 오천 원"),
    ("36,400원", "삼만 육천사백 원"),
    ("69,900원", "육만 구천구백 원"),
    ("95.38%", "구십오 점 삼팔 퍼센트"),
    ("72.39%", "칠십이 점 삼구 퍼센트"),
    ("350만 주", "삼백오십만 주"),
    ("23만 1,914주", "이십삼만 천구백십사 주"),
    ("10월 9일", "시월 구일"),
    # 不应误伤：股票代码等裸数字保持原样
    ("005930", "005930"),
    ("종목명 005930 삼성전자", "종목명 005930 삼성전자"),
]


def main():
    bad = 0
    for src, want in CASES:
        got = normalize_numbers(src)
        ok = (got == want)
        bad += 0 if ok else 1
        print(f"  {'✅' if ok else '🔴'} {src!r} -> {got!r}"
              + ("" if ok else f"   期望 {want!r}"))
    print(f"\n{len(CASES) - bad}/{len(CASES)} 通过")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
