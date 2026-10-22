# -*- coding: utf-8 -*-
"""韩语数字/金融单位标准化：把阿拉伯数字转成韩语读法。

为什么必须独立做这一层
----------------------
人工盲听反馈（2026-10-09）明确点名：数字密集处最容易读错、读快、甚至乱码。
原有 `pipeline.KO_NUMBER_UNIT` 只把 `%` 换成"퍼센트"，**数字本身仍是阿拉伯数字**，
读法完全交给模型临场判断：

    35,000원  →  "35,000원"   （模型自己猜）
    95.38%    →  "95.38퍼센트"

这既违反"金融数据必须使用正确的韩语数字读法"，也会让模型在长数字上加速、重复。

本模块把规则写死：

| 原文 | 读法 |
|---|---|
| 35,000원 | 삼만 오천 원 |
| 36,400원 | 삼만 육천사백 원 |
| 69,900원 | 육만 구천구백 원 |
| 95.38% | 구십오 점 삼팔 퍼센트 |
| 72.39% | 칠십이 점 삼구 퍼센트 |
| 350만 주 | 삼백오십만 주 |
| 23만 1,914주 | 이십삼만 천구백십사 주 |
| 10월 9일 | 시월 구일 |

设计原则（避免误伤）
-------------------
- **只处理带单位的数字**（원/주/만/억/%/일/월/개/배/명/년/포인트 …）。
  裸数字不动——股票代码（005930）、电话号码、年份代号各有读法，
  规则化反而会读错，交给模型更安全。
- 韩语数字以 **4 位一组**（만/억/조），组内 천/백/십 的 1 不读 일
  （1,914 → 천구백십사，不是 일천구백십사）；组位为 1 时保留 일（일만/일억）。
- 小数逐位读（95.38 → 구십오 점 삼팔），符合财经播报习惯。
- 月份有特殊读法：10월=시월, 6월=유월, 11월=십일월, 12월=십이월。

开关：`TTS_KO_NUM=1` 启用（默认关闭，先在服务器做同文案 A/B 再决定是否默认开）。
"""
import re

# 汉字数词
ONES = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
TENS = ["", "십", "이십", "삼십", "사십", "오십", "육십", "칠십", "팔십", "구십"]
# 月份特殊读法
# 🔴🔴 不要按汉字数词"规范化"改这张表！韩语月份有固定异读，改了就读错：
#     6월  = 유월  （不是 육월）
#     10월 = 시월  （不是 십월）  ← 2026-10-16 用户纠正过：改成 십월 是错的
#     11월 = 십일월 / 12월 = 십이월（这两个按汉字数词，没有异读）
#    其余 1/2/3/4/5/7/8/9 月按汉字数词，无特殊情况。
MONTHS = {1: "일월", 2: "이월", 3: "삼월", 4: "사월", 5: "오월", 6: "유월",
          7: "칠월", 8: "팔월", 9: "구월", 10: "시월", 11: "십일월",
          12: "십이월"}
# 带这些单位才转换（按长度降序匹配，避免 "%/" 之类漏掉）
UNITS = ["퍼센트", "포인트", "프로", "퍼센", "%", "원", "주", "개", "배",
         "명", "년", "월", "일", "분", "초", "달러", "억", "만", "조",
         "천원", "만원"]


def _group(g: int) -> str:
    """0~9999 的韩语读法。천/백/십 位为 1 时不读 일。"""
    if g == 0:
        return ""
    out = []
    thou, rest = divmod(g, 1000)
    hund, rest2 = divmod(rest, 100)
    ten, one = divmod(rest2, 10)
    if thou:
        out.append(("천" if thou == 1 else ONES[thou] + "천"))
    if hund:
        out.append(("백" if hund == 1 else ONES[hund] + "백"))
    if ten:
        out.append(("십" if ten == 1 else ONES[ten] + "십"))
    if one:
        out.append(ONES[one])
    return "".join(out)


def ko_number(n: int) -> str:
    """整数 → 韩语汉字数词读法（4 位一组：만/억/조）。"""
    if n == 0:
        return "영"
    if n < 0:
        return "마이너스 " + ko_number(-n)
    groups, units = [], ["", "만", "억", "조"]
    while n > 0:
        groups.append(n % 10000)
        n //= 10000
    out = []
    for i, g in enumerate(groups):
        if not g:
            continue
        u = units[i] if i < len(units) else ""
        # 组位为 1 且高位（만/억/조）时保留 일：일만 / 일억
        word = ("일" + u) if (g == 1 and u) else (_group(g) + u)
        out.append(word)
    return " ".join(reversed(out))


def _decimals(dec: str) -> str:
    """小数部分逐位读：38 → 삼팔。"""
    return "".join(ONES[int(c)] for c in dec if c.isdigit())


def _read_amount(num: str):
    """'35,000' / '95.38' → (整数读法, 小数读法 or '')"""
    num = num.replace(",", "").strip()
    if "." in num:
        ip, dp = num.split(".", 1)
        ip = ip or "0"
        return ko_number(int(ip)), _decimals(dp)
    return ko_number(int(num)), ""


def _money(m):
    """35,000원 → 삼만 오천 원"""
    ip, dp = _read_amount(m.group(1))
    unit = m.group(2)
    if dp:
        return f"{ip} 점 {dp} {unit}"
    return f"{ip} {unit}"


def _pct(m):
    """95.38% → 구십오 점 삼팔 퍼센트"""
    ip, dp = _read_amount(m.group(1))
    if dp:
        return f"{ip} 점 {dp} 퍼센트"
    return f"{ip} 퍼센트"


def _man(m):
    """350만 → 삼백오십만 ；23만 → 이십삼만"""
    return ko_number(int(m.group(1).replace(",", ""))) + "만"


def _date(m):
    """10월 9일 → 시월 구일"""
    mo, d = int(m.group(1)), int(m.group(2))
    mw = MONTHS.get(mo, ko_number(mo) + "월")
    return f"{mw} {ko_number(d)}일"


# 顺序很重要：先长单位/复合，再短的
RULES = [
    (re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일"), _date),
    (re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*%"), _pct),
    (re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(퍼센트|퍼센|프로)"), _pct),
    (re.compile(r"(\d[\d,]*)\s*만(?=\s|주|원|개|명|$)"), _man),
    (re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(원|주|개|배|명|달러|포인트)"), _money),
]


def normalize_numbers(text: str) -> str:
    """把带单位的数字转成韩语读法。裸数字（股票代码等）保持原样。"""
    if not text:
        return text
    t = text
    for pat, fn in RULES:
        t = pat.sub(fn, t)
    return t


def enabled() -> bool:
    import os
    return os.environ.get("TTS_KO_NUM", "0").strip().lower() in ("1", "true", "yes")
