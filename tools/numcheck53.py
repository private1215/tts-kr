#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""numcheck53.py —— 韩语文案「病态数字」检测器（只标注，不强制修改）。

## 为什么做这个
2026-10-20 三问深挖发现 L3 文案里有 `전일 대비 한 사쩜사퍼센트`：
native 数词「한」(1) 与 Sino 数词「사쩜사」(4.4) 混排，组不成合法数值。
TTS 忠实念出「한 사쩜사 퍼센트」，员工听到的就是怪音/错音。
ASR 侧（q2_num_audit3）证明其余数字念读全对 —— 问题在文案，不在链路。
本工具把这类「念不出来的数字」在合成前标出来，改不改由人定。

## 病态判据（两级：INVALID 必须改 / WARN 建议改）
R1  native+Sino 混排      `한 사쩜사퍼센트`                    INVALID
    · 例外（合法不报）：`한 육천오백원`（한=强调词+带大单位整数）、
      `한 일`（일=事/日 名词，单字不成数）、`한 두 달`（native 惯用）
R1b native+阿拉伯混排     `한 4.4%`                            INVALID
R2  native+大单位         `두 만` `세 억`（한 만->WARN）         INVALID/WARN
R3  小数点残缺            `사쩜`(尾空) `쩜사`(头空) `사쩜사쩜사`(双点)
                          `사쩜사만` `4.4만`(小数+大单位)       INVALID
R4  百分号倒置            `퍼센트 사점사`                      INVALID
R7  双数混排              `사쩜사 한 퍼센트`（数+native+单位）  INVALID
R5  native+百分号         `다섯 퍼센트`（应 오 퍼센트）          WARN
R8  小数被空格拆开        `사쩜 사`                            WARN

## 设计原则（避免误伤，宁漏勿误报）
- **native 数词 + 原生量词合法**：두 달/세 시/두 명/네 권/한 개/두 배 —— 不报。
- **「이」既是 Sino 2 也是主语词尾**：`수익이 두 배` 不触发 R7（数 token 须 ≥2 字符）。
- **句末 `.` 不是小数点**：`.` 只在右侧紧跟数字/Sino 字时才当小数点；
  `사.`（句号）/`4.`（列表号）不报。`쩜` 无歧义，始终当小数点。
- **「점」多义**：只认 `X점Y` / `X쩜Y` 三元组；`이 점`(这点)/`삼 점`? 不报。
- **单位后接韩字即非数字**：`네 조각`(조=块)/`두 프로젝트`(프로) 不报；
  `조/만/억` 后跟 `원`（조원/만원/억원）才算数。
- WARN 只提示；INVALID 才是「这数字念不出来」。

## 用法
    python3 tools/numcheck53.py <文本文件> [--json] [--min-sev WARN|INVALID]
    from tools.numcheck53 import scan_text      # -> [finding, ...]
    finding = {"pos":int, "expr":str, "rule":str, "sev":"INVALID"|"WARN",
               "why":str, "suggest":str}

开关：`TTS_NUMCHECK=0` 关闭生产侧扫描（默认开，只标注不拦截）。
"""
import json
import os
import re
import sys

# ---------------------------------------------------------------- 词表
NAT = ["한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉",
       "열", "스물", "스무", "서른", "마흔", "쉰", "예순", "일흔", "여든",
       "아흔", "둘", "셋", "넷"]
NAT_RE = "|".join(sorted(NAT, key=len, reverse=True))
SINO_CH = "일이삼사오육칠팔구십백천만억조쩜점"
SINO_DIGIT = "일이삼사오육칠팔구"
BIG_UNIT = "십|백|천|만|억|조"
# 百分号词：프로 后接韩字则为「프로젝트」等，不算
PCT_WORD = "퍼센트|프로(?![가-힣])|퍼센|%"
# 单位尾缀（R7 用；점 有多义，只在双数场景用）
UNIT_TAIL = ("원|주|퍼센트|프로|퍼센|%|포인트|명|건|배|달|시|분|초|년|월|일|"
             "개|대|층|번|조|만|억|천|백|십|달러")
_ONES = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]


def _has_sino_digit(s):
    return any(c in s for c in SINO_DIGIT)


def _has_big_unit(s):
    return any(u in s for u in ("십", "백", "천", "만", "억", "조"))


def _has_dec(s):
    return ("쩜" in s) or ("." in s)


def _glued_han(text, p0):
    """「한」紧跟前一韩字音节（无空白）= 하다/밝힌/대한 的冠形词尾，不是数词 1。
    `상승한 2,450선` / `전일 대비한 4.4%` 不报；`대비 한 사쩜사퍼센트`（有空白）照报。"""
    return p0 > 0 and "가" <= text[p0 - 1] <= "힣"


def _glued_in_word(text, p0):
    """native 数词**嵌在词里**（前一种是韩字、无空白）= 它不是在数数。

    🔴 真实误报（2026-10-21 男声标杆全文扫出来）：`외인 매도세만 잡힙니다`
    ——「매도세」(抛售势头) + 「만」(助词，仅仅)，세 是名词成分不是数词 3，
    만 也不是大单位。R2 把 `세`+`만` 读成了 native 数词 + Sino 大单位。

    判据与 `_glued_han` 同构：前一个字符是韩字（不是空白/数字/标点）→ 数词属于
    前一个词。真病态用例 `기업 가치가 세 억원이다` 的 세 前有空白，不受影响。
    """
    return p0 > 0 and "가" <= text[p0 - 1] <= "힣"


# ---------------------------------------------------------------- 读数渲染（建议用）
def _sino_int(n):
    """小整数 -> Sino 韩语读数（够改文案用）。q≥10 递归渲染（사십사만 的 q=44）。"""
    if n <= 0 or n >= 10 ** 12:
        return str(n)
    parts = []
    for unit, div in (("조", 10 ** 12), ("억", 10 ** 8), ("만", 10 ** 4),
                      ("천", 10 ** 3), ("백", 10 ** 2), ("십", 10 ** 1)):
        if n >= div:
            q = n // div
            if q != 1 or unit in ("만", "억", "조"):
                parts.append(_sino_int(q) if q >= 10 else _ONES[q])
            parts.append(unit)
            n %= div
    if n > 0:
        parts.append(_ONES[n])
    return "".join(parts)


def _dec_tail(s):
    return "".join(_ONES[int(c)] for c in s if c.isdigit())


def _parse_sino_int(t):
    """Sino 整数 -> 数值。사십사만=440000 / 이십=20 / 삼십오=35 / 만=10000"""
    total, sec, num = 0.0, 0.0, 0.0
    for c in t:
        if c in SINO_DIGIT:
            num = float(SINO_DIGIT.index(c) + 1)
        elif c in ("십", "백", "천"):
            sec += (num if num else 1.0) * {"십": 10, "백": 100,
                                            "천": 1000}[c]
            num = 0.0
        elif c in ("만", "억", "조"):
            total += ((sec + num) if (sec or num) else 1.0) * {
                "만": 10 ** 4, "억": 10 ** 8, "조": 10 ** 12}[c]
            sec, num = 0.0, 0.0
        else:
            return None
    return total + sec + num


def _parse_sino_value(tok):
    """Sino 数词 token -> 数值。先整串解析（십/백/천/만/억/조 是数值成分，
    不能被当单位词剥掉）；整串失败才剥单位词后缀（사쩜사퍼센트 -> 사쩜사）。"""
    for cand in (tok, re.sub("(%s)$" % UNIT_TAIL, "", tok)):
        m = re.match("^([%s]*)(쩜|점)([%s]+)$" % (SINO_CH, SINO_DIGIT), cand)
        if m:
            head = _parse_sino_int(m.group(1))
            if head is None:
                continue
            frac = float("0." + "".join(
                str(SINO_DIGIT.index(c) + 1) for c in m.group(3)))
            return head + frac
        v = _parse_sino_int(cand)
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------- 规则
def _rule_r1(text):
    """native + Sino 混排（INVALID；한+带大单位整数=强调词，合法）

    NUM_TOK 从 nat 匹配终点起（主正则的 \\s? 已吃掉中间空白）。
    闸门：tok 必须是「可解析的数值」——含小数点、含大单位、或后接单位词，
    否则视为名词（`한 일이 있어서요`：일=事/日 名词，이=主语词尾）。
    """
    out = []
    for m in re.finditer("(?:^|[^%s])(%s)\\s?(?=[%s])"
                         % (SINO_CH, NAT_RE, SINO_CH), text):
        nat = m.group(1)
        p0 = m.start(1)
        ts = m.end()          # NUM_TOK 起点（nat 与 tok 间空白已被吃掉）
        mt = re.compile("[%s][%s0-9,\\.%%]*" % (SINO_CH, SINO_CH)).match(
            text, ts)
        tok = mt.group(0) if mt else ""
        if len(tok) < 2 or not (_has_sino_digit(tok) or "쩜" in tok):
            continue          # 单字不成数（한 일=事/日）；量词不进 NUM_TOK
        tail = re.match("(%s)" % UNIT_TAIL,
                        text[ts + len(tok):] or "")
        unit = tail.group(1) if tail else ""
        if not (_has_dec(tok) or _has_big_unit(tok) or unit):
            continue          # 名词/词尾场景，不是数（한 일이 / 한 이틀）
        if nat == "한" and _glued_han(text, p0):
            continue          # 冠形词尾（상승한/대비한），不是数词
        if nat == "한" and not _has_dec(tok) and _has_big_unit(tok):
            continue          # 合法强调词：한 육천오백원 / 한 십 년
        v = _parse_sino_value(tok + unit)
        sug = []
        if v is not None and v == int(v):
            sug.append("若想表达 %s → 写「%s %s」"
                       % (int(v), _sino_int(int(v)), unit))
        else:
            sug.append("去掉「%s」，写「%s %s」" % (nat, tok, unit))
        if nat == "한" and isinstance(v, float) and v != int(v):
            d = ("%g" % v).split(".")[-1]
            sug.append("若想表达 1.%s → 写「일점%s %s」"
                       % (d, _dec_tail(d), unit))
        out.append({
            "pos": p0, "expr": text[p0:ts + len(tok) + len(unit)],
            "rule": "R1_native_sino_mix", "sev": "INVALID",
            "why": "native 数词「%s」与 Sino 数词「%s」混排，组不成合法数值"
                   % (nat, tok),
            "suggest": "；".join(sug) or "只用一种数词体系",
        })
    return out


def _rule_r1b(text):
    """native + 阿拉伯数字混排（INVALID）"""
    out = []
    for m in re.finditer("(?:^|[^%s0-9])(%s) (?=[0-9][0-9,\\.]*[0-9%%])"
                         % (SINO_CH, NAT_RE), text):
        nat, p0 = m.group(1), m.start(1)
        if nat == "한" and _glued_han(text, p0):
            continue          # 冠形词尾（상승한 2,450선），不是数词
        mt = re.match("[0-9][0-9,\\.]*", text[p0 + len(nat) + 1:])
        num = mt.group(0) if mt else ""
        if len(num) < 2 and "." not in num:
            continue
        out.append({
            "pos": p0, "expr": text[p0:p0 + len(nat) + 1 + len(num)],
            "rule": "R1b_native_arabic_mix", "sev": "INVALID",
            "why": "native 数词「%s」与阿拉伯数字「%s」混排" % (nat, num),
            "suggest": "统一成一种写法（韩语文案推荐汉字数词）",
        })
    return out


def _rule_r2(text):
    """native + 大单位（INVALID；한 -> WARN）。
    单位后必须是非韩字 / 원（조원/만원/억원）/ 系词이다·입니다·이에요，
    否则是 조각/만족/프로젝트 这类词。"""
    out = []
    for m in re.finditer("(?:^|[^%s])(%s)\\s?(만|억|조)(원)?"
                         "(?:(?<=원)|(?=이다)|(?=입니다)|(?=이에요)|(?![가-힣]))"
                         % (SINO_CH, NAT_RE), text):
        nat, u, won, p0 = m.group(1), m.group(2), m.group(3) or "", m.start(1)
        if nat == "한" and _glued_han(text, p0):
            continue          # 冠形词尾（상승한 2,450선），不是数词
        if _glued_in_word(text, p0):
            continue          # 数词嵌在词里（매도세만：세 属「매도세」，만 是助词）
        expr = text[p0:m.end()]
        if nat == "한":
            out.append({"pos": p0, "expr": expr, "rule": "R2_han_big_unit",
                        "sev": "WARN",
                        "why": "「한 %s%s」是口语省读，数字文本建议去掉「한」"
                               % (u, won),
                        "suggest": "写「%s%s」即可" % (u, won)})
        else:
            out.append({"pos": p0, "expr": expr,
                        "rule": "R2_native_big_unit", "sev": "INVALID",
                        "why": "native 数词「%s」不能带 Sino 大单位「%s%s」"
                               % (nat, u, won),
                        "suggest": "大写单位只用 Sino 数词（삼%s / 사%s）"
                                   % (u, u)})
    return out


def _rule_r3(text):
    """小数点残缺 / 双点 / 小数+大单位 / 空格拆开"""
    out = []

    def _add(p0, expr, why, sug, sev="INVALID", rule="R3_decimal_broken"):
        out.append({"pos": p0, "expr": expr, "rule": rule, "sev": sev,
                    "why": why, "suggest": sug})

    # 쩜：无歧义，始终当小数点
    for m in re.finditer("[%s0-9]*쩜[%s0-9]*" % (SINO_CH, SINO_CH), text):
        a, b, tok = m.start(), m.end(), m.group(0)
        if tok.count("쩜") > 1:
            _add(a, tok, "出现多个小数点", "一个数只能有一个小数点")
            continue
        left = tok.split("쩜")[0]
        right = tok.split("쩜")[1]
        nxt = text[b] if b < len(text) else ""
        if not left:
            _add(a, tok, "小数点前没有数字", "写成「일점%s」" % _dec_tail(right))
        elif not right:
            # 尾空：若后接「空白+数字」是 R8 场景（只 WARN），不重复报
            if re.match("[ \t]+[%s0-9]" % SINO_CH, text[b:]):
                continue
            _add(a, tok, "小数点后没有数字", "补上小数位或去掉「쩜」")
        elif right[-1] in "만억조":
            # SINO_CH 含 대형单位，正则已把它吃进 tok，查小数右半尾字
            _add(a, tok, "小数「%s」不能带大单位「%s」" % (tok, right[-1]),
                 "大单位前必须是整数，确认数值")
        elif nxt and nxt in "만억조":
            _add(a, tok + nxt, "小数「%s」不能带大单位「%s」" % (tok, nxt),
                 "大单位前必须是整数，确认数值")
    # 阿拉伯小数点：只在右侧紧跟数字/Sino 字时才算（句末 4. / 사. 不报）
    for m in re.finditer("[%s0-9]*\\.[%s0-9]+" % (SINO_CH, SINO_CH), text):
        a, b, tok = m.start(), m.end(), m.group(0)
        if tok.count(".") > 1:
            _add(a, tok, "出现多个小数点", "一个数只能有一个小数点")
            continue
        left = tok.split(".")[0]
        right = tok.split(".")[1]
        nxt = text[b] if b < len(text) else ""
        if not left:
            _add(a, tok, "小数点前没有数字", "补上整数部分")
        elif right and right[-1] in "만억조":
            _add(a, tok, "小数「%s」不能带大单位「%s」" % (tok, right[-1]),
                 "大单位前必须是整数，确认数值")
        elif nxt and nxt in "만억조":
            _add(a, tok + nxt, "小数「%s」不能带大单位「%s」" % (tok, nxt),
                 "大单位前必须是整数，确认数值")
    # 小数被空格拆开：사쩜 사 / 4.4 4
    for m in re.finditer("([%s0-9]쩜|[%s0-9]\\.[0-9])[ \t]+([%s0-9])"
                         % (SINO_CH, SINO_CH, SINO_CH), text):
        _add(m.start(), m.group(0), "小数位被空格拆开，可能被念成两个数",
             "写成连写「X%sY」" % ("쩜" if "쩜" in m.group(0) else "."),
             sev="WARN", rule="R8_spaced_decimal")
    return out


def _rule_r4(text):
    """百分号倒置（INVALID）。后接数 token 须 ≥2 字符且含数位。"""
    out = []
    for m in re.finditer("(%s)\\s?(?=[%s0-9])" % (PCT_WORD, SINO_CH), text):
        p0 = m.start()
        mt = re.compile("[%s][%s0-9,\\.]*" % (SINO_CH, SINO_CH)).match(
            text, m.end())
        tok = mt.group(0) if mt else ""
        if len(tok) < 2 or not _has_sino_digit(tok):
            continue          # 「퍼센트 이상」의 이 / 「% 만족」의 만 不算数
        out.append({
            "pos": p0, "expr": text[p0:m.end() + len(tok)],
            "rule": "R4_percent_inverted", "sev": "INVALID",
            "why": "百分号「%s」跑到了数字「%s」前面" % (m.group(1), tok),
            "suggest": "写成「%s %s」（数字在前）" % (tok, m.group(1)),
        })
    return out


def _rule_r5(text):
    """native + 百分号（WARN；한 除外——「한 퍼센트도 없다」是惯用）"""
    out = []
    nat_idx = {"두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4,
               "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}
    for m in re.finditer("(?:^|[^%s])(두|세|네|다섯|여섯|일곱|여덟|아홉|둘|셋|넷)"
                         "\\s?(%s)" % (SINO_CH, PCT_WORD), text):
        nat, pct, p0 = m.group(1), m.group(2), m.start(1)
        out.append({
            "pos": p0, "expr": text[p0:p0 + len(nat) + len(pct)],
            "rule": "R5_native_percent", "sev": "WARN",
            "why": "native 数词「%s」+百分号不规范，百分号前用 Sino 数词" % nat,
            "suggest": "写「%s %s」" % (_ONES[nat_idx[nat]], pct),
        })
    return out


def _rule_r7(text):
    """双数混排：数值 token(≥2 字符) + native 数词 + 单位（INVALID）"""
    out = []
    rx = re.compile("(?P<num>[%s0-9][%s0-9,\\.]*)\\s?(?P<nat>%s)"
                    "\\s?(?=(%s))" % (SINO_CH, SINO_CH, NAT_RE, UNIT_TAIL))
    for m in rx.finditer(text):
        num, nat = m.group("num"), m.group("nat")
        if len(num) < 2:
            continue          # 「수익이 두 배」의 이（主语词尾）不算数
        if nat == "한" and _glued_han(text, m.start("nat")):
            continue          # 冠形词尾（상승한 4.4 퍼센트），不是数词
        p0 = m.start()
        out.append({
            "pos": p0, "expr": text[p0:m.end()],
            "rule": "R7_double_number", "sev": "INVALID",
            "why": "「%s」与「%s」两个数挨在一起，中间没有连接词" % (num, nat),
            "suggest": "确认想表达哪个数，删掉另一个",
        })
    return out


_RULES = (_rule_r1, _rule_r1b, _rule_r2, _rule_r3, _rule_r4, _rule_r5, _rule_r7)


def scan_text(text, min_sev="WARN"):
    """扫一段韩语文案，返回 finding 列表（按 pos 排序，同 pos 去重）。"""
    if not text:
        return []
    finds = []
    for r in _RULES:
        try:
            finds.extend(r(text))
        except Exception:                                          # noqa: BLE001
            continue                      # 单条规则异常不影响其他规则
    order = {"INVALID": 0, "WARN": 1}
    finds.sort(key=lambda f: (f["pos"], order.get(f["sev"], 9)))
    seen, out = set(), []
    for f in finds:
        k = (f["pos"], f["expr"][:6])
        if k in seen:
            continue
        seen.add(k)
        out.append(f)
    if min_sev == "INVALID":
        out = [f for f in out if f["sev"] == "INVALID"]
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="文本文件（UTF-8）；缺省读 stdin")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--min-sev", default="WARN", choices=["WARN", "INVALID"])
    a = ap.parse_args()
    text = open(a.file, encoding="utf-8").read() if a.file else sys.stdin.read()
    finds = scan_text(text, a.min_sev)
    if a.json:
        print(json.dumps(finds, ensure_ascii=False, indent=1))
        return
    if not finds:
        print("✅ 未检出病态数字")
        return
    for f in finds:
        print("[%s] pos=%d %r" % (f["sev"], f["pos"], f["expr"]))
        print("   %s（%s）" % (f["why"], f["rule"]))
        print("   建议：%s" % f["suggest"])
    print("\n共 %d 处（INVALID %d / WARN %d）"
          % (len(finds), sum(1 for f in finds if f["sev"] == "INVALID"),
             sum(1 for f in finds if f["sev"] == "WARN")))


if __name__ == "__main__":
    main()
