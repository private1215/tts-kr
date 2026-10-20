# -*- coding: utf-8 -*-
"""q2_num_audit2.py —— Q2 数字审计 v2（修正 v1 的单词污染问题）。

v1 教训：RUN 正则把 오늘/이/만약/전일/구독 这类普通词也当成数字（176 vs 81 个）。
v2 改法：
1. 原文侧只用**结构化数字模式**匹配，且要求满足至少一条：
   - 含位值标记 십/백/천；
   - 含 만/억/조 且前面有数字音节；
   - 含 쩜（小数点）；
   - 带单位后缀（원/주/%/월/일/시/분/초/년/개/명/건/배/포인트/점/주/번/대）；
   - 带固有词前缀（한/두/세/네/다섯/여섯/열한/열두…）。
2. **单位感知解析**：先剥掉尾部单位再解析数字
   （이일 = 2日 而非 2+1；사십사만주 = 440000주）。
3. 不规则读法：시월=10月、유월=6月、십일월=11月、십이월=12月；
   열한=11、열두=12。
4. ASR 侧剥掉主格/宾格助词（이/가/을/를…），电话号（010…）单独标记。
5. 用 difflib 对**数值序列**做对齐（兼容两侧个数不同）。
"""
import difflib
import json
import re

SINO = {"영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4,
        "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
SMALL = {"십": 10, "백": 100, "천": 1000}
BIG = {"만": 10 ** 4, "억": 10 ** 8, "조": 10 ** 12}
NATIVE = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5,
          "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
# 月份不规则读法
MONTH_IRREG = {"시월": 10, "유월": 6, "십일월": 11, "십이월": 12}

UNIT_TAIL = ["퍼센트", "프로", "조원", "억원", "만원", "원", "주일", "주",
             "월", "일", "시", "분", "초", "년", "개", "명", "건", "배",
             "포인트", "점", "번", "대"]
PARTICLES = ["이랑", "하고", "에서", "에게", "한테", "까지", "부터", "처럼",
             "보다", "마저", "조차", "만", "도", "은", "는", "이", "가",
             "을", "를", "의", "에", "와", "과", "나"]

# 原文侧数字模式
SRC_PAT = re.compile(
    r"(?P<nat>한|두|세|네|다섯|여섯|일곱|여덟|아홉|열한|열두|열세|열네|열다섯|열여섯)?"
    r"(?P<sino>[일이삼사오육칠팔구영공십백천만억조]+)"
    r"(?P<dec>쩜[일이삼사오육칠팔구])?"
    r"(?P<unit>퍼센트|프로|조원|억원|만원|원|주일|주|월|일|시|분|초|년|개|명|건|배|포인트|점|번|대)?")
ASR_PAT = re.compile(r"(?P<num>[0-9][0-9,]*(?:\.(?P<dec>[0-9]+))?)\s*(?P<unit>[가-힣%A-Za-z]*)?")


def parse_sino(s):
    total, section, num = 0, 0, None
    for ch in s:
        if ch in SINO:
            num = SINO[ch]
        elif ch in SMALL:
            v = num if num is not None else 1
            section += v * SMALL[ch]
            num = None
        elif ch in BIG:
            v = section + (num or 0)
            if v == 0:
                v = 1
            total += v * BIG[ch]
            section, num = 0, None
        else:
            return None
    return total + section + (num or 0)


def split_unit(s):
    for u in UNIT_TAIL:
        if s.endswith(u) and len(s) > len(u):
            return s[:-len(u)], u
    return s, ""


def parse_src(m):
    nat, sino, dec, unit = m.group("nat"), m.group("sino"), m.group("dec"), m.group("unit") or ""
    raw = m.group()
    body, u2 = split_unit(sino)
    unit = unit or u2
    # 月份不规则
    if unit == "월" and body in MONTH_IRREG:
        return {"value": float(MONTH_IRREG[body]), "unit": "월", "syl": len(nat or "") + len(sino), "raw": raw}
    v = parse_sino(body)
    if v is None:
        return None
    if nat:
        v = NATIVE[nat] + v
    if dec:
        v = float("%d.%d" % (int(v), SINO.get(dec[1], 0)))
    syl = len(nat or "") + len(sino) + (len(dec) if dec else 0)
    return {"value": float(v), "unit": unit, "syl": syl, "raw": raw}


def structural_ok(m):
    """判断这个匹配是真数字表达还是普通单词。"""
    sino = m.group("sino")
    nat, dec, unit = m.group("nat"), m.group("dec"), m.group("unit")
    if dec:
        return True
    if nat:
        return True
    if any(c in sino for c in "십백천"):
        return True
    # 만/억/조 前面必须紧跟数字音节
    for i, ch in enumerate(sino):
        if ch in BIG and i > 0 and sino[i - 1] in SINO:
            return True
    if unit:
        return True
    return False


def parse_asr(m):
    # 🔴 num 命名组已包含可选小数部分（"4" + ".4" = "4.4"），不要再拼 dec
    num, dec, unit = m.group("num"), m.group("dec"), (m.group("unit") or "")
    v = float(num.replace(",", ""))
    # 剥助词
    u = unit
    changed = True
    while changed and u:
        changed = False
        for p in PARTICLES:
            if u.endswith(p) and len(u) > len(p):
                u = u[:-len(p)]
                changed = True
                break
    syl = len(num.replace(",", "").replace(".", "")) + (len(dec) if dec else 0)
    return {"value": v, "unit": u, "syl": syl, "raw": m.group()}


def asr_is_number(m):
    u = m.group("unit") or ""
    for p in PARTICLES:
        if u.endswith(p) and len(u) > len(p):
            u = u[:-len(p)]
    if u in UNIT_TAIL or u in ("%",) or u == "":
        # 无单位也可能是纯数字（电话号/序号），只要不是助词粘着的普通词
        return True
    return False


def norm(t):
    return re.sub(r"[^0-9A-Za-z가-힣%\.\,]", "", t or "")


def main():
    j = json.load(open("/mnt/tts/data/jobs.json"))
    jobs = j if isinstance(j, list) else j.get("jobs", j)
    if isinstance(jobs, dict):
        jobs = list(jobs.values())
    src = ""
    for it in jobs:
        if isinstance(it, dict) and it.get("title") == "202610190902":
            src = it.get("text") or ""
            break

    words = json.load(open("/mnt/tts/diag/v0902/words.json"))
    if isinstance(words, dict):
        words = words.get("words", [])
    asr_txt = "".join(w[2] for w in words)

    src_runs = []
    for m in SRC_PAT.finditer(src):
        if not structural_ok(m):
            continue
        r = parse_src(m)
        if r:
            r["pos"] = m.start()
            src_runs.append(r)

    asr_runs = []
    for m in ASR_PAT.finditer(asr_txt):
        if not asr_is_number(m):
            continue
        r = parse_asr(m)
        if r:
            r["pos"] = m.start()
            asr_runs.append(r)

    print("原文数字表达 %d 个，ASR 数字表达 %d 个\n" % (len(src_runs), len(asr_runs)))

    sv = [r["value"] for r in src_runs]
    av = [r["value"] for r in asr_runs]
    sm = difflib.SequenceMatcher(None, sv, av, autojunk=False)

    print("=== 数值序列对齐（只列有差异的块） ===")
    nbad = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        nbad += 1
        print("\n[%s] 原文 #%d~#%d  ↔  ASR #%d~#%d" % (tag, i1, i2, j1, j2))
        for k in range(i1, i2):
            r = src_runs[k]
            print("   原文[%3d] t?  %-24s = %-14s unit=%-6s  ...%s..."
                  % (r["pos"], r["raw"][:24], _fmt(r["value"]), r["unit"],
                     src[max(0, r["pos"] - 10):r["pos"]]))
        for k in range(j1, j2):
            r = asr_runs[k]
            print("   ASR [%3d]      %-24s = %-14s unit=%-6s  ...%s..."
                  % (r["pos"], r["raw"][:24], _fmt(r["value"]), r["unit"],
                     asr_txt[max(0, r["pos"] - 10):r["pos"]]))

    eq = sum(b.size for b in sm.get_matching_blocks())
    print("\n=== 汇总 ===")
    print("数值完全对上：%d / %d（原文侧）" % (eq, len(sv)))
    print("有差异的块：%d 个" % nbad)

    # 时长反推
    print("\n=== 时长反推（全篇 cps ≈ 7.50 字/s） ===")
    char_t = []
    for st, en, tx in words:
        c = norm(tx)
        if not c:
            continue
        mid = (st + en) / 2.0
        for _ in c:
            char_t.append(mid)
    asr_n = norm(asr_txt)
    for r in src_runs:
        cand = _fmt(r["value"])
        p = asr_n.find(cand)
        while p >= 0:
            i0, i1 = p, p + len(cand)
            if 0 < i0 < len(char_t) and i1 - 1 < len(char_t):
                t0, t1 = char_t[i0], char_t[i1 - 1]
                dur = t1 - t0
                if dur > 0.02 and r["syl"] > 0:
                    lcps = r["syl"] / dur
                    warn = ""
                    if lcps > 7.50 * 1.35:
                        warn = "⚠ 局部偏快（多字？）"
                    elif lcps < 7.50 * 0.65:
                        warn = "⚠ 局部偏慢（漏字/拖长？）"
                    print("  t=%7.2f~%7.2f dur=%5.2fs  %-22s 音节%2d  cps=%5.2f %s"
                          % (t0, t1, dur, r["raw"][:22], r["syl"], lcps, warn))
                    break
            p = asr_n.find(cand, p + 1)


def _fmt(v):
    if v == int(v):
        return "{:,}".format(int(v))
    return "%g" % v


if __name__ == "__main__":
    main()
