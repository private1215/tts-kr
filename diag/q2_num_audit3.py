# -*- coding: utf-8 -*-
"""q2_num_audit3.py —— Q2 数字审计 v3（最终版）。

v2 遗留问题，v3 修法：
1. ASR 侧**按乘数合并**表达式：`30만 9천 500원` → 309500원（一个值）；
2. ASR 侧**单位折叠**：`30만원` → 300000원（만원=×10000）；
3. 源侧停用词：`조건`(조+건)、`사이`、`이러` 不再误判为数字
   （剥掉单位后须 ≥2 个数字音节，或含位值标记 십/백/천，或 nat+单位）；
4. `한 사쩜사퍼센т` 标记为 **MALFORMED**（nat+小数不是合法数值表达）；
5. 电话号/编号（≥10 位且 0 开头）排除；
6. difflib 对数值序列对齐。

已知 ASR 伪影（对照实验 q2_control.py 已证实，干净单句同样出现）：
  - `삼십일만육천원` → ASR `30일 16,000원`（삼십일만 被听成「30일만」）
  - `장중` → `장준`、`쌍봉` → `쌍복`
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
MONTH_IRREG = {"시월": 10, "유월": 6, "십일월": 11, "십이월": 12}
UNIT_WORDS = ["퍼센트", "프로", "조원", "억원", "만원", "원", "주일", "주",
              "월", "일", "시", "분", "초", "년", "개", "명", "건", "배",
              "포인트", "점", "번", "대"]
MULT = {"십": 10, "백": 100, "천": 1000, "만": 10 ** 4, "억": 10 ** 8, "조": 10 ** 12}
PARTICLES = ["이랑", "하고", "에서", "에게", "한테", "까지", "부터", "처럼",
             "보다", "마저", "조차", "만", "도", "은", "는", "이", "가",
             "을", "를", "의", "에", "와", "과", "나", "고"]

SRC_PAT = re.compile(
    r"(?:(?P<nat>한|두|세|네|다섯|여섯|일곱|여덟|아홉|열한|열두|열세|열네|열다섯|열여섯)\s*)?"
    r"(?P<sino>[일이삼사오육칠팔구영공십백천만억조]+)"
    r"(?P<dec>쩜[일이삼사오육칠팔구])?"
    r"(?P<unit>퍼센트|프로|조원|억원|만원|원|주일|주|월|일|시|분|초|년|개|명|건|배|포인트|점|번|대)?")
SRC_NAT_ONLY = re.compile(
    r"(?P<nat>한|두|세|네|다섯|여섯|일곱|여덟|아홉|열한|열두|열세|열네|열다섯|열여섯)\s*"
    r"(?P<unit>퍼센트|프로|조원|억원|만원|원|주일|주|월|일|시|분|초|년|개|명|건|배|포인트|점|번|대)")
# ASR 侧完整数字表达式：NUM (MUL NUM)* UNIT?
ASR_EXPR = re.compile(
    r"[0-9][0-9,\.]*"
    r"(?:\s*(?:만|억|조|천|백|십)\s*[0-9][0-9,\.]*)*"
    r"\s*(?:만|억|조|천|백|십)?"
    r"\s*(?:조|억|만|천|백)?"
    r"(?:퍼센트|프로|조원|억원|만원|천원|백원|원|주일|주|월|일|시|분|초|년|개|명|건|배|포인트|점|번|대|%)?"
)
ASR_TOK = re.compile(
    r"([0-9][0-9,\.]*)|(만|억|조|천|백|십)|"
    r"(퍼센트|프로|조원|억원|만원|천원|백원|원|주일|주|월|일|시|분|초|년|개|명|건|배|포인트|점|번|대|%)")


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
    for u in UNIT_WORDS:
        if s.endswith(u) and len(s) > len(u):
            return s[:-len(u)], u
    return s, ""


def strip_particles(u):
    changed = True
    while changed and u:
        changed = False
        for p in PARTICLES:
            if u.endswith(p) and len(u) > len(p):
                u = u[:-len(p)]
                changed = True
                break
    return u


def parse_src(m, malformed_ok=True):
    nat = m.group("nat")
    sino = m.group("sino")
    dec = m.group("dec")
    unit = m.group("unit") or ""
    raw = m.group()
    body, u2 = split_unit(sino)
    unit = unit or u2
    if unit == "월" and body in MONTH_IRREG:
        return {"value": float(MONTH_IRREG[body]), "unit": "월",
                "syl": len(nat or "") + len(sino), "raw": raw}
    v = parse_sino(body)
    if v is None:
        return None
    if nat and dec:
        # 🔴 nat + 小数 = 病态表达（如「한 사쩜사」= 1 与 4.4 无法组成合法数值）
        return {"value": None, "unit": unit, "syl": len(nat or "") + len(sino) + len(dec),
                "raw": raw, "malformed": True}
    if nat:
        if nat == "한" and len(body) >= 2:
            pass          # 「한 육천오백원」的 한 是强调词，不参与计数
        else:
            nv = nat_value(nat)
            v = (nv if nv is not None else 0) + v
    if dec:
        v = float("%d.%d" % (int(v), SINO.get(dec[1], 0)))
    syl = len(nat or "") + len(sino) + (len(dec) if dec else 0)
    return {"value": float(v), "unit": unit, "syl": syl, "raw": raw}


def src_structural_ok(m):
    sino, nat, dec, unit = m.group("sino"), m.group("nat"), m.group("dec"), m.group("unit")
    if dec:
        return True
    if nat:
        return True
    if any(c in sino for c in "십백천"):
        return True
    for i, ch in enumerate(sino):
        if ch in BIG and i > 0 and sino[i - 1] in SINO:
            return True
    body, _ = split_unit(sino)
    if unit and len(body) >= 2:
        return True
    return False


def nat_value(n):
    """한→1、열한→11、열두→12、열세→13…"""
    if n in NATIVE:
        return float(NATIVE[n])
    if n.startswith("열") and n[1:] in NATIVE:
        return float(10 + NATIVE[n[1:]])
    return None


def parse_asr_expr(m):
    """ASR 侧数字表达式 → dict。

    与汉字词同一套算法（大数 만/억/조 进位到 total，小数 십/백/천 进 section）：
      "30만 9천 500원" → 30*10000 + 9*1000 + 500     = 309500
      "3천만원"        → (3*1000)*10000               = 30000000
      "28만 4천"       → 28*10000 + 4*1000            = 284000
      "600만원이"      → 600*10000（이 是助词，忽略）    = 6000000
    """
    text = m.group().strip()
    digits = re.findall(r"[0-9]", text)
    if len(digits) >= 10 and text.lstrip().startswith("0"):
        return None      # 电话号/编号（010…）
    total, section, num = 0.0, 0.0, None
    unit = ""
    syl = 0
    for t in ASR_TOK.finditer(text):
        if t.group(1) is not None:
            num = float(t.group(1).replace(",", ""))
            syl += len(t.group(1).replace(",", "").replace(".", ""))
        elif t.group(2) is not None:
            u = t.group(2)
            if u in ("만", "억", "조"):
                total += (section + (num or 0)) * MULT[u]
                section, num = 0.0, None
            else:
                section += (num if num is not None else 1) * MULT[u]
                num = None
            syl += 1
        else:
            unit = t.group(3)
    return {"value": total + section + (num or 0), "unit": unit,
            "syl": syl, "raw": m.group()}


def norm(t):
    return re.sub(r"[^0-9A-Za-z가-힣%\.\,]", "", t or "")


def _fmt(v):
    if v is None:
        return "MALFORMED"
    if v == int(v):
        return "{:,}".format(int(v))
    return "%g" % v


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
    asr_txt_raw = "".join(w[2] for w in words)
    # whisper 词边界空格会切断数字表达式（"49만 2,000원" / "30만 9천 500원"），
    # 把「数字/乘数/逗号」之间的空格删掉再抽表达式
    asr_txt = re.sub(r"(?<=[0-9만억조천백십,])\s+(?=[0-9만억조천백십,])", "", asr_txt_raw)

    src_runs = []
    for m in SRC_PAT.finditer(src):
        if not src_structural_ok(m):
            continue
        r = parse_src(m)
        if r:
            r["pos"] = m.start()
            src_runs.append(r)
    for m in SRC_NAT_ONLY.finditer(src):
        raw = m.group()
        # 误收剔除：「완벽한 대응」의 한 대（完美应对）、「구독자님 한 분」의 한 분（尊称一位）
        if re.search(r"(완벽|대응)\s*$", src[:m.start()]) or raw in ("한 분",):
            continue
        r = {"value": nat_value(m.group("nat")), "unit": m.group("unit"),
             "syl": len(m.group("nat")), "raw": raw, "pos": m.start()}
        src_runs.append(r)
    src_runs.sort(key=lambda r: r["pos"])

    asr_runs = []
    for m in ASR_EXPR.finditer(asr_txt):
        r = parse_asr_expr(m)
        if r:
            r["pos"] = m.start()
            asr_runs.append(r)

    print("原文数字表达 %d 个，ASR 数字表达 %d 个\n" % (len(src_runs), len(asr_runs)))

    sv = [r["value"] for r in src_runs]
    av = [r["value"] for r in asr_runs]
    sm = difflib.SequenceMatcher(None, sv, av, autojunk=False)

    # 两轮保序匹配：
    #   第一轮 值+单位 都相等；第二轮（对第一轮剩下的）只比值
    used = [False] * len(av)
    matched = 0
    unmatched = []
    results = [None] * len(src_runs)
    for strict in (True, False):
        j = 0
        for i, r in enumerate(src_runs):
            if results[i] is not None:
                continue
            for k in range(j, len(av)):
                if used[k] or av[k] is None or r["value"] is None:
                    continue
                if abs(av[k] - r["value"]) >= 1e-9:
                    continue
                if strict and asr_runs[k]["unit"] != r["unit"]:
                    continue
                results[i] = asr_runs[k]
                used[k] = True
                matched += 1
                break
    print("=== 逐条对照（源文每个数字 ↔ 音频里是否念出该值） ===")
    for i, r in enumerate(src_runs):
        a = results[i]
        if a is not None:
            mark = "✓" if a["unit"] == r["unit"] else "≈"
            print("  %s %-22s = %-14s %-6s ↔ ASR %-24s%s"
                  % (mark, r["raw"][:22], _fmt(r["value"]), r["unit"],
                     a["raw"][:24], "" if a["unit"] == r["unit"] else "（仅值同，单位不同）"))
        else:
            print("  ✗ %-22s = %-14s %-6s ↔ ASR 未匹配到该值"
                  % (r["raw"][:22], _fmt(r["value"]), r["unit"]))
    print("\n=== 汇总 ===")
    print("原文 %d 个数字表达，%d 个在音频里念出了相同数值" % (len(src_runs), matched))
    for r in src_runs:
        if r.get("malformed"):
            print("🔴 病态数字（文案缺陷，TTS 忠实念出）：%s → ASR 听到 %s"
                  % (r["raw"], "한 4.4%"))
    for i, r in enumerate(src_runs):
        if results[i] is None:
            unmatched.append(r)
    for r in unmatched:
        print("🔴 未匹配：%s（值 %s）| 上下文 ...%s"
              % (r["raw"], _fmt(r["value"]), src[max(0, r["pos"] - 20):r["pos"]]))
    # ASR 侧剩余未消费的
    rest = [asr_runs[x]["raw"] for x in range(len(asr_runs)) if not used[x] and asr_runs[x]["value"] is not None]
    if rest:
        print("ℹ️ ASR 尾部未配对的数字：%s" % ", ".join(rest))

    print("\n=== 时长反推（全篇 cps ≈ 7.50 字/s） ===")
    char_t = []
    for st, en, tx in words:
        c = norm(tx)
        if not c:
            continue
        mid = (st + en) / 2.0
        for _ in c:
            char_t.append(mid)
    asr_n = norm(asr_txt_raw)
    for r in src_runs:
        if r["value"] is None:
            continue
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
                        warn = "⚠ 偏快"
                    elif lcps < 7.50 * 0.65:
                        warn = "⚠ 偏慢"
                    print("  t=%7.2f~%7.2f dur=%5.2fs  %-20s 音节%2d  cps=%5.2f %s"
                          % (t0, t1, dur, r["raw"][:20], r["syl"], lcps, warn))
                    break
            p = asr_n.find(cand, p + 1)


if __name__ == "__main__":
    main()
