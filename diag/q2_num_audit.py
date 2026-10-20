# -*- coding: utf-8 -*-
"""q2_num_audit.py —— Q2：数字念读审计（先确保"数字正确"，再谈吐字）。

背景：whisper 会把韩语口述数字转写成阿拉伯数字（사십사만주 → 44만주），
字符级 ASR diff 会因此制造大量假 insert/delete。要判断"到底念对没念对"，
必须把**两侧都归一到整数**再比对。

做法：
1. 原文侧：抽出每个数字表达（汉字词/固有词/쩜小数/单位），解析成数值；
2. ASR 侧：抽出阿拉伯数字表达，解析成数值；
3. 按出现顺序逐一配对，打印不一致项；
4. 对每个数字表达，用 ASR 词级时间戳定位音频区间，
   算「区间净发声时长 ÷ 该数字的口语音节数」= 局部语速，
   语速异常 ⇒ 音频里实际念的音节数可能与文本不符（多字/漏字的真证据）。

汉字词数字体系（已联网确认）：
  일이삼사오육칠팔구 + 십/백/천/만/억/조；
  100=백（非 일백）、1000=천、10000=만；
  사십사만 = 44만 = 440000；삼십만오백 = 300500；백퍼센트 = 100%。
"""
import json
import re

SINO = {"영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4,
        "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
SMALL = {"십": 10, "백": 100, "천": 1000}
BIG = {"만": 10 ** 4, "억": 10 ** 8, "조": 10 ** 12}
NATIVE = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5,
          "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}
UNIT_WORDS = ["퍼센트", "프로", "달러", "조원", "억원", "만원", "원", "주",
              "포인트", "점", "월", "일", "시", "분", "초", "년", "개",
              "명", "건", "배", "주일", "주", "번", "대", "퍼"]

RUN = re.compile(
    r"[일이삼사오육칠팔구영공십백천만억조쩜퍼센트%프로달러원주월일시분초년개명건배점대"
    r"여불무한두세네다섯여섯아홉일곱여덟0-9\.\,]+")
SRC_PAT = re.compile(
    r"^(한|두|세|네|다섯|여섯|일곱|여덟|아홉)?"
    r"([일이삼사오육칠팔구영공십백천만억조]+)"
    r"(?:쩜([일이삼사오육칠팔구]))?"
    r"([가-힣A-Za-z%]*)$")
ASR_PAT = re.compile(
    r"^([0-9][0-9,]*)(?:\.([0-9]+))?\s*([가-힣A-Za-z%]*)$")


def parse_sino(s):
    """사십사만→440000；삼십만오백→300500；백→100；이→2。"""
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


def norm(t):
    return re.sub(r"[^0-9A-Za-z가-힣%\.\,]", "", t or "")


def parse_src_run(s):
    """原文侧数字表达 → dict(value, unit, syl)。syl=口语音节数。"""
    m = SRC_PAT.match(s)
    if not m:
        return None
    nat, sino, dec, unit = m.groups()
    v = parse_sino(sino)
    if v is None:
        return None
    if nat:
        v = NATIVE[nat] + v          # 한 사쩜사 = 1 + 4.4（连读读法）
    if dec:
        v = float("%d.%d" % (v, SINO.get(dec, 0)))
    syl = len(nat or "") + len(sino) + (len(dec) + 1 if dec else 0)
    return {"value": v, "unit": unit, "syl": syl, "raw": s}


def parse_asr_run(s):
    """ASR 侧数字表达 → dict(value, unit, syl)。"""
    m = ASR_PAT.match(s.strip())
    if not m:
        return None
    a, b, unit = m.groups()
    v = float(a.replace(",", "") + ("." + b if b else ""))
    syl = len(a.replace(",", "").replace(".", "")) + (len(b) if b else 0)
    return {"value": v, "unit": unit, "syl": syl, "raw": s}


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

    src_runs, asr_runs = [], []
    for m in RUN.finditer(src):
        r = parse_src_run(m.group())
        if r and re.search(r"[0-9일이삼사오육칠팔구영공십백천만억조]", m.group()):
            r["pos"] = m.start()
            src_runs.append(r)
    for m in RUN.finditer(asr_txt):
        r = parse_asr_run(m.group())
        if r:
            r["pos"] = m.start()
            asr_runs.append(r)

    print("原文数字表达 %d 个，ASR 数字表达 %d 个\n" % (len(src_runs), len(asr_runs)))
    print("%-4s %-26s %-12s | %-4s %-22s %-12s %s"
          % ("#", "原文", "解析值", "#", "ASR", "解析值", "判定"))
    print("-" * 100)

    # 按顺序配对（数量不同时用最长公共子序列思路太复杂，先按顺序一对一，
    # 只报「值不一致」的；ASR 可能把相邻数字合并/拆开，单独标出）
    n = min(len(src_runs), len(asr_runs))
    bad = 0
    for i in range(n):
        s, a = src_runs[i], asr_runs[i]
        same = abs(s["value"] - a["value"]) < 1e-9
        flag = "OK" if same else "🔴 不一致"
        if not same:
            bad += 1
        print("%-4d %-26s %-12s | %-4d %-22s %-12s %s"
              % (i, s["raw"][:26], _fmt(s["value"]),
                 i, a["raw"][:22], _fmt(a["value"]), flag))

    print("\n共配对 %d 组，值不一致 %d 组" % (n, bad))
    if len(src_runs) != len(asr_runs):
        print("⚠️ 两侧数字个数不同（%d vs %d），顺序配对可能错位，需人工看上面明细"
              % (len(src_runs), len(asr_runs)))

    # 时长反推：每个原文数字表达，在 ASR 里找对应区间，算局部语速
    print("\n" + "=" * 100)
    print("时长反推（局部语速 = 净发声时长 ÷ 口语音节数；全篇 cps ≈ 7.50 字/s）")
    print("=" * 100)
    # 建 ASR 字符 → 时间 映射（同 verify_0902e 口径）
    char_t = []
    for st, en, tx in words:
        c = norm(tx)
        if not c:
            continue
        mid = (st + en) / 2.0
        for _ in c:
            char_t.append(mid)
    asr_n = norm(asr_txt)
    for s in src_runs:
        # 在 ASR 文本里找这个数字的阿拉伯形式
        val = s["value"]
        if isinstance(val, float) and val == int(val):
            val = int(val)
        cand = _fmt(val)
        p = asr_n.find(cand)
        while p >= 0:
            i0, i1 = p, p + len(cand)
            if 0 < i0 < len(char_t) and i1 - 1 < len(char_t):
                t0, t1 = char_t[i0], char_t[i1 - 1]
                dur = t1 - t0
                if dur > 0.02 and s["syl"] > 0:
                    lcps = s["syl"] / dur
                    warn = ""
                    if lcps > 7.50 * 1.35:
                        warn = "⚠ 局部偏快（多字？）"
                    elif lcps < 7.50 * 0.65:
                        warn = "⚠ 局部偏慢（漏字/拖长？）"
                    print("  t=%7.2f~%7.2f  dur=%5.2fs  %-24s 音节%2d  局部cps=%5.2f  %s"
                          % (t0, t1, dur, s["raw"][:24], s["syl"], lcps, warn))
                    break
            p = asr_n.find(cand, p + 1)


def _fmt(v):
    if isinstance(v, float) and v == int(v):
        v = int(v)
    if isinstance(v, int):
        return "{:,}".format(v)
    return "%g" % v


if __name__ == "__main__":
    main()
