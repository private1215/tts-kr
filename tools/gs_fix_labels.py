# -*- coding: utf-8 -*-
"""用案例自带的 ground-truth 文案(full.txt)纠正 ASR 标注错误。

为什么必须做这一步：
    ASR 把「한화솔루션」听成「선화 솔루션」、「시초가 탈환」听成「시초과 탈완」。
    拿这种错标去训练，模型会**学会把公司名念错**——而我们的对比文案第一句
    就是「반갑습니다, 한화솔루션 주주 여러분」，等于直接毁掉评测。

保守策略：只在 ASR 的词 **不在** full.txt 词表时，才用最近的词替换。
词表里已有的词一律不动（正确的词不会被改坏）。
"""
import os, re, sys, difflib
from collections import Counter

CASE_TXT = "/mnt/tts/data/cases/c3117b7b0fb1/full.txt"
DATA = "/mnt/tts/projects/gpt-sovits/dataset"

# 这些是助词/词尾，即使不在词表也别乱改
STOP = set("""은 는 이 가 을 를 의 에 서 로 으로 와 과 도 만 이나 나 거나 게 지 는요
이요 라고 고 지요 네요 죠 요 다 니 까 면 서 지 만 하 다 있 없 되 한 할 해 그 저 내 우리 제
그리고 그러나 그래서 그런데 하지만 또한 즉 또는 및 등 수 있 것 같 때문 때문""".split())

THRESH = 0.72
MIN_LEN = 2


def build_vocab(path):
    raw = open(path, encoding="utf-8").read()
    words = re.findall(r"[가-힣A-Za-z0-9]+", raw)
    cnt = Counter(words)
    return set(w for w in cnt if len(w) >= MIN_LEN and cnt[w] >= 1)


def fix_line(line, vocab, report):
    path, spk, lang, txt = line.split("|", 3)
    toks = txt.split()
    out = []
    for t in toks:
        core = re.sub(r"[^가-힣A-Za-z0-9]", "", t)
        if len(core) < MIN_LEN or core in STOP or core in vocab or core.isascii():
            out.append(t)
            continue
        cand = difflib.get_close_matches(core, vocab, n=1, cutoff=THRESH)
        if cand and len(cand[0]) >= len(core) - 1 and len(cand[0]) <= len(core) + 3:
            new = t.replace(core, cand[0])
            report.append((core, cand[0]))
            out.append(new)
        else:
            out.append(t)
    return "%s|%s|%s|%s" % (path, spk, lang, " ".join(out))


def main():
    vocab = build_vocab(CASE_TXT)
    print("词汇表大小: %d" % len(vocab), flush=True)
    report = []
    for split in ("train", "val"):
        p = os.path.join(DATA, split + ".list")
        lines = [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
        fixed = [fix_line(l, vocab, report) for l in lines]
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(fixed) + "\n")
        print("%s: %d 条已纠正" % (split, len(fixed)), flush=True)
    seen = {}
    for a, b in report:
        seen[(a, b)] = seen.get((a, b), 0) + 1
    print("\n=== 实际纠正（出现次数 >=2 的）===")
    for (a, b), n in sorted(seen.items(), key=lambda x: -x[1]):
        if n >= 2:
            print("  %s -> %s   x%d" % (a, b, n))
    print("纠正总数: %d，去重后 %d 种" % (len(report), len(seen)))
    print("DONE")


if __name__ == "__main__":
    main()
