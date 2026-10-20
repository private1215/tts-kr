# -*- coding: utf-8 -*-
"""q4_inserts.py —— 「额/啊」来源定性：全篇 ASR 增删里哪些是真多字/漏字。

res3.json 是 verify_0902e 的产物（已做数字归一化），ins 126 条 / del 97 条。
本脚本：
1. 滤掉已知数字伪影（归一化后仍残留的 0/만/원 等碎片）；
2. 把相邻的 ins/del 合并成块，恢复成「词」看；
3. 对每个块取音频时间区间，算区间峰值/均值的 dB 与局部语速，
   判定「音频里真有声（TTS 多念了）」还是「ASR 幻觉（那段接近静音）」；
4. 与员工反馈时间点（0/5/14/29/44/58/64/73/83/88/100/115/127/135/142s）对照。
"""
import json
import re

import numpy as np
import soundfile as sf

WAV = "/mnt/tts/diag/v0902/202610190902-1.mp3.16k.wav"
SR = 16000
# 员工反馈点（秒）：0开头/5磁/14没音/29吃字/44额/58额/64兹/73吃字/83/88/100啊/115没音/127额/135兹/142额
FB = [0, 5, 14, 29, 44, 58, 64, 73, 83, 88, 100, 115, 127, 135, 142]

NUMKO = re.compile(r"[일이삼사오육칠팔구십백천만억조]{2,}")
NUMAR = re.compile(r"[0-9]+")


def audio_env(x, sr):
    hop = int(0.01 * sr)
    n = len(x) // hop
    v = np.array([20 * np.log10(np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2)) + 1e-12)
                  for i in range(n)])
    return v, hop


def near(v, hop, t0, t1):
    i0, i1 = max(0, int(t0 / 0.01)), min(len(v), int(t1 / 0.01) + 1)
    if i1 <= i0:
        return None
    seg = v[i0:i1]
    return {"pk": float(seg.max()), "mean": float(seg.mean()),
            "dur": (i1 - i0) * 0.01}


def main():
    x, sr = sf.read(WAV, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    v, hop = audio_env(x, sr)
    d = json.load(open("/mnt/tts/diag/v0902/res3.json"))
    ins, dele = d["ins"], d["del"]

    def is_num_frag(s):
        s2 = NUMKO.sub("0", s)
        s2 = NUMAR.sub("0", s2)
        return s2.replace("0", "") == ""

    # 合并相邻块（间隔 <0.35s 且同为 ins 或 del）
    def merge(items, tag):
        items = sorted(items, key=lambda r: r[0])
        blocks = []
        for t0, t1, tx in items:
            if blocks and tag == blocks[-1]["tag"] and t0 - blocks[-1]["t1"] < 0.35:
                blocks[-1]["t1"] = max(blocks[-1]["t1"], t1)
                blocks[-1]["txt"] += tx
                blocks[-1]["n"] += 1
            else:
                blocks.append({"tag": tag, "t0": t0, "t1": t1, "txt": tx, "n": 1})
        return blocks

    allb = merge([(t0, t1, tx) for t0, t1, tx in ins], "ins") + \
           merge([(t0, t1, tx) for t0, t1, tx in dele], "del")
    allb.sort(key=lambda b: b["t0"])

    print("=" * 100)
    print("全篇增删块（已滤纯数字碎片）：t | 类型 | ASR 文本 | 音频区间峰值/均值 dB | 局部语速 | 员工反馈点?")
    print("=" * 100)
    real_ins, halluc, real_del = [], [], []
    for b in allb:
        if is_num_frag(b["txt"]):
            continue
        e = near(v, hop, b["t0"] - 0.05, b["t1"] + 0.05)
        if e is None:
            continue
        nsyl = len(b["txt"])
        lcps = nsyl / e["dur"] if e["dur"] > 0.02 else 0
        # 判定：区间峰值 > -25dB 视为「音频里真有声」
        verdict = ""
        if b["tag"] == "ins":
            if e["pk"] > -25:
                verdict = "🔴 真多念"
                real_ins.append((b, e))
            elif e["pk"] < -32:
                verdict = "⚪ ASR幻觉(近静音)"
                halluc.append((b, e))
            else:
                verdict = "⚠ 弱声待查"
        else:
            if e["pk"] > -25:
                verdict = "🔴 真漏念"
                real_del.append((b, e))
            else:
                verdict = "⚪ ASR幻觉"
                halluc.append((b, e))
        fb = ""
        for f in FB:
            if abs(f - b["t0"]) <= 6:
                fb = "👂%ds" % f
                break
        print("  %7.2f~%7.2f %-4s %-28s pk%6.1f mean%6.1f (%4.2fs) cps%5.1f  %-16s %s"
              % (b["t0"], b["t1"], b["tag"], b["txt"][:28], e["pk"], e["mean"],
                 e["dur"], lcps, verdict, fb))

    print("\n" + "=" * 100)
    print("汇总：真多念 %d 处 / 真漏念 %d 处 / ASR幻觉 %d 处"
          % (len(real_ins), len(real_del), len(halluc)))
    print("\n🔴 真多念清单（TTS 念了脚本里没有的音——这是员工「额/啊」的直接候选）：")
    for b, e in real_ins:
        fb = [f for f in FB if abs(f - b["t0"]) <= 6]
        print("   t=%7.2f  %-30s pk%.1fdB  %s"
              % (b["t0"], b["txt"][:30], e["pk"], "👂对应%ds" % fb[0] if fb else ""))
    print("\n🔴 真漏念清单：")
    for b, e in real_del:
        print("   t=%7.2f  %-30s pk%.1fdB" % (b["t0"], b["txt"][:30], e["pk"]))


if __name__ == "__main__":
    main()
