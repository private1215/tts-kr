# -*- coding: utf-8 -*-
"""从**未训练的 hold 案例**上切出多段测试文案，供「同文案引擎对照」。

为什么可以切多段：
    gs_prep_group.py 里 `train_cids = [c for c in cids if c != hold]` —— hold
    那条案例**整条都不进训练集**（不是只留前 60 秒）。所以从它身上切任意
    片段当测试文案，模型都没背过，分数是诚实的。
    这样不必为每个新文案重训一遍（40 epoch × 2 说话人），成本从几小时
    降到每次推理几十秒。

怎么切：
    用 ASR 的**词级时间戳**定位到秒，取 [start, start+sec] 区间内的文本。
    不用字符比例硬切 —— 那是早期 lora_data.py 的做法，会切出半句话。
    落点会向后退到最近的句末标点，保证每段是完整句。

用法：
    python tools/gs_case_texts.py --spk SOLA --hold 00b6a45ad6aa
    # 默认切 3 段：0s / 150s / 300s 起，每段 ~60 秒
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

BASE = os.environ.get("TTS_BASE", "").strip() or "/mnt/tts"
CASES_JSON = os.path.join(BASE, "data", "cases.json")
CASES_DIR = os.path.join(BASE, "data", "cases")
GS_OUT = os.environ.get("GS_OUT", "/mnt/tts/projects/gpt-sovits/out")

# 每段起点（秒）—— 跨过开头、中段、后段，覆盖不同语境
STARTS = [0.0, 150.0, 300.0]
DUR = 60.0
_END = "。！？.!?\n"


def transcribe_seg(wav, a, b, out_wav):
    """裁 [a,b) 秒到 out_wav，返回 (文本, 时长)。"""
    import soundfile as sf
    x, sr = sf.read(wav, dtype="float32")
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    s, e = int(a * sr), int(b * sr)
    seg = x[s:e]
    if len(seg) < sr * 2:
        return "", 0.0
    sf.write(out_wav, seg, sr, subtype="PCM_16")
    from app.asr_check import transcribe
    return (transcribe(out_wav, "ko") or "").strip(), len(seg) / float(sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spk", required=True)
    ap.add_argument("--hold", required=True)
    ap.add_argument("--starts", default=",".join("%g" % s for s in STARTS))
    ap.add_argument("--dur", type=float, default=DUR)
    a = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.path.join(BASE, "models", "asr"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    cid, hold = a.hold, a.hold
    merged = os.path.join(CASES_DIR, cid, "merged.wav")
    if not os.path.exists(merged):
        raise SystemExit("找不到 %s" % merged)

    import soundfile as sf
    total = len(sf.read(merged)[0]) / 32000.0
    print("案例 %s 总时长 %.1fs" % (cid, total), flush=True)

    texts = []
    for st in [float(x) for x in a.starts.split(",") if x.strip()]:
        en = min(st + a.dur, total)
        if en - st < 10:
            continue
        wav = os.path.join(GS_OUT, "probe_%s_%03d.wav" % (a.spk.lower(), int(st)))
        txt, d = transcribe_seg(merged, st, en, wav)
        if not txt:
            continue
        # 落到最近的句末标点，避免半句话
        p = max(txt.rfind(c) for c in _END)
        if p > len(txt) * 0.5:
            txt = txt[:p + 1]
        tp = os.path.join(GS_OUT, "text_%s_%03d.txt"
                       % (a.spk.lower(), int(st)))
        open(tp, "w", encoding="utf-8").write(txt)
        print("  [%3.0fs~%3.0fs] %.1fs  %d字 -> %s"
              % (st, en, d, len(txt), os.path.basename(tp)), flush=True)
        texts.append(tp)
    print("共 %d 段测试文案" % len(texts), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
