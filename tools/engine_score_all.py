# -*- coding: utf-8 -*-
"""引擎总评：**音色 + 语调/韵律 + 内容完整性** 一次跑完。

🔴 为什么必须有这个脚本
------------------------
2026-10-12 的教训：只测音色会得出**方向性错误**的结论。
    GS 微调女声音色 0.9426 全场最高 —— 但同一次推理的音频
    **覆盖率只有 0.150、CER 0.808**，GPT 自回归中后段跑飞，
    60 秒的文案只出了 33.6 秒，其中大半是胡话。
    只看音色会选它上线；加上完整性这一栏立刻排除。

所以三个维度必须**同表并列**，任何一个不过关就不能上线：
    · 音色      campplus 余弦（参照 = 案例原声）
    · 语调/韵律 音高中位 / 句内动态 / 音高跨度 / 能量动态 / 语速CV /
                句尾STD / 停顿熵 / 句尾降幅 / 语速 / 停顿密度
    · 完整性    ASR 覆盖率 / CER / 冗余率（>0.3 = 重复念）

用法
----
    python tools/engine_score_all.py --spec tools/score_f.json
    # spec: [{"name","path","is_male","ref":true,"text_file"}, ...]
    # 标 ref:true 的那条是案例原声，其余全部与它比。

依赖：qwen3-tts 的 venv（parselmouth / faster-whisper / onnxruntime /
torchaudio.kaldi 全在里面），不要拿 cosy 的 venv 跑（缺 parselmouth）。
"""
import argparse
import difflib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from app import prosody_report as PR                       # noqa: E402
from app import proso                                      # noqa: E402
from app import asr_check as ASR                           # noqa: E402
import complete_chk as CC                                  # noqa: E402
import prosody_cmp as PC                                   # noqa: E402
import spkchk                                              # noqa: E402

CAMPPLUS = os.environ.get(
    "TTS_CAMPPLUS", "/mnt/tts/models/Fun-CosyVoice3-0.5B/campplus.onnx")
_SESS = None
COV_MIN = 0.85        # 覆盖率低于此值判定吞字
RED_MAX = 0.30        # 冗余率高于此值判定重复


def _sess():
    global _SESS
    if _SESS is None:
        import onnxruntime
        _SESS = onnxruntime.InferenceSession(
            CAMPPLUS, providers=["CPUExecutionProvider"])
    return _SESS


def _wav16(path):
    """一律转 16k mono wav。campplus 与 ASR 都要 16k。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", tmp.name], check=True)
    return tmp.name


def timbre(path):
    """campplus 嵌入（多窗平均，避免官方随机截窗）。"""
    w = _wav16(path)
    try:
        sp = spkchk.read16(w)
        return spkchk.embed(_sess(), sp)
    finally:
        os.unlink(w)


def score_one(path, is_male, ref_text=None):
    """返回 {韵律 + 完整性指标}；emb 单独存（要跟参照比）。

    ⚠️ 韵律一律走 `prosody_cmp.measure`（**Praat 口径**）。
       第一版这里自己拼了一遍并调用 `PR.measure`，那个走的是 librosa.pyin，
       和 Praat 的 med / drop 混在同一张表里 —— 跨口径数字不可比。
    """
    w16 = _wav16(path)
    try:
        m = PC.measure(w16, is_male, ref_text, name=os.path.basename(path))
        # 完整性（ASR 要 16k，w16 已经是）
        hyp = ASR.transcribe(w16, language="ko")
        h = ASR.normalize(hyp)
        r = ASR.normalize(ref_text) if ref_text else ""
        m["cov"] = CC.coverage(r, h) if r else None
        m["cer"] = ASR.cer(r, h) if r else None
        m["red"] = ((len(h) - len(r)) / float(len(r))) if r else None
        m["emb"] = spkchk.embed(_sess(), spkchk.read16(w16))
    finally:
        os.unlink(w16)
    return m


KEYS = [("spk", "音色", 3), ("cov", "覆盖率", 3), ("cer", "CER", 3),
        ("med", "音高中位", 1), ("A", "句内动态", 2), ("B", "音高跨度", 2),
        ("C", "能量动态", 2), ("D", "语速CV", 3), ("E", "句尾STD", 2),
        ("F", "停顿熵", 3), ("drop", "句尾降幅", 2), ("sps", "语速", 2),
        ("prate", "停顿/分", 1)]


def _f(v, d):
    return ("%.*f" % (d, v)) if isinstance(v, (int, float)) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    spec = json.load(open(a.spec, encoding="utf-8"))
    ref_txt = None
    for it in spec:
        if it.get("ref") and it.get("text_file"):
            ref_txt = open(it["text_file"], encoding="utf-8").read()
            break

    rows = []
    for it in spec:
        p = it["path"]
        if not os.path.exists(p):
            print("跳过（不存在）:", p)
            continue
        m = score_one(p, it.get("is_male"), ref_txt)
        m["name"] = it.get("name") or os.path.basename(p)
        m["_ref"] = bool(it.get("ref"))
        rows.append(m)

    if not rows:
        return 1
    ref = next((r for r in rows if r["_ref"]), rows[0])
    for r in rows:
        r["spk"] = float(np.dot(r["emb"], ref["emb"]))

    print("\n=== 绝对值（参照 = 案例原声）===")
    print("%-30s" % "样本" + "".join(" %s" % l.rjust(7) for _, l, _ in KEYS))
    for r in rows:
        line = "%-30s" % r["name"][:30]
        for k, _, d in KEYS:
            line += " %s" % _f(r.get(k), d).rjust(7)
        print(line + ("   ←参照" if r is ref else ""))

    print("\n=== 相对参照偏差 ===")
    print("%-30s" % "样本" + "".join(" %s" % l.rjust(7) for _, l, _ in KEYS))
    for r in rows:
        if r is ref:
            continue
        line = "%-30s" % r["name"][:30]
        for k, _, d in KEYS:
            v, rv = r.get(k), ref.get(k)
            if isinstance(v, (int, float)) and isinstance(rv, (int, float)):
                line += " %+*.*f" % (7, d, v - rv)
            else:
                line += " " * 7 + "-"
        print(line)

    print("\n=== 判定 ===")
    for r in rows:
        if r is ref:
            continue
        bad = []
        if r.get("cov") is not None and r["cov"] < COV_MIN:
            bad.append("吞字(覆盖率%.2f)" % r["cov"])
        if r.get("red") is not None and r["red"] > RED_MAX:
            bad.append("重复(冗余%+.2f)" % r["red"])
        if r.get("spk") is not None and r["spk"] < 0.90:
            bad.append("音色偏低(%.4f)" % r["spk"])
        print("  %-30s %s" % (r["name"][:30],
                              " / ".join(bad) if bad else "通过"))

    if a.json_out:
        out = [dict([(k, r.get(k)) for k, _, _ in KEYS] +
                    [("name", r["name"])]) for r in rows]
        for o in out:
            o.pop("emb", None)
        json.dump(out, open(a.json_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("\n已写", a.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
