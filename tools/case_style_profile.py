# -*- coding: utf-8 -*-
"""🔴 案例「风格/情绪画像」提取 —— 把案例语音里的情绪拆成可复刻的数字。

## 为什么需要它
用户（2026-10-12）明确澄清了情绪的方向：
> 「主要是拆解案例语音里的情绪，然后复刻，可以拿来用，
>   不是一定要用哪种情绪来生成语音」

也就是说：**情绪不是一个让员工选的下拉框，而是案例自带的东西，我们要还原它。**
而"情绪"落到可操作的层面，就是一组韵律统计量：音高怎么走、句尾怎么收、
哪里停、停多久、轻重怎么摆。这些全都能量，也全都能通过后处理对齐。

现状的缺口：`app/server.py` 的 `proso_targets()` 只用了 **A / E 两个维度**，
而且是**按性别取中位数**——不同说话人（SOLA / yohan / elara）被混成一条线。
这既太粗，也做不到「复刻这一条案例的味道」。

## 输出什么
按**说话人组**（campplus 嵌入聚类，见 mkcase_voices.embed_groups）聚合出
一组目标值，同时也保留每条案例自己的画像（供"对标单条案例"时用）：

    med   音高中位(Hz)      A  句内音高动态(semi)
    B     音高跨度(semi)    C  能量动态(dB)
    D     语速CV            E  句尾斜率标准差
    F     停顿熵(0~1)       drop 句尾定向降幅(semi, 负=降)
    sps   语速(音节/秒)     prate 停顿密度(次/分)

## 用法
    python tools/case_style_profile.py                  # 全库，按说话人组
    python tools/case_style_profile.py --by gender      # 退化成按性别
    python tools/case_style_profile.py --out prof.json

⚠️ 依赖 qwen3-tts 的 venv（要 parselmouth）。
⚠️ 参照口径统一走 Praat（app/proso），不混 pyin（跨口径分数不可比）。
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from app import prosody_report as PR                       # noqa: E402
from app import proso                                      # noqa: E402
import prosody_cmp as PC                                   # noqa: E402

CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")
# 只把「标杆平台」的案例当目标：我们自己的输出不能当自己的尺子
CASE_SOURCE = set(filter(None, os.environ.get(
    "TTS_PROSO_CASE_SOURCE", "elevenlabs").split(",")))

DIMS = ["med", "A", "B", "C", "D", "E", "F", "drop", "sps", "prate"]

# 每条案例只分析前 MAX_SEC 秒（韵律统计已收敛，见 profile_of 注释）。
# 0 或负数 = 不截断（会非常慢，36 条必超时）。
MAX_SEC = float(os.environ.get("CASE_PROFILE_SEC", "180"))


CASES_JSON = os.environ.get("TTS_CASES_JSON", "/mnt/tts/data/cases.json")


def list_cases():
    """返回 [(cid, meta)]，只收白名单 source 且有音频的案例。

    ⚠️ 元数据不在案例目录里（那里只有 merged.wav / seg*.wav / seg*.txt），
    而在 `/mnt/tts/data/cases.json` 这个总表（dict[cid] -> meta）。
    一开始按目录找 meta.json 会一条都收不到。
    """
    if not os.path.exists(CASES_JSON):
        raise SystemExit("找不到 %s" % CASES_JSON)
    allc = json.load(open(CASES_JSON, encoding="utf-8"))
    out = []
    for cid, meta in allc.items():
        if not os.path.exists(os.path.join(CASES_DIR, cid, "merged.wav")):
            continue
        src = (meta.get("source") or "").lower()
        if CASE_SOURCE and src not in CASE_SOURCE:
            continue
        out.append((cid, meta))
    return sorted(out, key=lambda t: t[0])


def case_text(cid, meta):
    """拼接该案例所有 segment 的文案（用于算 sps）。"""
    buf = []
    for s in (meta.get("segments") or []):
        tp = os.path.join(CASES_DIR, cid, s.get("text") or "")
        if os.path.exists(tp):
            buf.append(open(tp, encoding="utf-8").read().strip())
    return " ".join(buf) if buf else None


def profile_of(wav, is_male, text=None, max_sec=MAX_SEC):
    """单条音频的画像。text 给了才能算 sps。

    ⚠️ 为什么要 max_sec 截断：案例动辄 4~8 分钟，而韵律统计（句内动态、
    句尾多样性、停顿分布）在前几分钟就收敛了。全篇分析会把
    `sent_windows` 的逐帧 Python 循环放大到十几分钟一条，36 条必超时
    （实测第一版跑满 900s 被杀，一条都没出）。
    截断后单条 ~20 秒，全库十几分钟，且横向可比性不受影响
    ——所有案例都是「前 max_sec 秒」，口径一致。
    """
    x, sr = sf.read(wav, dtype="float32")
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    total_dur = len(x) / float(sr)
    if max_sec and max_sec > 0 and len(x) > int(max_sec * sr):
        x = x[:int(max_sec * sr)]
    if int(sr) != 24000:
        import librosa
        x = librosa.resample(x.astype(np.float64), orig_sr=sr,
                             target_sr=24000).astype(np.float32)
        sr = 24000
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, x, sr, subtype="PCM_16")
    try:
        m = PC.measure(tmp.name, is_male, None)     # 先不给 text，避免 sps 算错
    finally:
        os.unlink(tmp.name)
    # 🔴 sps 必须用**整篇时长**作分母：上面只分析了前 max_sec 秒，
    #    拿整篇字数除以截断时长会把语速算成实际值的 (total/max_sec) 倍
    #    （实测 13.65 vs 真实约 7.4）。等价于 S / total_dur。
    syl = PC._syllables(text) if text else None
    m["sps"] = (syl / total_dur) if (syl and total_dur > 0) else None
    return m


def med_of_vals(v):
    v = [x for x in v if isinstance(x, (int, float))]
    if not v:
        return None
    return float(np.median(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", default="speaker", choices=["speaker", "gender"],
                    help="speaker=按声音聚类分组（推荐）；gender=按性别")
    ap.add_argument("--out", default="/mnt/tts/data/cases/style_profile.json")
    ap.add_argument("--min-cases", type=int, default=1)
    a = ap.parse_args()

    cases = list_cases()
    print("白名单案例 %d 条（source=%s）" % (len(cases), sorted(CASE_SOURCE)))

    # 分组：speaker 用嵌入聚类，gender 用元数据里的性别
    groups = {}
    if a.by == "speaker":
        try:
            import mkcase_voices as MK
            # ⚠️ embed_groups 要的是 {cid: case_dict}，不是 list
            cdict = {c: m for c, m in cases}
            gmap, gmeta = MK.embed_groups(cdict)
            for k, v in gmap.items():
                if len(v) >= a.min_cases:
                    groups["%s(%s)" % (k[0] or "unnamed", k[1] or "?")] = list(v)
                    print("  组 %s：%d 条，组内最低相似度 %.4f"
                          % (k, len(v), (gmeta.get(k) or {}).get("sim_min", -1)))
        except Exception as ex:
            print("嵌入聚类失败，退回按名字分组: %r" % (ex,))
            a.by = "name"
    if a.by != "speaker":
        for cid, meta in cases:
            k = "%s" % (meta.get("gender") or "unknown")
            groups.setdefault(k, []).append(cid)

    result = {"dims": DIMS, "groups": {}, "cases": {}}
    for gname, cids in sorted(groups.items()):
        print("\n=== 组 %s（%d 条）===" % (gname, len(cids)))
        profs = {}
        is_male = None
        for cid in cids:
            meta = dict(cases).get(cid, {})
            g = (meta.get("gender") or "").lower()
            if g == "male":
                is_male = True
            elif g == "female":
                is_male = False
            wav = os.path.join(CASES_DIR, cid, "merged.wav")
            txt = case_text(cid, meta)
            try:
                p = profile_of(wav, is_male, txt)
            except Exception as ex:
                print("  %s 失败 %r" % (cid[:12], ex))
                continue
            profs[cid] = p
            result["cases"][cid] = {k: p.get(k) for k in DIMS}
        if not profs:
            continue
        agg = {k: med_of_vals([p.get(k) for p in profs.values()]) for k in DIMS}
        agg["n"] = len(profs)
        agg["is_male"] = is_male
        result["groups"][gname] = agg
        print("  " + "  ".join("%s=%s" % (k, ("%.3f" % agg[k])
                                          if isinstance(agg[k], float) else "-")
                               for k in DIMS))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(result, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
