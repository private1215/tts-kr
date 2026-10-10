# -*- coding: utf-8 -*-
"""长文本打分：完整性 / 内容准确率(CER) / 时长 / 全程音色稳定性。

短文本相似度再高也不算数——线上是单篇 3000 字、成片几分钟。
这份脚本专查长文本特有的三类故障：
  1. 吞字 / 重复 —— CER 会突然变大（短文本上 GS 反而更低，所以必须单独测）
  2. 结尾截断  —— 尾巴那句没了，是最常见的长文本事故
  3. 中途跑调  —— 音色在头/中/尾不一致，说明 GPT 段发散了

用法：
    python tools/longtext_score.py
"""
import os
import re
import sys
import json
import subprocess

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")
os.environ.setdefault("HF_HOME", "/mnt/tts/models/asr")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from spkchk import read16, embed, CAMPPLUS  # noqa: E402
from app.asr_check import transcribe, cer  # noqa: E402

GT = open("/mnt/tts/diag/gs_long.txt", encoding="utf-8").read().strip()
REF = "/mnt/tts/projects/gpt-sovits/out/prompt_10s.wav"   # 이민정 案例原声 10s（测试区）

CAND = [
    ("GPT-SoVITS 微调 e30", "/mnt/tts/projects/gpt-sovits/out/gs_long_hh_v2pro_e30_s480.wav"),
    ("Qwen3TTS EL_female(当前生产)", "/mnt/tts/diag/gs-long-qwen.wav"),
]

# 同一份文本的真人原声（산일전기 案例），只用来当"时长应该差不多多久"的参照。
# ⚠️ 案例全文 4218 字，我们只取前 2902 字，所以不能直接拿案例总时长比 ——
#    要按真人语速折算成"同样 2902 字该多久"，否则 dur_ratio 会算成假的 0.77。
HUMAN = "/mnt/tts/data/cases/799e4d7060ce/merged.wav"
HUMAN_CHARS = 4218
# 参考语速：也可直接用本系统自己的验收口径（8.26 字/秒，来自 4046 字 / 489.8s）
OUR_CPS = 4046 / 489.8


CACHE = "/mnt/tts/diag/.asr_cache.json"


def _load_cache():
    try:
        return json.load(open(CACHE, encoding="utf-8"))
    except Exception:
        return {}


def asr_of(path):
    """转写很贵（几分钟音频跑一两分钟），按 路径+mtime 缓存，方便反复调口径。"""
    c = _load_cache()
    key = "%s|%s" % (path, os.path.getmtime(path))
    if key in c:
        return c[key]
    txt = transcribe(path, "ko")
    c[key] = txt
    try:
        json.dump(c, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception:
        pass
    return txt


def dur(path):
    if not os.path.exists(path):
        return None
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return None


def _norm(s):
    """去掉空格/标点差异再比。

    🔴 第一轮直接做子串匹配，结果两条都判成"尾句缺失"，可看 ASR 尾巴明明都
       说到了「이겁니다」——差别只是 회복해 줄/회복해줄、시장가/시장과 这类
       转写空格与音近字。不归一就会把转写噪声误报成截断事故。
    """
    s = re.sub(r"[\s,.\-·'\"“”?!]", "", s or "")
    return s


def tail_hit(gt, asr, n=25, tol=0.20):
    """结尾那句有没有被吃掉。

    🔴 踩过两次坑，别再走回头路：
      1. 直接子串匹配 → 两条都被误判"尾句缺失"，可 ASR 尾巴明明都说到「이겁니다」，
         差别只是 회복해 줄/회복해줄、시장가/시장과 这类转写噪声。
      2. 归一化后取 16 字硬匹配 → 仍因 1 个音近字（가/과）误判。
    现在改成：归一化后取**末尾 25 字算 CER**，<= tol 判完整。
    这样"转写噪声"和"真截断"能分开，且失败时会把两边的尾巴都打出来供人核对。
    """
    a, b = _norm(gt)[-n:], _norm(asr)[-n:]
    if not a or not b:
        return False
    try:
        c = cer(a, b)
    except Exception:
        return a[-12:] == b[-12:]
    return (c is not None) and (c <= tol)


def repeat_hit(asr, k=12, min_n=3):
    """复读检测：长文本发散最典型的表现就是某句被反复念。

    取所有长度 k 的滑窗子串，出现 >= min_n 次就认定复读。
    """
    s = _norm(asr)
    if len(s) < k:
        return 0, ""
    cnt = {}
    for i in range(len(s) - k + 1):
        sub = s[i:i + k]
        cnt[sub] = cnt.get(sub, 0) + 1
    rep = [(v, sub) for sub, v in cnt.items() if v >= min_n]
    if not rep:
        return 0, ""
    rep.sort(reverse=True)
    return rep[0][0], rep[0][1]


def win_emb(sess, ref, x, s, win):
    """取一段窗口算相似度。x 是 (1, n) 的 2D tensor，必须按 dim=1 切。"""
    seg = x[:, s:s + win]
    if seg.shape[1] < 16000:
        return float("nan")
    e = embed(sess, seg)
    if e is None:
        return float("nan")
    return float(np.dot(ref, e))


def main():
    print("目标文案 %d 字" % len(GT))
    print("尾句:", GT[-40:])
    hd = dur(HUMAN)
    hcps = HUMAN_CHARS / hd if hd else 0        # 真人语速（字/秒）
    exp_h = len(GT) / hcps if hcps else 0       # 同样字数真人该说多久
    exp_o = len(GT) / OUR_CPS                   # 我们自己的验收口径
    print("真人原声: 全文 %d 字 / %.1fs -> %.2f 字/秒" % (HUMAN_CHARS, hd or -1, hcps))
    print("按真人语速折算，%d 字应约 %.1fs" % (len(GT), exp_h))
    print("按本系统验收口径(%.2f 字/秒)，应约 %.1fs" % (OUR_CPS, exp_o))
    print()

    sess = ort.InferenceSession(CAMPPLUS, providers=["CPUExecutionProvider"])
    ref = embed(sess, read16(REF))
    print("参照音色: %s" % REF)
    print()

    for name, p in CAND:
        if not os.path.exists(p):
            print("%-28s 未生成（%s）" % (name, p))
            continue
        d = dur(p)
        try:
            asr = asr_of(p)
            c = cer(GT, asr)
        except Exception as e:
            print("%-28s ASR ERR %s" % (name, e))
            asr, c = "", None
        try:
            x = read16(p)
            n = x.shape[1]          # 🔴 read16 返回的是 (1, n)，len() 拿到的是 1
            fe = embed(sess, x)
            full = float(np.dot(ref, fe)) if fe is not None else float("nan")
            # 头/中/尾各取 60 秒，看有没有中途跑掉
            win = int(60 * 16000)
            parts = []
            for i in range(3):
                s = min(int(n * i / 3.0), max(0, n - win))
                parts.append(win_emb(sess, ref, x, s, win))
        except Exception as e:
            print("%-28s EMB ERR %s" % (name, e))
            full, parts = float("nan"), [float("nan")] * 3

        ok = tail_hit(GT, asr)
        rn, rs = repeat_hit(asr)
        print("%-28s" % name)
        print("   时长 %.1fs   语速 %.2f 字/秒   真人口径 %.3f   本系统口径 %.3f"
              % (d or -1, len(GT) / d if d else -1,
                 (d / exp_h) if (d and exp_h) else -1,
                 (d / exp_o) if (d and exp_o) else -1))
        print("   CER %.4f   转写字数 %d（原文 %d）" % (c if c is not None else -1, len(asr or ""), len(GT)))
        print("   结尾完整: %s" % ("✅" if ok else "❌ 尾句缺失/被吞"))
        if not ok:
            print("     原文尾巴: ...%s" % _norm(GT)[-25:])
            print("     ASR尾巴: ...%s" % _norm(asr)[-25:])
        print("   ASR 尾巴: ...%s" % (asr or "")[-60:])
        print("   复读: %s" % ("无" if rn == 0 else "⚠️ 「%s」出现 %d 次" % (rs, rn)))
        print("   音色 全程 %.4f | 头 %.4f 中 %.4f 尾 %.4f  漂移 %.4f"
              % (full, parts[0], parts[1], parts[2],
                 (max(parts) - min(parts)) if not any(np.isnan(parts)) else float("nan")))
        print()


if __name__ == "__main__":
    main()
