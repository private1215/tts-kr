# -*- coding: utf-8 -*-
"""长音频漂移诊断：查「说着说着飙方言、语气就变了」。

## 为什么需要它

2026-10-07 员工反馈：「语音好了不少可是还是 ai 味比较重。语气不太自然，
然后这语音模板怎么说着说着飙方言啊，语气就变了」。

前半句（AI 味）是六维韵律的事，后半句（**说着说着就变了**）是个新症状：
它不是"整体上不像案例"，而是**同一条音频内部前后不一致**。整段均值类的指标
（A/B/E/F/sil/C/dur/peak/tone）**完全抓不到** —— 前后一平均，漂移就被抵消了。

## 它量什么

三档，从粗到细：

| 档 | 切法 | 抓什么 |
|---|---|---|
| **窗口档** | 等长 30 s 窗 | 有没有漂、漂多少 |
| **ASR 档** | 同上 + Whisper | **飙方言**：逐窗语言自动检测（ko/zh/en/ja）+ 非韩文字符占比 |
| **分块档** | 按**成品分块文件**切 | ⭐ **漂在块边界还是块内** —— 这是"要不要做 context stitching"的判据 |

🔴 **必须有对照组**：标杆音频自己也会漂（语速起伏、韵律变化）。
只报"我们漂了多少"没有意义，要报「我们 vs 案例，同样切法后的漂移之比」。
🔴 口径（2026-10-12）：案例是 **ElevenLabs 生成的**（不是真人录音），
   所以别写"真人换气" —— 它是 EL 产出的自然起伏，但**照样是合格的对照片**。

🔴 **分块档是核心**：如果「块间方差 >> 块内方差」，说明每个块独立起头、
块与块之间不连续 → 必须做 context stitching（把上一块尾部音频接到下一块的
prompt 里）。如果块内也漂，那是模型本身长程不稳，stitching 救不了。

用法（服务器，qwen 环境）：
    venv/bin/python tools/drift.py --prefix r23 --only 8197d9
    venv/bin/python tools/drift.py --prefix r23 --asr 1       # 开语言检测
    venv/bin/python tools/drift.py --case 1a6c20a60481        # 只量案例
"""
import os
import re
import sys
import glob
import json
import argparse

import numpy as np
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "app"))
from tools.fidelity import ltas_db                            # noqa: E402
from app.prosody_report import f0_of                          # noqa: E402

SR = 24000


def load(path, sr=SR):
    """任意音频 → 24 kHz 单声道 float32。ffmpeg 解码，与平台成品口径一致。

    ⚠️ 不能用 soundfile 直读 mp3：libsndfile 的 mp3 解码器与 ffmpeg 的
    libmp3lame 解码结果有细微差异，跨口径做减法会系统性偏 2.5~3.1 dB。
    """
    import subprocess
    import tempfile
    if not os.path.exists(path):
        return None
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=WORK)
    os.close(fd)
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path,
                        "-ac", "1", "-ar", str(sr), tmp], check=True,
                       timeout=300)
        x = sf.read(tmp, dtype="float32")[0]
        return np.asarray(x, dtype=np.float32)
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")
OUT_DIR = os.environ.get("TTS_OUT_DIR", "/mnt/tts/output")
WORK = "/mnt/tts/diag/driftwork"
FIVE = {
    "1a6c20": "1a6c20a60481", "00b6a4": "00b6a45ad6aa",
    "ba24c0": "ba24c0925311", "8197d9": "8197d9ac8a35",
    "6cdd25": "6cdd25bc75f2",
}
# 非韩文：中文汉字 / 日文假名 / 拉丁字母串。韩文是 가-힣。
_NON_KO = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
_LATIN = re.compile(r"[a-zA-Z]{3,}")


def windows(x, sr, win_s=30.0):
    n = int(win_s * sr)
    k = max(1, len(x) // n)
    return [x[i * n:(i + 1) * n] for i in range(k)]


def wc_metrics(w, sr):
    """一个窗口的指标。"""
    r = {}
    try:
        f = f0_of(w, sr)
        if f is not None and len(f) >= 8:
            r["f0_med"] = float(np.median(f))
            r["f0_std"] = float(np.std(f))
            r["f0_p85_p15"] = float(12 * np.log2(
                np.percentile(f, 85) / np.percentile(f, 15) + 1e-12))
    except Exception:
        pass
    W, H = int(0.05 * sr), int(0.025 * sr)
    n = (len(w) - W) // H
    if n > 10:
        e = np.sqrt(np.array([np.mean(w[i * H:i * H + W] ** 2)
                              for i in range(n)]) + 1e-12)
        dbe = 20 * np.log10(e + 1e-12)
        thr = np.percentile(dbe, 25) + 6.0
        r["voiced"] = float((dbe > thr).mean())
        r["rms_db"] = float(20 * np.log10(np.sqrt(np.mean(w ** 2)) + 1e-12))
    try:
        ldb, mf = ltas_db(w, sr)
        if ldb is not None and len(ldb):
            r["ltas"] = np.asarray(ldb, dtype=np.float64)
            lo = mf < 500
            hi = (mf >= 1000) & (mf < 6000)
            if lo.any() and hi.any():
                r["tilt"] = float(np.mean(ldb[lo]) - np.mean(ldb[hi]))
    except Exception:
        pass
    return r


# ---------------------------------------------------------------- ASR 档
_asr_model = None


def _asr_load():
    """直接加载 faster-whisper，拿得到 info.language（app.asr_check 只回文本）。"""
    global _asr_model
    if _asr_model is None:
        from faster_whisper import WhisperModel
        _asr_model = WhisperModel(
            os.environ.get("TTS_ASR_MODEL", "Systran/faster-whisper-small"),
            device="cpu", compute_type="int8", cpu_threads=4,
            local_files_only=True)
    return _asr_model


def _asr_segment(x, sr, lang="ko", detect=False):
    """转写一个片段。返回 (文本, 检测语言, 语言概率)。

    ⚠️ 两个坑（都是踩过的）：
    1. 不能把文件路径交给 transcribe —— av ≥19 移除了 metadata_errors 参数，
       会 TypeError 被吞掉 → 静默返回空串。必须自己 soundfile 读成 ndarray。
    2. 传 ndarray 时 faster-whisper **假定 16 kHz**，不会重采样。
       成品 24 kHz 直接丢进去 = 音高语速全错且不报错 → 必须手动重采样。
    """
    m = _asr_load()
    if sr != 16000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), 16000)
        x = resample_poly(np.asarray(x, dtype=np.float32),
                          16000 // g, int(sr) // g).astype(np.float32)
    lc = None if detect else (lang or None)
    segs, info = m.transcribe(x, language=lc, vad_filter=True, beam_size=1)
    txt = " ".join(s.text for s in segs).strip()
    return txt, getattr(info, "language", "?"), getattr(
        info, "language_probability", 0.0)


def asr_window(w, sr, idx, lang="ko", detect=False):
    """ASR 一个窗口，返回 dict。"""
    try:
        txt, lg, lp = _asr_segment(w, sr, lang, detect)
    except Exception as e:
        return dict(err=repr(e)[:90])
    tot = max(len(txt), 1)
    bad = len(_NON_KO.findall(txt)) + sum(len(m) for m in _LATIN.findall(txt))
    return dict(text=txt, lang=lg, lp=float(lp), bad=bad / tot)


def series_line(ms, key, bounds=None, fmt="{:6.1f}"):
    """逐窗序列字符串，`bounds` 是分块边界落在第几个窗（从 1 起）。

    🔴 有没有"边界跳变"只能看序列：块间 sd 会被块内平均稀释掉，
    必须直接看相邻窗的落差是不是恰好落在边界上。
    """
    out = []
    b = set(int(x) for x in (bounds or []))
    for i, m in enumerate(ms, 1):
        v = m.get(key)
        out.append(fmt.format(v) if v is not None else "   .")
        out.append("#" if i in b else " ")
    return "".join(out)


def summarize(ms, key):
    v = [m.get(key) for m in ms if m.get(key) is not None]
    if not v:
        return None
    a = np.asarray(v, dtype=np.float64)
    return dict(mean=float(a.mean()), std=float(a.std()),
                cv=float(a.std() / max(abs(a.mean()), 1e-9)),
                first=float(a[0]), last=float(a[-1]),
                rng=float(a.max() - a.min()))


def run(name, xs, sr, do_asr=False, lang="ko", detect=False, quiet=False,
        bounds=None):
    ms = [wc_metrics(w, sr) for w in xs]
    outs = []
    if do_asr:
        for i, w in enumerate(xs):
            outs.append(asr_window(w, sr, i, lang, detect))
    if not quiet:
        print(f"\n--- {name}：{len(xs)} 窗 ---")
        if bounds:
            print("  逐窗序列（# = 分块边界）")
            for k in ("f0_med", "voiced", "rms_db", "tilt"):
                print(f"    {k:<8}" + series_line(ms, k, bounds,
                                                 "{:6.2f}" if k != "f0_med"
                                                 else "{:6.1f}"))
        for k in ("f0_med", "f0_std", "voiced", "rms_db", "tilt"):
            s = summarize(ms, k)
            if s:
                print(f"  {k:<8} 首{s['first']:8.2f} 末{s['last']:8.2f} "
                      f"均值{s['mean']:8.2f} sd{s['std']:6.2f} "
                      f"极差{s['rng']:7.2f} CV{s['cv']:.3f}")
        base = ms[0].get("ltas")
        if base is not None:
            ds = []
            for m in ms:
                l = m.get("ltas")
                if l is not None and len(l) == len(base):
                    ds.append(float(np.mean(np.abs(l - base))))
            if ds:
                print(f"  ltas距第一窗 dB  首{ds[0]:.3f} 末{ds[-1]:.3f} "
                      f"最大{max(ds):.3f} 均值{np.mean(ds):.3f}")
        if do_asr:
            langs = [o.get("lang") for o in outs if o.get("lang")]
            bads = [o["bad"] for o in outs if "bad" in o]
            if bads:
                print(f"  非韩文占比      最大{max(bads):.3f} "
                      f"均值{np.mean(bads):.3f}")
            if langs:
                from collections import Counter
                print(f"  检测语言        {dict(Counter(langs))}")
            for i, o in enumerate(outs):
                if o.get("err"):
                    print(f"   窗{i+1} ASR 失败: {o['err']}")
                elif o.get("bad", 0) > 0.02 or o.get("lang") not in (
                        None, "ko"):
                    print(f"   窗{i+1} lang={o.get('lang')} "
                          f"p={o.get('lp',0):.2f} 非韩文{o.get('bad',0):.3f}: "
                          f"{o.get('text','')[:60]}")
    return ms, outs


# ------------------------------------------------------------- 分块档
def chunk_report(tag, files, sr, win=30.0):
    """逐个分块文件量指标，对比「块间方差」vs「块内方差」。"""
    per = []
    for f in files:
        x = load(f, sr)
        if x is None or len(x) < sr * 2:
            continue
        m = wc_metrics(x, sr)
        m["dur"] = len(x) / float(sr)
        per.append(m)
    if len(per) < 2:
        return None
    within = []
    for f in files:
        x = load(f, sr)
        if x is None or len(x) < sr * 2:
            continue
        ws = windows(x, sr, win)
        if len(ws) < 2:
            continue
        sub = [wc_metrics(w, sr) for w in ws]
        for k in ("f0_med", "voiced", "tilt"):
            v = [s.get(k) for s in sub if s.get(k) is not None]
            if len(v) >= 2:
                within.append((k, float(np.std(v))))
    res = dict(n=len(per), per=per, within=within)
    print(f"\n--- {tag} 分块档：{len(per)} 块 ---")
    print(f"  块时长: " + " ".join(f"{p['dur']:.0f}s" for p in per))
    for k in ("f0_med", "f0_std", "voiced", "tilt", "rms_db"):
        v = [p.get(k) for p in per if p.get(k) is not None]
        if len(v) >= 2:
            s = np.asarray(v, dtype=np.float64)
            wv = [w for kk, w in within if kk == k]
            wsd = float(np.mean(wv)) if wv else float("nan")
            ratio = (s.std() / wsd) if wsd > 1e-9 else float("nan")
            print(f"  {k:<8} 块间 sd {s.std():7.2f}  "
                  f"块内 sd {wsd:7.2f}  倍数 {ratio:5.2f}   "
                  + "  ".join(f"{x:.1f}" for x in v))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="r23")
    ap.add_argument("--only", default="")
    ap.add_argument("--case", default="", help="只量某个案例（给 cid）")
    ap.add_argument("--win", type=float, default=30.0)
    ap.add_argument("--asr", default="0")
    ap.add_argument("--detect", default="0",
                    help="1=语言自动检测（抓飙方言，比非韩文占比更直接）")
    ap.add_argument("--chunk", default="1", help="1=跑分块档")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    def concat(mp3s, out):
        import subprocess
        lst = out + ".txt"
        with open(lst, "w") as w:
            for f in mp3s:
                w.write("file '%s'\n" % f)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                        "-f", "concat", "-safe", "0", "-i", lst,
                        "-ac", "1", "-ar", str(SR), out], check=True)
        return sf.read(out, dtype="float32")[0]

    if a.case:
        x = concat([os.path.join(CASES_DIR, a.case, "merged.wav")],
                   os.path.join(WORK, f"case_{a.case[:6]}.wav"))
        run("案例 " + a.case[:6], windows(x, SR, a.win), SR,
            a.asr != "0", "ko", a.detect != "0")
        return

    do_asr = a.asr != "0"
    for short, cid in FIVE.items():
        if a.only and short not in a.only.split(","):
            continue
        prod = sorted(glob.glob(os.path.join(OUT_DIR, f"{a.prefix}-{short}",
                                             "*.mp3")))
        if not prod:
            continue
        px = concat(prod, os.path.join(WORK, f"{a.prefix}_{short}.wav"))
        cx = concat([os.path.join(CASES_DIR, cid, "merged.wav")],
                    os.path.join(WORK, f"case_{short}.wav"))
        print("\n" + "=" * 74)
        print(f"{short}  成品 {len(px)/SR:.1f}s / 案例 {len(cx)/SR:.1f}s"
              f"   窗口 {a.win:.0f}s")
        print(f"  ⚠️ 成品由 {len(prod)} 个分块拼成 —— 分块边界是漂移的头号嫌疑")
        # 分块边界落在第几个窗：按块时长累加，除以窗长取整
        bounds = []
        acc = 0.0
        for f in prod[:-1]:
            x = load(f, SR)
            acc += (len(x) / float(SR)) if x is not None else 0.0
            bounds.append(int(acc // a.win))
        print(f"  分块边界落在第 {bounds} 个窗之后（1-based 计数）")
        pm, _ = run("成品", windows(px, SR, a.win), SR, do_asr,
                    "ko", a.detect != "0", bounds=bounds)
        cm, _ = run("案例", windows(cx, SR, a.win), SR, do_asr,
                    "ko", a.detect != "0")
        print("  【漂移比 我们/案例】")
        for k in ("f0_med", "f0_std", "voiced", "tilt"):
            ps, cs = summarize(pm, k), summarize(cm, k)
            if ps and cs:
                sd = (ps["std"] / cs["std"]) if cs["std"] > 1e-9 else None
                print(f"  {k:<8} sd 我们{ps['std']:7.2f} vs 案例{cs['std']:7.2f}"
                      + (f"  倍数 {sd:.2f}" if sd else ""))
        if a.chunk == "1":
            chunk_report("成品", prod, SR, a.win)


if __name__ == "__main__":
    main()
