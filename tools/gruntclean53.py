# -*- coding: utf-8 -*-
"""gruntclean53.py —— 移除「极短填充音」（员工说的「特别特别短的嗯，超级短」）。

## 问题（2026-10-17 z25 员工反馈 + 离线定位）

员工原话：「第一道题 A 特别特别短的 嗯 超级短 存在」（两段男声都有）。
离线定位（16k 统一口径，`tools/grunt53b.py`）：

    裸输出 yohan 参考   12 个 (1.30/分钟)
    裸输出 EL 参考       3 个 (0.33/分钟)
    EL参考+fixchain      5 个 (0.54/分钟)
    现役成品 mp3         9 个 (0.96/分钟)
    男 EL 标杆           **0 个**

→ 「嗯」是 **OmniVoice 模型自发生成**（不是后处理引入：裸输出 12 → fixchain 后 9）。
→ 换 EL 参考能减少 50-75% 但消不掉；EL 本身是 0。
→ 员工明确听得见，且与她说 N 次的「断句有异声」吻合。

## 做法（v3，2026-10-17 按员工反馈二次修订）

z27 员工反馈（方向对，但没清干净）：
> 「嗯还有，但是都减少了很小一部分，说明整个嗯的发音，每个嗯清了一部分
>   没有全部清理」

`tools/gruntext53.py` 实测证实：18 个种子**全部**能向左/右各扩展 20~35ms，
即「嗯」有起音和收尾，能量低于 `db > p35` 门槛，被判成静音漏掉。
例如 t=55.87s：种子 120ms，实际 170ms，左右各剩 25ms。

→ **v3 在 v2 判据基础上加「种子扩展」**：以检测到的段为中心，
  向左右扩展到「不低于 种子峰值 − EXT_DB」的边界（允许 3 帧断续），
  把整个「嗯」（起音+持续+收尾）一次性替换掉。

v2 判据（三者同时满足）：
    · 时长 20~250ms（5ms 帧 RMS，db > p35 判发声）
    · 前后静音都 > 0.08s（保证是孤立填充音，不是正常语流中的短音节）
    · **浊音特征**：过零率 < 0.12 且 300Hz 以下能量占比 > 0.45
      （「嗯」是鼻浊音：低频饱满、无高频摩擦；正常韩语音节 zcr 通常 >0.14）
    · 峰值 > -32dB（有实质能量）

用**等长静音**替换（总时长不变），5ms 淡入淡出避免 click。

🔴 为什么用静音替换而不是删段：删段会改变总时长、破坏与其他指标的配比；
   静音替换只去掉那一声「嗯」，其余一个样本都不动。

🔴 为什么必须加浊音判据：单纯放宽时长/静音会把正常语音碎片也算进来
   （男声 58 个里 36 个是正常语音），删了会切碎词语。

🔴 为什么必须加种子扩展：`db > p35` 是**相对**门槛，「嗯」的起音/收尾
   能量低于它就会被判成静音，导致每个「嗯」都剩一截（员工原话：
   「每个嗯清了一部分没有全部清理」）。

用法：
    python3 gruntclean53.py in.wav out.wav [--min-ms 20] [--max-ms 250]
                                      [--sil 0.08] [--ext-db 12] [--no-voiced]
                                      [--dry]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
PEAK_DB = -32.0
ZCR_MAX = 0.12
LOW_RATIO = 0.45
EXT_DB = 12.0


def load16(p, sr=SR):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def fade(a, ms=5):
    n = int(SR * ms / 1000)
    if len(a) < 2 * n:
        return a
    w = np.linspace(0, 1, n)
    a = a.copy()
    a[:n] *= w
    a[-n:] *= w[::-1]
    return a


def voiced_quality(a, sr=SR):
    """浊音判定：过零率 + 300Hz 以下能量占比。「嗯」是鼻浊音，两者都极端。"""
    if len(a) < 40:
        return 1.0, 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(a)))) / 2)
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    fr = np.fft.rfftfreq(len(a), 1.0 / sr)
    low = float(sp[fr < 300].sum() / (sp.sum() + 1e-12))
    return zcr, low


def extend_sil(ci, cj, prev_end, next_start, max_each_ms=120):
    """把种子段扩展到「两侧静音边界」——即整个孤立小岛全部抹平。

    判据已确认该段夹在两段静音之间（前后静音都 > sil）。既然它本来
    就该是静音，就把 [prev_end, next_start] 整段替换掉，不用猜
    「嗯」的起音/收尾到底延伸多少 dB（v3 的 ext_db 就是在猜这个）。

    两侧各最多多扩 max_each_ms，避免静音很长时把大段区域整块改电平
    （会在原本安静的地方引入台阶）。

    ci/cj/prev_end/next_start 都是帧号。
    """
    cap = int(max_each_ms / 1000 / FRAME)
    li = max(prev_end, ci - cap)
    ri = min(next_start, cj + cap)
    return li * FRAME, (ri - li) * FRAME


def extend_seed(x, s, d, pk, db, ext_db=EXT_DB, pad_ms=400, max_gap=3):
    """把种子段向左右扩展到「该嗯的完整边界」。

    「嗯」有起音/收尾，能量低于 p35 门槛会被判成静音，导致只清中间一截。
    这里从种子峰值往下 ext_db 以内都算同一个「嗯」，允许 max_gap 帧断续。

    返回 (new_start_s, new_dur_s)。
    """
    if ext_db <= 0:
        return s, d
    pad = int(pad_ms / 1000 / FRAME)
    ci = int(round(s / FRAME))
    cj = int(round((s + d) / FRAME))
    i0 = max(0, ci - pad)
    i1 = min(len(db), cj + pad)
    thr = pk - ext_db
    li, gap = ci, 0
    while li > i0 and gap <= max_gap:
        if db[li - 1] >= thr:
            li -= 1
            gap = 0
        else:
            gap += 1
            li -= 1
    ri, gap = cj, 0
    while ri < i1 and gap <= max_gap:
        if db[ri] >= thr:
            ri += 1
            gap = 0
        else:
            gap += 1
            ri += 1
    return li * FRAME, (ri - li) * FRAME


def find_grunts(x, min_ms, max_ms, sil, voiced, peak_db=PEAK_DB,
                ext_db=EXT_DB, ext_mode="sil"):
    """返回 [(start_s, dur_s)]，已做种子扩展。

    ext_mode:
        "sil" —— 向两侧扩展到静音边界（v4，最彻底，默认）
        "db"  —— 从峰值往下 ext_db 以内算同一个「嗯」（v3）
        "off" —— 不扩展（v2）
    """
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    voiced_f = db > np.percentile(db, 35)
    segs, i = [], 0
    while i < len(voiced_f):
        if voiced_f[i]:
            j = i
            while j < len(voiced_f) and voiced_f[j]:
                j += 1
            segs.append((i * FRAME, (j - i) * FRAME, float(db[i:j].max())))
            i = j
        else:
            i += 1
    out = []
    lo, hi = min_ms / 1000.0, max_ms / 1000.0
    for k in range(1, len(segs) - 1):
        s, d, pk = segs[k]
        if not (lo <= d <= hi):
            continue
        pe = segs[k - 1][0] + segs[k - 1][1]
        nx = segs[k + 1][0]
        if (s - pe) <= sil or (nx - (s + d)) <= sil:
            continue
        if pk < peak_db:
            continue
        if voiced:
            a = x[int(s * SR):int((s + d) * SR)]
            zcr, low = voiced_quality(a)
            if not (zcr < ZCR_MAX and low > LOW_RATIO):
                continue
        ns, nd = s, d
        if ext_mode == "sil":
            # segs 存的是 (起始秒, 时长秒, 峰值dB)，这里换算成帧号
            ci = int(round(s / FRAME))
            cj = int(round((s + d) / FRAME))
            pe_f = int(round((segs[k - 1][0] + segs[k - 1][1]) / FRAME))
            nx_f = int(round(segs[k + 1][0] / FRAME))
            cand = extend_sil(ci, cj, pe_f, nx_f)
            # 🔴 扩展后必须仍是「浊音小岛」：整岛抹平会把「嗯」前后
            #    的静音一起吃掉，若静音里混着辅音/摩擦音，扩展后的
            #    zcr 会明显升高。实测女声 3 处扩展后 zcr 从 <0.12 升到
            #    0.18~0.21（包进了摩擦音），这种必须回退到保守的 db 模式。
            if check_voiced(x, cand[0], cand[1],
                            zcr_max=ZCR_MAX + 0.03, low_min=LOW_RATIO - 0.08):
                ns, nd = cand
            elif ext_db > 0:
                ns, nd = extend_seed(x, s, d, pk, db, ext_db)
        elif ext_mode == "db" and ext_db > 0:
            ns, nd = extend_seed(x, s, d, pk, db, ext_db)
        out.append((ns, nd))
    return out


def check_voiced(x, s, d, zcr_max, low_min):
    """校验区间是否仍以鼻浊音为主。"""
    a = x[int(s * SR):int((s + d) * SR)]
    zcr, low = voiced_quality(a)
    return zcr < zcr_max and low > low_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--min-ms", type=float, default=20.0)
    ap.add_argument("--max-ms", type=float, default=250.0)
    ap.add_argument("--sil", type=float, default=0.08)
    ap.add_argument("--ext-db", type=float, default=EXT_DB,
                    help="种子向左右扩展的门限（峰值往下这么多 dB 以内算同一个「嗯」）")
    ap.add_argument("--ext-mode", default="sil", choices=["sil", "db", "off"],
                    help="sil=扩展到两侧静音边界（v4，最彻底，默认）；"
                         "db=从峰值往下 ext-db 以内（v3）；off=不扩展（v2）")
    ap.add_argument("--no-sil-gate", action="store_true",
                    help="不要求前后静音（抓紧贴发言的「嗯」，需谨慎："
                         "实测 EL 句末鼻音比我们多，默认关闭）")
    ap.add_argument("--no-voiced", dest="voiced", action="store_false",
                    default=True, help="关闭浊音判据，回退 v1（只按时长+静音）")
    ap.add_argument("--dry", action="store_true", help="只报告不写文件")
    a = ap.parse_args()

    x = load16(a.src)
    sil = 0.0 if a.no_sil_gate else a.sil
    g = find_grunts(x, a.min_ms, a.max_ms, sil, a.voiced,
                    ext_db=a.ext_db, ext_mode=a.ext_mode)
    dur = len(x) / SR
    print("%s  %.1fs  极短填充音 %d 个 (%.2f/分钟)  "
          "[时长%d~%dms 静音>%s 浊音=%s 扩展=%s]"
          % (os.path.basename(a.src), dur, len(g), len(g) / dur * 60,
             a.min_ms, a.max_ms,
             ("%.2f" % sil) if sil > 0 else "不限",
             "开" if a.voiced else "关", a.ext_mode),
          flush=True)
    for s, d in g:
        zcr, low = voiced_quality(x[int(s * SR):int((s + d) * SR)])
        print("   t=%7.2fs  %3.0fms  zcr=%.3f 低频=%.2f" % (s, d * 1000, zcr, low),
              flush=True)
    if a.dry or not g:
        if not g and not a.dry:
            sf.write(a.dst, x, SR)
            print("无需处理，已原样写出", flush=True)
        return 0

    y = x.copy()
    for s, d in g:
        i0 = int(s * SR)
        i1 = int((s + d) * SR)
        # 等长静音替换 + 5ms 淡入淡出（用邻近静音的电平，不是数字零）
        n = i1 - i0
        # 取该段前的静音电平做底（避免突然掉到 -inf）
        pre = x[max(0, i0 - int(0.2 * SR)):i0]
        floor = float(np.sqrt((pre ** 2).mean())) if len(pre) else 1e-4
        seg = np.full(n, floor, dtype=np.float32)
        y[i0:i1] = fade(seg)
    sf.write(a.dst, y, SR)
    print("-> %s  已替换 %d 处（等长静音，总时长不变）" % (a.dst, len(g)),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
