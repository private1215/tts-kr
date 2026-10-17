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

## 做法（v2，2026-10-17 按 z26 员工反馈修订）

z26 员工反馈：「整体有改善，但是没清理干净」。
审计（`tools/gruntaudit53.py`）发现原判据漏了两类：
    · 时长 120~250ms 的「嗯」（原上限 120ms）
    · 前后静音不足 0.15s 的（原门槛 0.15s）
男声原判据只抓 9 个，放宽后 58 个。但直接全放宽会误伤正常语音
（`tools/gruntdiff53.py`：多出的 49 个里只有 13 个是浊音「嗯」，
  其余 36 个 zcr 高/低频低，是正常语音碎片）。

→ **v2 判据**（三者同时满足）：
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

用法：
    python3 gruntclean53.py in.wav out.wav [--min-ms 20] [--max-ms 250]
                                      [--sil 0.08] [--no-voiced] [--dry]
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


def find_grunts(x, min_ms, max_ms, sil, voiced, peak_db=PEAK_DB):
    """返回 [(start_s, dur_s)]。
    voiced=True 时要求浊音特征（v2 判据）；False 回退 v1（只按时长+静音）。"""
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
        out.append((s, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--min-ms", type=float, default=20.0)
    ap.add_argument("--max-ms", type=float, default=250.0)
    ap.add_argument("--sil", type=float, default=0.08)
    ap.add_argument("--no-voiced", dest="voiced", action="store_false",
                    default=True, help="关闭浊音判据，回退 v1（只按时长+静音）")
    ap.add_argument("--dry", action="store_true", help="只报告不写文件")
    a = ap.parse_args()

    x = load16(a.src)
    g = find_grunts(x, a.min_ms, a.max_ms, a.sil, a.voiced)
    dur = len(x) / SR
    print("%s  %.1fs  极短填充音 %d 个 (%.2f/分钟)  [时长%d~%dms 静音>%.2f 浊音=%s]"
          % (os.path.basename(a.src), dur, len(g), len(g) / dur * 60,
             a.min_ms, a.max_ms, a.sil, "开" if a.voiced else "关"), flush=True)
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
