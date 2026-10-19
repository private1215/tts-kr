# -*- coding: utf-8 -*-
"""tailclean53c.py —— 清理「句首紧贴型」低能量鼻音碎片。

## 背景（19:45 复核结论）

EL 同文本对照（d172b8c4efb3）证明：
    · 真孤立碎片（前后静音>100ms）：我们 0 vs EL 2 → 我们没有更多
    · 句末粘连尾巴（tailclean 口径）：我们 49 vs EL 36 → 多 36%，tailclean 已清
    · **句首紧贴型碎片（后间隔 20~50ms）**：我们 21 vs EL 12，多 73%（20~50ms 段更悬殊）

`tailclean53b` 要求后间隔>50ms，抓不到「紧贴下一句开头」的碎片 → 本工具补这一缺口。

## 与既有工具的分工（按 gap_pre / gap_nxt）

| 工具 | gap_pre | gap_nxt | 说明 |
|---|---|---|---|
| tailclean53/53b | <30ms | >50ms | 句末粘连尾巴（紧跟前段） |
| **tailclean53c** | **>20ms** | **20~50ms** | **句首紧贴碎片（紧贴下一句开头）** |
| 不清 | >100ms | >100ms | 真孤立，EL 也有，属正常现象 |

## 判据

    · 时长 5~80ms
    · 峰值 −45 ~ −25dB（低能量；正常韵尾多在 −12~−20dB，天然被排除）
    · zcr < 0.15 且 300Hz 以下能量 > 0.45（浊音/鼻音）
    · 前间隔 > 20ms（区别于句末粘连）
    · 20ms <= 后间隔 <= 50ms（紧贴下一句开头，但有一点点间隙）

替换为等长静音（填 0）+ 5ms fade，总时长不变。

⚠️ 高风险：紧挨主语音，误判会吃掉下一句开头的辅音。
   必须先对 EL 跑 --dry，确认 EL 的命中也是同类碎片（数量应≈12），
   且清理后 EL 的 CER/听感不塌，才能用于我们。

用法：
    python3 tailclean53c.py in.wav [out.wav] [--dry]
    python3 tailclean53c.py el.wav --dry      # 只看 EL 命中
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
VTH = -40.0


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
    if len(a) < 16:
        return 1.0, 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(a)))) / 2)
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    fr = np.fft.rfftfreq(len(a), 1.0 / sr)
    low = float(sp[fr < 300].sum() / (sp.sum() + 1e-12))
    return zcr, low


def find_head_clusters(x, gap_pre_min=0.020, gap_nxt_min=0.020,
                       gap_nxt_max=0.050):
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    v = db > VTH
    raw, i = [], 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            raw.append((i, j))
            i = j
        else:
            i += 1

    out = []
    for k in range(1, len(raw) - 1):
        a, b = raw[k]
        d = (b - a) * FRAME
        if not (0.005 <= d <= 0.080):
            continue
        pk = float(db[a:b].max())
        if not (-45 <= pk <= -25):
            continue
        gap_pre = (a - raw[k - 1][1]) * FRAME
        gap_nxt = (raw[k + 1][0] - b) * FRAME
        if gap_pre < gap_pre_min:
            continue
        if not (gap_nxt_min <= gap_nxt <= gap_nxt_max):
            continue
        seg = x[int(a * FRAME * SR):int(b * FRAME * SR)]
        zcr, low = voiced_quality(seg)
        if not (zcr < 0.15 and low > 0.45):
            continue
        out.append((a * FRAME, d, pk, zcr, low, gap_pre, gap_nxt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    x = load16(a.src)
    hits = find_head_clusters(x)
    dur = len(x) / SR
    print("%s  %.1fs  句首紧贴型碎片 %d 个 (%.2f/分钟)"
          % (os.path.basename(a.src), dur, len(hits),
             len(hits) / dur * 60), flush=True)
    for s, d, pk, zcr, low, gp, gn in hits:
        print("   t=%7.2fs %3.0fms 峰值%.0fdB zcr=%.3f 低频=%.2f "
              "前间隔%.0fms 后间隔%.0fms"
              % (s, d * 1000, pk, zcr, low, gp * 1000, gn * 1000),
              flush=True)

    if a.dry or not hits:
        if a.dst and not hits:
            sf.write(a.dst, x, SR)
            print("无需处理，原样写出", flush=True)
        return 0

    if not a.dst:
        print("未给输出路径，仅报告", flush=True)
        return 0

    y = x.copy()
    for s, d, *_ in hits:
        i0 = int(s * SR)
        i1 = int((s + d) * SR)
        y[i0:i1] = fade(np.zeros(i1 - i0, dtype=np.float32))
    sf.write(a.dst, y, SR)
    print("-> %s  已替换 %d 处（等长静音）" % (a.dst, len(hits)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
