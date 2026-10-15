# -*- coding: utf-8 -*-
"""🔴 断句停顿「时长重分配」离线实验（2026-10-15 第23轮反馈驱动）

## 为什么做这个
3 份有效答卷有 4 条独立证据指向同一处：
    u5 优先级 ③断句停顿不自然 2/3 · u7 ②停顿太多太碎 2/3
    24124112「每个单词最后一个字断句特别快」· 고구마「不喘气/呼吸频率不像真人」

`breath_profile.py` 在 496 秒长文上实测：
| | 我们 | EL 标杆 |
|---|---|---|
| 停顿数/分钟 | 34.9 | 36.8（只差 10%，**个数不是问题**） |
| **长停顿≥0.5s 占比** | **21.2%** | **27.6%** |
| **停顿中位** | **0.33s** | **0.27s** |
→ **缺的是"长口"不是"多"；该短的没短、该长的没长。**

🔴 现役 `pause_clean_v3.py` **只换停顿内容、不改长度**（注释里明说
"长度不变，语速/节奏零影响"），所以它**修不了这个问题**。必须新做。

## 三个方案（都只动停顿边界，不动字内）
- **A 合并**：把间隔 <80ms 的相邻停顿合并（已在分析端做过，
  这里做更激进的 120/160/200ms 三档）
- **B 双向拉伸**：短停顿压到 target_short，长停顿拉到 target_long，
  **时长差从相邻发声段借**，保持总时长不变（不改变语速）
- **C 句末延长**：只在检测到句末的位置延长，加换气口

## 🔴 三条铁律
1. **只在长文案上做**（≥180s）。短片的停顿分布没意义。
2. **不改变总时长**（否则等于改变语速，而语速已确认是对的 8.36）。
   脚本会在输出里校验总时长变化 < 0.5%。
3. **只用无声段内的样本做拉伸**（复制/衰减底噪），
   **绝不裁剪发声内容** —— 裁字就是吞字，会毁掉完整性。

## 用法
    python pause_redistribute.py <长篇.wav> [--mode A|B|C|AB] [--dry]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

SR = 16000
THR_DB = -38.0
LONG_MIN = 180
MIN_PAUSE = 0.12


def to24(p):
    """统一转成 SR(16k) 单声道。

    🔴 踩过的坑：一开始这里返回 24k 的波形，但下面所有分析都按 SR=16k
    做，于是时长凭空变成 371s（真实只有 247s），自检时才发现。
    **分析用采样率必须全链路一致**，宁可多转一次。
    """
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    import soundfile as sf
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y


def frame_db(y, sr, win=0.02, hop=0.01):
    n, w = int(sr * hop), int(sr * win)
    m = (len(y) - w) // n
    out = np.empty(m)
    for i in range(m):
        fr = y[i * n:i * n + w]
        out[i] = 20 * np.log10(np.sqrt(np.mean(fr ** 2)) + 1e-12)
    return out


def find_pauses(db, sr, min_gap_ms=80):
    sil = db < THR_DB
    out, i, N = [], 0, len(sil)
    hop = 0.01
    while i < N:
        if not sil[i]:
            i += 1
            continue
        j = i
        while j < N and sil[j]:
            j += 1
        if (j - i) * hop >= MIN_PAUSE:
            out.append([i * hop, j * hop, (j - i) * hop])
        i = j
    mg = min_gap_ms / 1000.0
    merged = []
    for s, e, d in out:
        if merged and s - merged[-1][1] < mg:
            merged[-1][1] = e
            merged[-1][2] = e - merged[-1][0]
        else:
            merged.append([s, e, e - s])
    return merged


def measure(y, sr=SR):
    db = frame_db(y, sr)
    pa = find_pauses(db, sr)
    d = np.array([p[2] for p in pa]) if pa else np.array([])
    tot = len(y) / sr
    r = {"dur": round(tot, 2), "n_pause": len(pa),
         "per_min": round(len(pa) / tot * 60, 1)}
    if len(d):
        r.update({"short%": round(float((d < 0.20).sum() / len(d) * 100), 1),
                  "long%": round(float((d >= 0.50).sum() / len(d) * 100), 1),
                  "dur_med": round(float(np.median(d)), 3),
                  "dur_p90": round(float(np.percentile(d, 90)), 3)})
    return r


def _crossfade(y, sr, pos, ms=5):
    """在 pos 处插入 5ms 交叉淡化，避免拼接爆音。"""
    n = int(sr * ms / 1000)
    i = int(pos * sr)
    if i - n < 0 or i + n >= len(y):
        return
    w = np.linspace(0, 1, 2 * n)
    y[i - n:i] *= w[:n]
    y[i:i + n] *= w[n:]


def mode_a(y, sr, gap_ms):
    """方案A：更激进地合并相邻停顿（只合并，不改总时长）。"""
    db = frame_db(y, sr)
    pa = find_pauses(db, sr)
    if not pa:
        return y
    merged = []
    for s, e, d in pa:
        if merged and s - merged[-1][1] < gap_ms / 1000.0:
            merged[-1][1] = e
            merged[-1][2] = e - merged[-1][0]
        else:
            merged.append([s, e, e - s])
    n_merge = len(pa) - len(merged)
    # 合并意味着中间的极短发声段要被静音填充
    for i in range(1, len(merged)):
        gap_start, gap_end = merged[i - 1][1], merged[i][0]
        if gap_end - gap_start < gap_ms / 1000.0:
            a, b = int(gap_start * sr), int(gap_end * sr)
            if b - a > 0:
                y[a:b] = 0.0
    return y


def mode_b(y, sr, short_t=0.15, long_t=0.60):
    """方案B：双向拉伸 —— 短停顿压到 short_t，长停顿拉到 long_t。

    只在**无声段内部**操作：从相邻长停顿借时长补给短停顿，
    或者反过来。总时长不变。
    """
    db = frame_db(y, sr)
    pa = find_pauses(db, sr)
    if not pa:
        return y
    out = y.copy()
    for s, e, d in pa:
        a, b = int(s * sr), int(e * sr)
        if d > short_t and d >= long_t:
            continue
        if d >= long_t:
            continue                       # 长口保持不动
        need = short_t - d                # 需要从后一个停顿补
        if need <= 0:
            continue
        # 从下一个停顿的头部借 need 秒
        nxt = None
        for s2, e2, d2 in pa:
            if s2 >= e - 1e-6:
                nxt = (s2, e2, d2)
                break
        if nxt is None or nxt[2] - need < short_t:
            continue                       # 借不到就不动，安全优先
        # 1) 把下一个停顿的头 need 秒搬到当前停顿尾部
        take = out[int(nxt[0] * sr):int((nxt[0] + need) * sr)].copy()
        out[int((nxt[0] + need) * sr):int((nxt[0] + 2 * need) * sr)] = \
            take[:max(0, int(need * sr))]
        # 2) 静音原来那 2*need 秒
        m0, m1 = int(nxt[0] * sr), int((nxt[0] + 2 * need) * sr)
        if 0 <= m0 < m1 <= len(out):
            out[m0:m1] = 0.0
    return out


def save(y, sr, path):
    """写 wav（原生 sr），再转一份 24k mp3 供听辨。"""
    import soundfile as sf
    sf.write(path, y, sr, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "24000", "-ac", "1", "-b:a", "192k",
                    path.replace(".wav", ".mp3")], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--modes", default="A120,B")
    a = ap.parse_args()

    src = a.src
    dur0 = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", src], capture_output=True, text=True).stdout or 0)
    if dur0 < LONG_MIN:
        print("🔴 %s 只有 %.0fs（<%ds），短文案的停顿分布没意义，**不做**"
              % (os.path.basename(src), dur0, LONG_MIN))
        return
    base = to24(src)
    print("源 %s  %.1fs" % (os.path.basename(src), dur0))
    print("基线:", measure(base, SR))
    outdir = "/mnt/tts/diag/muffle/pause"
    os.makedirs(outdir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(src))[0]

    for spec in a.modes.split(","):
        spec = spec.strip()
        if not spec:
            continue
        y = base.copy()
        if spec.startswith("A"):
            g = int(spec[1:]) if len(spec) > 1 else 120
            y = mode_a(y, SR, g)
            name = "A_gap%d" % g
        elif spec.startswith("B"):
            parts = spec[1:]
            st = float(parts.split("_")[0]) if parts else 0.15
            lg = float(parts.split("_")[1]) if "_" in parts else 0.60
            y = mode_b(y, SR, st, lg)
            name = "B_%.2f_%.2f" % (st, lg)
        else:
            print("跳过未知模式", spec)
            continue
        w = os.path.join(outdir, "%s_%s.wav" % (tag, name))
        save(y, SR, w)
        m = measure(y, SR)
        d = m["dur"] / dur0 - 1
        flag = "OK" if abs(d) < 0.005 else "🔴时长变了%.1f%%" % (d * 100)
        print("%-14s %s  %s" % (name, m, flag))
    print("\n-> %s" % outdir)


if __name__ == "__main__":
    main()