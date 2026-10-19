# -*- coding: utf-8 -*-
"""ltasmatch53b.py —— ltasmatch53 的**保采样率**版。

## 与 53 版的唯一差别
53 版写死 `SR = 16000`，`apply_fq()` 和最终写出都带 `-ar 16000`，
**成品一律降到 16 kHz**。2026-10-19 实测发现这是条链式坑：

    16k 成品 → 压 192k mp3 → 音色 0.9331 掉到 0.6903（不同人）
    同一对 16k 的 wav/mp3 互比                只有 0.7900
    对照 44.1k 的 wav/mp3 互比                0.9999（mp3 透明）

→ **libmp3lame 在 16 kHz 输入上明显伤音色，在 44.1 kHz 上完全透明。**
  而项目所有客观指标（measure53 / click / 杂音）口径都是「先降到 16k 再测」，
  这等于把「成品被降到 16k」这件事**完全藏起来了**，只有 campplus 抓得到。

## 做法
测量（LTAS / click）仍然在 16k 做 —— 那是口径，不能改；
但 **firequalizer 的施加与最终写出都在原生采样率上**。

用法：python3 ltasmatch53b.py <ours.wav> <el.wav> <out.wav> [--iters 3]

## 为什么

男声中频 EQ(+1~2dB) 已把 1k-3k 对齐，但**其他频段仍有偏差**：

    <300Hz   我们 25.63% vs EL 27.15%
    300-1k   我们 50.71% vs EL 48.26%
    1k-3k    我们 19.38% vs EL 20.65%
    3k-6k    我们  4.08% vs EL  3.77%

EL 是标杆 → **让整条 LTAS 逼近 EL 是确定性优化**（对齐标杆，不依赖听感拍板）。

## 做法

1. 算我们与 EL 的发声帧平均功率谱，归一到同总能量
2. 各频段目标增益 G = 10*log10(EL能量 / 我们能量)
3. 用 firequalizer 多段 gain_entry 应用
4. 迭代 N 轮，直到各频段误差都很小
5. 输出成品 + 验证（click / 杂音 / 质心 不得恶化）

⚠️ 本机 ffmpeg `equalizer` 失效，必须用 firequalizer。

用法：python3 ltasmatch53.py <ours.wav> <el.wav> <out.wav> [--iters 3]
"""
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
VTH = -40.0
# 频段中心（用于 firequalizer entry）+ 边界
EDGES = [(0, 300), (300, 1000), (1000, 3000), (3000, 6000), (6000, 8000)]
CENTERS = [150, 650, 2000, 4500, 7000]


def load16(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    return x.astype(np.float64)


def ltas(p):
    """发声帧平均功率谱（归一到总能量=1）在各频段的占比。"""
    x = load16(p)
    n = int(FRAME * SR)
    nfr = len(x) // n
    fr = x[:nfr * n].reshape(nfr, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    v = db > VTH
    sp = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    m = sp[v].mean(axis=0) if v.sum() else sp.mean(axis=0)
    tot = m.sum() + 1e-12
    return np.array([float(m[(freqs >= a) & (freqs < b)].sum() / tot)
                     for a, b in EDGES])


def native_sr(p):
    import soundfile as _sf
    return int(_sf.info(p).samplerate)


def apply_fq(p, gains, sr=None):
    """gains: 每个频段中心处的 dB 增益。

    🔴 53 版这里带 `-ar 16000`，会把成品降采样。
       本版保留**原生采样率**（firequalizer 在任意采样率下都能算 FIR）。
    """
    ent = ";".join("entry(%d,%g)" % (c, g)
                   for c, g in zip(CENTERS, gains))
    ent = "entry(0,%g);%s;entry(8000,%g)" % (gains[0] * 0.5, ent, gains[-1])
    t = tempfile.mktemp(suffix=".wav")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", p, "-af",
           "firequalizer=gain_entry='%s'" % ent, "-ac", "1"]
    if sr:
        cmd += ["-ar", str(sr)]
    cmd.append(t)
    subprocess.run(cmd, check=True)
    return t


def click_of(p):
    x = load16(p)
    pk = np.max(np.abs(x))
    xn = x * (0.70 / pk) if pk > 0 else x
    return float((np.abs(np.diff(xn)) >= 0.30).mean()) * 100


def main():
    if len(sys.argv) < 4:
        print("用法：ltasmatch53.py <ours.wav> <el.wav> <out.wav> [--iters N]")
        return 1
    ours, el, out = sys.argv[1:4]
    global _NSR
    _NSR = native_sr(ours)
    print("原生采样率 %d Hz" % _NSR, flush=True)
    iters = 3
    if "--iters" in sys.argv:
        iters = int(sys.argv[sys.argv.index("--iters") + 1])
    tgt = ltas(el)
    print("EL 目标频段占比: " + "  ".join("%.2f%%" % (v * 100) for v in tgt))
    cur = ours
    gains = [0.0] * len(EDGES)
    for it in range(iters):
        cur_l = ltas(cur)
        print("第%d轮 当前: " % (it + 1)
              + "  ".join("%.2f%%" % (v * 100) for v in cur_l))
        # 增益 = 10*log10(目标/当前)，限幅 ±6dB 防过冲
        dg = [max(-6.0, min(6.0, 10 * np.log10((t + 1e-9) / (c + 1e-9))))
              for t, c in zip(tgt, cur_l)]
        gains = [g + d for g, d in zip(gains, dg)]
        print("       增益: " + "  ".join("%+.2fdB" % g for g in dg))
        nxt = apply_fq(ours, gains)
        cur = nxt
    fin = ltas(cur)
    print("\n最终:   " + "  ".join("%.2f%%" % (v * 100) for v in fin))
    print("EL:     " + "  ".join("%.2f%%" % (v * 100) for v in tgt))
    err = np.abs(fin - tgt).sum()
    print("总误差 %.4f（越接近 0 越好）" % err)
    print("累计增益: " + "  ".join("%+.2fdB" % g for g in gains))
    print("click%%: 原 %.4f -> 匹配后 %.4f"
          % (click_of(ours), click_of(cur)))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", cur,
                    "-ar", str(_NSR), "-ac", "1", out], check=True)
    print("-> %s（保采样率 %d Hz）" % (out, _NSR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
