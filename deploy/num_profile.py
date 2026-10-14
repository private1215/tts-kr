import sys, numpy as np, soundfile as sf

W = 0.010  # 10ms frame

def prof(path, t0, t1, thr=-38.0):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    a, b = int(t0 * sr), int(t1 * sr)
    seg = x[a:b]
    n = int(W * sr)
    nfr = len(seg) // n
    r = seg[: nfr * n].reshape(nfr, n)
    rms = np.sqrt((r ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    print(f"\n### {path.split('/')[-1]}  [{t0:.2f}-{t1:.2f}]  sr={sr}  frames={nfr}")
    print("    t(s)   dB   " + " " * 5 + "| -60 .......... -20 .......... 0")
    for i, d in enumerate(db):
        t = t0 + i * W
        lo, hi = -60.0, -5.0
        p = int(round((np.clip(d, lo, hi) - lo) / (hi - lo) * 50))
        bar = " " * p + "#"
        mark = "  <<" if d < thr else ""
        print(f"  {t:7.2f} {d:6.1f}  |{bar}{mark}")
    # 间隙结构
    voiced = db >= thr
    gaps = []
    i = 0
    while i < len(voiced):
        if not voiced[i]:
            j = i
            while j < len(voiced) and not voiced[j]:
                j += 1
            gaps.append((t0 + i * W, (j - i) * W))
            i = j
        else:
            i += 1
    print(f"  -- 低于 {thr}dB 的间隙 {len(gaps)} 处: " +
          ", ".join(f"{a:.2f}s/{d*1000:.0f}ms" for a, d in gaps if d >= 0.03))

if __name__ == "__main__":
    prof(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]),
         float(sys.argv[4]) if len(sys.argv) > 4 else -38.0)
