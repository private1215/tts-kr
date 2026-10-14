import sys, numpy as np, soundfile as sf

W = 0.010

def rms_db(x, sr):
    n = int(W * sr)
    nfr = len(x) // n
    r = x[: nfr * n].reshape(nfr, n)
    db = 20 * np.log10(np.sqrt((r ** 2).mean(axis=1)) + 1e-12)
    return db

def nuclei(db, t0, thr=-35.0, min_sep_ms=60, min_dur_ms=60):
    """返回音节核: (起, 止, 峰值dB)"""
    v = db >= thr
    runs = []
    i = 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    if not runs:
        return []
    # 合并间隔 < min_sep 的相邻段
    merged = [runs[0]]
    for a, b in runs[1:]:
        if (a - merged[-1][1]) * W * 1000 < min_sep_ms:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    # 丢弃过短
    out = []
    for a, b in merged:
        if (b - a) * W * 1000 >= min_dur_ms:
            out.append((t0 + a * W, (b - a) * W, db[a:b].max()))
    return out

def main():
    path, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    thr = float(sys.argv[4]) if len(sys.argv) > 4 else -35.0
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    a, b = int(t0 * sr), int(t1 * sr)
    seg = x[a:b]
    db = rms_db(seg, sr)
    nuc = nuclei(db, t0, thr)
    print(f"\n### {path.split('/')[-1]} [{t0:.2f}-{t1:.2f}] thr={thr} 音节核 {len(nuc)} 个")
    prev_end = None
    for k, (s, d, pk) in enumerate(nuc):
        gap = f"  gap={ (s-prev_end)*1000:6.0f}ms" if prev_end is not None else ""
        print(f"  #{k+1:2d}  {s:7.2f}s  dur={d*1000:5.0f}ms  peak={pk:5.1f}dB{gap}")
        prev_end = s + d
    if len(nuc) >= 2:
        onsets = [s for s, d, p in nuc]
        iv = np.diff(onsets)
        span = onsets[-1] - onsets[0]
        print(f"  -- 首末 onset 跨度 {span:.3f}s, 平均 onset 间隔 {iv.mean()*1000:.0f}ms "
              f"(min {iv.min()*1000:.0f} / max {iv.max()*1000:.0f})")
        print(f"  -- 音节核总时长 {sum(d for s,d,p in nuc):.3f}s, "
              f"核间静音总时长 {span - sum(d for s,d,p in nuc[:-1]):.3f}s")

main()
