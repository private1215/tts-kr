# -*- coding: utf-8 -*-
import os, sys, json
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
os.chdir("/mnt/tts/projects/qwen3-tts")
import server as S

OLD_M = ["1a6c20a60481", "ba24c0925311", "6cdd25bc75f2", "8197d9ac8a35", "00b6a45ad6aa"]
cases = json.load(open("/tmp/cases.json"))
by = {c["id"]: c for c in cases}
live = [c for c in cases if c["id"] not in ("c951ceb2c176", "653fff1fcca5")]

def med(v):
    v = sorted(v)
    n = len(v)
    return v[n//2] if n % 2 else (v[n//2-1]+v[n//2])/2.0

print(f"{'目标':<14}{'旧(5例)':>22}{'新(24例去重)':>22}")
def row(name, f, g):
    def pick(lst):
        v = [f(c) for c in lst if (c.get("gender") == g) and f(c) is not None]
        return med(v) if v else None
    o = pick([by[i] for i in OLD_M if i in by])
    n = pick(live)
    fo = f"{o:.3f}" if o is not None else "-"
    fn = f"{n:.3f}" if n is not None else "-"
    print(f"{name:<14}{fo:>22}{fn:>22}")

for g, gl in (("male", "男"), ("female", "女")):
    print(f"\n--- {gl} ---")
    row("lufs", lambda c: c.get("lufs") if c.get("lufs") else None, g)
    row("peak_dbfs", lambda c: c.get("peak_dbfs") if c.get("peak_dbfs") else None, g)
    row("cps", lambda c: c.get("cps"), g)
    row("proso_A", lambda c: (c.get("proso") or {}).get("A"), g)
    row("proso_E", lambda c: (c.get("proso") or {}).get("E"), g)
    row("C", lambda c: (c.get("prosody") or {}).get("C"), g)
    row("A", lambda c: (c.get("prosody") or {}).get("A"), g)
    row("pause/segs", lambda c: (lambda p: p.get("pauses")/max(p.get("segs"),1))(c.get("prosody") or {}) if (c.get("prosody") or {}).get("segs") else None, g)
    np_ = [len(c.get("pause_samples") or []) for c in live if c.get("gender") == g]
    print(f"{'停顿样本数':<14}{'-':>22}{sum(np_):>22}")

print("\n--- 线上实时目标值（server 函数）---")
for g in ("male", "female", "unknown"):
    print(f"  {g:<8} proso_targets={S.proso_targets(g)}")
    try:
        print(f"  {'':<8} loudness_targets={S.loudness_targets(g)}")
    except Exception as e:
        print("  loudness err", e)
