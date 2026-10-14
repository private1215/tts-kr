# -*- coding: utf-8 -*-
"""male_fix_v2.py 补丁：加 boost 全局系数（第5参数）、响度取最后匹配。"""
P = "/mnt/tts/diag/campplus_rank-20261013/male_fix_v2.py"
s = open(P, encoding="utf-8").read()

old1 = '''    raw, el, prefix, tcsv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    targets = sorted([float(x) for x in tcsv.split(",")], reverse=True)'''
new1 = '''    raw, el, prefix, tcsv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    boost = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    targets = sorted([float(x) for x in tcsv.split(",")], reverse=True)'''
assert old1 in s, "anchor1"
s = s.replace(old1, new1, 1)

old2 = '''    fs = [ (fs[i-1]+2*fs[i]+fs[i+1])/4 if 0<i<len(fs)-1 else fs[i] for i in range(len(fs)) ]
    fs = [min(max(f, 1.0), 1.30) for f in fs]'''
new2 = '''    fs = [ (fs[i-1]+2*fs[i]+fs[i+1])/4 if 0<i<len(fs)-1 else fs[i] for i in range(len(fs)) ]
    fs = [min(max(f*boost, 1.0), 1.35) for f in fs]'''
assert old2 in s, "anchor2"
s = s.replace(old2, new2, 1)

old3 = '''        m = re.search(r"I:\\s+(-?[\\d.]+)\\s+LUFS", r.stderr)
        return float(m.group(1)) if m else -24.0'''
new3 = '''        ms = re.findall(r"I:\\s+(-?[\\d.]+)\\s+LUFS", r.stderr)
        return float(ms[-1]) if ms else -24.0'''
assert old3 in s, "anchor3"
s = s.replace(old3, new3, 1)

open(P, "w", encoding="utf-8").write(s)
print("male_fix_v2 patched: boost arg + lufs last-match")
