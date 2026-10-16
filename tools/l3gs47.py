# -*- coding: utf-8 -*-
"""第47轮 L3 验收：男声 GS 修复后，1500 字长篇（约 222s = 3.7 分钟）。
直连 GS 绕开 qwen 的重掷逻辑（重掷会因数字假错白跑 3 遍）。
参考音频 = yohan_gs10（7.55s / 51字 / 6.75字每秒，句边界切）。
"""
import io, os, sys, json, time
import numpy as np, soundfile as sf, requests
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
import app.asr_check as asr_check

GS = "http://127.0.0.1:8200"
OUT = "/mnt/tts/diag/l3gs47"
os.makedirs(OUT, exist_ok=True)

txt = open("/mnt/tts/data/cases/6cdd25bc75f2/full.txt", encoding="utf-8").read()
TEXT = txt[:1500]
V = json.load(open("/mnt/tts/data/voices/voices.json"))
REFW = V["yohan_gs10"]["path"]
RT = V["yohan_gs10"]["ref_text"]
print(f"文本 {len(TEXT)} 字；参考 {sf.info(REFW).duration:.2f}s / {len(RT)}字 "
      f"/ {len(RT)/sf.info(REFW).duration:.2f}字每秒")
print(f"ref_text = {RT}")

wavp = os.path.join(OUT, "gs_m_l3.wav")
if not os.path.exists(wavp):
    t0 = time.time()
    r = requests.post(GS + "/synth", data={
        "text": TEXT, "spk": "yohan", "ref_wav": REFW,
        "ref_text": RT, "seed": 42, "seg": 50}, timeout=3600)
    print("GS ->", r.status_code)
    if r.status_code != 200:
        print(r.text[:300]); sys.exit(1)
    x, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    sf.write(wavp, x, sr)
    print(f"[ok] {len(x)/sr:.2f}s  用时 {time.time()-t0:.0f}s")

x, sr = sf.read(wavp, dtype="float32")
dur = len(x) / sr
cps = len(TEXT) / dur
print(f"\n=== L3 结果 ===")
print(f"时长 {dur:.1f}s   cps {cps:.2f}  (目标 6.7~6.9, 修复前 2.03)")

res = asr_check.check_chunk(wavp, TEXT, "Korean")
cer = res.get("cer")
print(f"CER {cer:.4f}   (修复前 0.66)")

# 杂音：静音段 4-8kHz
def hiss(y, sr):
    hop = int(0.02 * sr)
    n = len(y) // hop
    rms = np.array([np.sqrt(np.mean(y[i*hop:(i+1)*hop]**2)+1e-12) for i in range(n)])
    N = 2048
    win = np.hanning(N)
    k = min(n, 4000)
    eng = []
    idx = np.argsort(rms)[:max(1, int(n*0.2))]   # 最安静 20% 帧
    for i in idx[:k]:
        seg = y[i*hop:i*hop+N]
        if len(seg) < N:
            continue
        S = np.abs(np.fft.rfft(seg*win))**2
        f = np.fft.rfftfreq(N, 1/sr)
        m = (f >= 4000) & (f < 8000)
        eng.append(10*np.log10(S[m].sum()+1e-12))
    return float(np.mean(eng)) if eng else float("nan")
print(f"静音段 4-8kHz = {hiss(x, sr):.3f}  (EL男 10.97 / 修复前 gs47m3 9.16)")

hyp = res.get("hyp", "")
import difflib
a, b = asr_check.normalize(TEXT), asr_check.normalize(hyp)
big = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
       if op[0] == "insert" and (op[4]-op[3]) >= 20]
print(f"ASR 大段 insert(>=20字) 次数 = {len(big)}  (真乱说话的判据；修复前 6 次)")
for op in big[:5]:
    print(f"    +{b[op[3]:op[4]]}")
json.dump({"dur": dur, "cps": cps, "cer": cer},
          open(os.path.join(OUT, "result.json"), "w"), indent=1)
print("done")
