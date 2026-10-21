# -*- coding: utf-8 -*-
"""案例库说话人聚类：36 个案例到底是不是同一批音色？微调必须按说话人分组。"""
import os, sys, json, tempfile, subprocess
import numpy as np, torch, soundfile as sf, onnxruntime
import torchaudio.compliance.kaldi as kaldi

SR = 16000
WIN = int(5.0 * SR)
NWIN = 8
CAMPPLUS = "/mnt/tts/models/Fun-CosyVoice3-0.5B/campplus.onnx"
CASES = "/mnt/tts/data/cases"
SAME = 0.75

opt = onnxruntime.SessionOptions()
opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = onnxruntime.InferenceSession(CAMPPLUS, sess_options=opt,
                                    providers=["CPUExecutionProvider"])

def read16(path, s0=30.0, dur=60.0):
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir="/tmp"); os.close(fd)
    try:
        subprocess.run(["ffmpeg","-nostdin","-v","error","-y","-ss",str(s0),
                        "-i",path,"-t",str(dur),"-ac","1","-ar",str(SR),tmp],
                       check=True)
        return torch.from_numpy(sf.read(tmp, dtype="float32")[0]).unsqueeze(0)
    finally:
        try: os.remove(tmp)
        except OSError: pass

def embed(speech):
    f = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=SR)
    f = f - f.mean(dim=0, keepdim=True)
    n = f.shape[0]
    if n < 50: return None
    step = max(1, (n - 200) // max(NWIN-1, 1))
    vs = []
    for s in range(0, max(n-200, 1), step)[:NWIN]:
        seg = f[s:s+200]
        if seg.shape[0] < 30: continue
        e = sess.run(None, {sess.get_inputs()[0].name:
                            seg.unsqueeze(0).cpu().numpy()})[0].flatten()
        e = np.asarray(e, dtype=np.float64); nv = np.linalg.norm(e)
        if nv > 1e-12: vs.append(e/nv)
    if not vs: return None
    v = np.mean(vs, axis=0); nv = np.linalg.norm(v)
    return v/nv if nv > 1e-12 else None

cj = json.load(open('/mnt/tts/data/cases.json', encoding='utf-8'))
rows = []
for cid, v in cj.items():
    p = os.path.join(CASES, cid, 'merged.wav')
    if not os.path.exists(p): 
        print('  缺 merged.wav', cid); continue
    try:
        e = embed(read16(p))
    except Exception as ex:
        print('  ERR', cid, repr(ex)[:60]); continue
    if e is None: print('  emb失败', cid); continue
    rows.append((cid, v.get('gender'), e)); print('.', end='', flush=True)
print()

import collections
by = collections.defaultdict(list)
for cid, g, e in rows: by[g].append((cid, e))

for g, anchor in (('female','00b6a45ad6aa'), ('male','1a6c20a60481')):
    L = by.get(g) or []
    print('\n===== %s：%d 个（锚点=%s）=====' % (g, len(L), anchor))
    ai = next((i for i,(c,_) in enumerate(L) if c == anchor), None)
    if ai is None: continue
    sims = sorted(((c, float(np.dot(e, L[ai][1]))) for c, e in L),
                  key=lambda x: -x[1])
    n_same = 0
    for cid, s in sims:
        tag = 'OK ' if s >= SAME else ('~  ' if s >= 0.60 else 'X  ')
        if s >= SAME: n_same += 1
        print('  %s%-14s %.3f' % (tag, cid, s))
    dur = sum(x.get('dur') or 0 for x in cj[cid].get('segments', [])
              for cid, s in sims if s >= SAME)
    print('  --> 与锚点同人(>=%.2f)：%d 个，合计 %.1f 分钟' % (SAME, n_same, dur/60))
