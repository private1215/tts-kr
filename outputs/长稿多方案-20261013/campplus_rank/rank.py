# -*- coding: utf-8 -*-
"""长稿候选 vs 案例原声 campplus 余弦总表（对应员工第三轮盲听）。"""
import os, sys, json
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")
import numpy as np
import spkchk, onnxruntime

CAMPPLUS = "/mnt/tts/models/Fun-CosyVoice3-0.5B/campplus.onnx"
SESS = onnxruntime.InferenceSession(CAMPPLUS, providers=["CPUExecutionProvider"])

LS = "/mnt/tts/diag/longsubmit-20261013/deliver"
B2 = "/mnt/tts/diag/b2-20261013/deliver"

FEMALE = {
    "ref_EL案例原声":      LS + "/case1-ref.wav",
    "cand1_千问线上链":     LS + "/case1-cand1.wav",
    "cand2_cosy线上链":     LS + "/case1-cand2.wav",
    "cand3_cosy线上链":     LS + "/case1-cand3.wav",
    "cand4_cosy裸推理":     LS + "/case1-cand4.wav",
    "cand5_千问裸推理":     LS + "/case1-cand5.wav",
    "cand6_串联千问→cosy":  "/mnt/tts/diag/series-20261013/S2_f.wav",
    "cand7_GS":            "/mnt/tts/diag/gs_backup-20261013/G2_f.wav",
    "cand8_VoxCPM2":       "/mnt/tts/diag/voxcpm-20261013/V2_f.wav",
    "cand9_OmniVoice":     "/mnt/tts/diag/omnivoice-20261013/O2_f.wav",
}
MALE = {
    "ref_EL案例原声":      LS + "/case2-ref.wav",
    "cand1_千问线上链":     LS + "/case2-cand1.wav",
    "cand2_千问裸推理":     LS + "/case2-cand2.wav",
    "cand3_cosy线上链":    LS + "/case2-cand3.wav",
    "cand4_cosy线上链":    LS + "/case2-cand4.wav",
    "cand5_cosy裸推理":    LS + "/case2-cand5.wav",
    "cand6_串联千问→cosy": "/mnt/tts/diag/series-20261013/S1_m.wav",
    "cand7_GS":            "/mnt/tts/diag/gs_backup-20261013/G1_m.wav",
    "cand8_VoxCPM2":       "/mnt/tts/diag/voxcpm-20261013/V1_m.wav",
    "cand9_OmniVoice":     "/mnt/tts/diag/omnivoice-20261013/O1_m.wav",
}

def emb_of(path):
    sp = spkchk.read16(path)
    e = spkchk.embed(SESS, sp)
    return e / (np.linalg.norm(e) + 1e-9)

out = {}
for gender, table in [("female", FEMALE), ("male", MALE)]:
    ref = emb_of(table["ref_EL案例原声"])
    rows = []
    for name, p in sorted(table.items()):
        if not os.path.exists(p):
            rows.append({"name": name, "cos": None, "err": "missing"})
            continue
        e = emb_of(p)
        cos = float(np.dot(ref, e))
        rows.append({"name": name, "cos": round(cos, 4)})
        print(gender, name, round(cos, 4), flush=True)
    rows = sorted([r for r in rows if r["cos"] is not None], key=lambda r: -r["cos"]) + \
           [r for r in rows if r["cos"] is None]
    out[gender] = rows

with open("/mnt/tts/diag/campplus_rank-20261013/rank.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("DONE", flush=True)
