# -*- coding: utf-8 -*-
"""head_repro.py —— 文件头孤立段复现率实验。

背景（上轮发现）：我们 4 条女声产出里 2 条在文件头 0.07~0.5s 有
「孤立浊音段 + 近静音断口」（EL 参考 0/4 复现）。样本太小，需扩大。

做法：同一短文案、同一音色、同模式连跑 N 条，逐条测文件头：
  起音时刻 / 起音后是否出现 >150ms 的近静音断口 / 孤立段谱峰
跑完打印复现率。复现率 ≥50% → 链路确定性缺陷；<20% → 偶发。
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
os.environ.setdefault("TTS_BASE", "http://127.0.0.1:8000")
from tools.eval5 import submit_job, wait_job, _get, login  # noqa: E402

# 短文案：取自 202610190902 开头（保留原句，避免引入新变量）
TEXT = ("로보티즈 주주 여러분, 반갑습니다. 정민철 프로입니다. "
        "오늘 우리 주주님들께, 진짜 피가 되고 살이 되는 중요한 말씀 드리려고, "
        "급하게 카메라 켰습니다. 와, 지금 로보티즈, 자리 기가 막히거든요. "
        "왜 이 자리가 중요하냐고요? 시월 이일, 오전 열한 시 삼십사분 기준, "
        "로보티즈 삼십만구천오백원. 전일 대비 한 사쩜사퍼센트 정도 띄워주고 있습니다.")
N = int(os.environ.get("HEAD_N", "10"))
TITLE = "headrepro-1019"


def head_check(wav):
    x, sr = sf.read(wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    hop = int(0.01 * sr)
    n = len(x) // hop
    v = np.array([20 * np.log10(np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2)) + 1e-12)
                  for i in range(n)])
    loud = [i for i, q in enumerate(v) if q > -35]
    if not loud:
        return None
    onset = loud[0] * 0.01
    # 起音后 150ms 内找 >150ms 的近静音断口（< -47dB）
    gaps = []
    i = loud[0]
    end = min(n, int((onset + 1.0) / 0.01))
    while i < end:
        if v[i] < -47:
            j = i
            while j < end and v[j] < -47:
                j += 1
            if (j - i) * 0.01 > 0.15:
                gaps.append((i * 0.01, j * 0.01))
            i = j
        else:
            i += 1
    return {"onset": onset, "gaps": gaps,
            "env": [round(float(q), 1) for q in v[:80]]}


def main():
    login()
    print("连跑 %d 条（文案 %d 字）" % (N, len(TEXT)))
    rows = []
    for k in range(N):
        title = "%s-%d" % (TITLE, k + 1)
        r = submit_job(TEXT, "EL_female", title)
        jid = r.get("id") or r.get("job_id") or (r.get("job") or {}).get("id")
        if not jid:
            print("  #%d 提交失败: %s" % (k + 1, json.dumps(r, ensure_ascii=False)[:200]))
            continue
        wait_job(jid, timeout=1800)
        st = json.loads(_get("/api/jobs/" + jid).decode("utf-8"))
        mp3 = None
        for f in st.get("files") or []:
            mp3 = f.get("path")
        src = os.path.join("/mnt/tts/output", mp3)
        dst = "/mnt/tts/diag/v0902/%s.mp3" % title
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        wav = dst.replace(".mp3", ".16k.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", dst,
                        "-ar", "16000", "-ac", "1", wav], check=True)
        h = head_check(wav)
        rows.append({"title": title, "jid": jid, "head": h})
        g = "有" if h and h["gaps"] else "无"
        detail = ",".join("%.2f-%.2f" % (a, b) for a, b in (h["gaps"] if h else []))
        print("  #%-2d %s 起音 %.2fs 断口: %s %s"
              % (k + 1, title, h["onset"] if h else -1, g, detail))
    json.dump(rows, open("/mnt/tts/diag/v0902/headrepro.json", "w"),
              ensure_ascii=False, indent=1)
    bad = sum(1 for r in rows if r["head"] and r["head"]["gaps"])
    print("\n复现率：%d/%d = %.0f%%" % (bad, len(rows), 100.0 * bad / max(1, len(rows))))
    print("\n各条前 60ms 包络（10ms/点，dBFS）：")
    for r in rows:
        if not r["head"]:
            continue
        print("  %-18s %s" % (r["title"], " ".join("%.0f" % q for q in r["head"]["env"][:60])))


if __name__ == "__main__":
    main()
