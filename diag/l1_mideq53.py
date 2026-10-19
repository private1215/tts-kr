# -*- coding: utf-8 -*-
"""l1_mideq53.py —— L1 快速验证：中频 EQ 补丁在**线上**真的生效了吗。

## 为什么需要这一步（2026-10-19 血的教训）
第一版 `patch_mideq53b.py` 写成了
    return _clickfix53b_wrap(*_mideq53b_wrap(y2, sr2, _g, _engine_name))
`_mideq53b_wrap` 返回 `(audio, sr)` 二元组，`*` 解包后只够 audio+sr 两个
位置参数，**缺 gender / engine_name** →
    TypeError: _clickfix53b_wrap() missing 2 required positional arguments
→ 整块回退 raw，**EQ 和 clickfix 两个都没跑**。
而 `audit53.py` 的字符串检查**全绿**（它只确认代码在，不确认被执行）——
所以必须跑 L1 实测，不能只信体检。

## 做法
用现役 EL_male 合成一段 ~30s，压 mp3，测中频占比与 EL 对照。
预期：中频从 0.86×EL 抬到 ≈1.0×EL（+2dB 窄带 EQ）。

用法: venv/bin/python -u /mnt/tts/diag/l1_mideq53.py
"""
import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")

from eval5 import login, submit_job, wait_job  # noqa: E402

T = "/mnt/tts/projects/qwen3-tts/tools"
EL = "/mnt/tts/data/cases/d172b8c4efb3/merged.wav"
SR = 16000
FRAME = 0.005


def metrics(p):
    """与 measure53 同口径：发声帧中频 1k-3k 占比（16k，5ms 帧）。"""
    t = "/tmp/_l1_16k.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float64")
    n = int(FRAME * SR)
    nfr = len(x) // n
    fr = x[:nfr * n].reshape(nfr, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    v = 20 * np.log10(rms + 1e-12) > -40
    sp = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    mid = [float(s[(freqs >= 1000) & (freqs <= 3000)].sum() /
                 (s.sum() + 1e-12)) for s in sp[v]]
    return float(np.median(mid)) * 100 if mid else 0.0


def main():
    text = ("안녕하세요. 세미파이브 주주 여러분, 반갑습니다. "
            "오늘은 수급의 구조를 한번 짚어보려고 합니다. "
            "외인창구와 기관창구의 흐름을 같이 보시면 "
            "개인들이 어디서 물량을 받는지 보이실 겁니다.")
    login()
    j = submit_job(text, "EL_male", "l1_mideq53")
    jid = j["id"] if isinstance(j, dict) else j
    print("job =", jid, flush=True)
    for _ in range(40):
        r = wait_job(jid, timeout=30)
        st = r.get("status") if isinstance(r, dict) else None
        if st in ("done", "finished", "completed", "success"):
            break
        if st in ("error", "failed"):
            print("失败：", r)
            return 1
        time.sleep(5)
    d = "/mnt/tts/output/l1_mideq53"
    hits = sorted(glob.glob(d + "*"), key=os.path.getmtime)
    if not hits:
        print("❌ 找不到产物目录")
        return 1
    wavs = sorted(glob.glob(os.path.join(hits[-1], "*.mp3")) +
                  glob.glob(os.path.join(hits[-1], "*.wav")),
                  key=os.path.getmtime)
    src = wavs[-1]
    mp3 = "/mnt/tts/diag/l1_53.mp3"
    shutil.copy2(src, mp3)
    e = metrics(EL)
    o = metrics(mp3)
    print("EL   中频 %.2f%%" % e, flush=True)
    print("我们  中频 %.2f%%   (%.2f×EL)" % (o, o / e), flush=True)
    if 0.95 <= o / e <= 1.10:
        print("✅ EQ 生效（中频已对齐 EL）")
        return 0
    print("❌ EQ 可能没生效（中频仍偏离 EL）—— 查 service.log 有没有 TypeError")
    return 1


if __name__ == "__main__":
    sys.exit(main())
