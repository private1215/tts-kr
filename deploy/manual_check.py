#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证「手动输入文案」这条路径确实走最新生产链路。

为什么要单独验：
    用户关心「手动输入文案生成语音要保持最新的模型」。
    光看代码不够 —— 要真提交一条、下载成品、量一遍，
    确认它落在 EL 基线附近（即用了最新 EQ），而不是还挂着旧参数。

做法：拿 EL 案例 8197 的原文前若干字提交（与 EL 标杆同文案，可比），
跑完后测频段 z、F0、F1/F2，和 v10/v7 一起看。
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"
TOK = os.environ.get("TTS_DEBUG_TOKEN", "smoke-8f3a2c91")
CASE_TXT = "/mnt/tts/data/cases/8197d9ac8a35/full.txt"
CHARS = int(os.environ.get("CHARS", "1200"))

BASE_BANDS = {
    "b60_150": (0.0186, 0.0020), "b150_300": (0.2541, 0.0051),
    "b300_500": (0.3394, 0.0122), "b500_1000": (0.1485, 0.0076),
    "b1000_2000": (0.1488, 0.0085), "b2000_4000": (0.0766, 0.0041),
    "b4000_8000": (0.0129, 0.0006),
}
BANDS = [(60, 150), (150, 300), (300, 500), (500, 1000),
         (1000, 2000), (2000, 4000), (4000, 8000)]


def req(path, data=None):
    if data is None:
        r = urllib.request.Request(BASE + path,
                                   headers={"Authorization": "Bearer " + TOK})
    else:
        body = urllib.parse.urlencode(data).encode("utf-8")
        r = urllib.request.Request(
            BASE + path, data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "Authorization": "Bearer " + TOK})
    return json.loads(urllib.request.urlopen(r, timeout=180).read())


def main():
    text = open(CASE_TXT, encoding="utf-8").read().strip()[:CHARS]
    print("提交 %d 字（来自 EL 案例 8197 原文）" % len(text), flush=True)
    r = req("/api/jobs", {"voice": "EL_male", "text": text, "mode": "clone",
                          "language": "Korean", "title": "manual-check",
                          "emotion": "standard"})
    jid = r.get("id") or r.get("job", {}).get("id")
    print("job id =", jid, flush=True)
    for _ in range(90):
        time.sleep(5)
        d = req("/api/jobs/" + jid)
        if d.get("status") == "done":
            break
        if d.get("status") == "error":
            print("生成失败:", d.get("error"))
            return 1
    else:
        print("超时未完成")
        return 1

    st = d.get("stats") or {}
    files = d.get("files") or []
    print("stats: engine=%s cps=%s chunks=%s"
          % (st.get("plan", {}).get("engine"), st.get("cps"),
             st.get("chunks")), flush=True)
    if not files:
        print("无成品")
        return 1
    rel = files[0].get("path") or files[0].get("name")
    url = "%s/api/download/%s" % (BASE, rel.lstrip("/"))
    out = "/mnt/tts/diag/x8197/manual_check.mp3"
    # urlretrieve 不接受 Request 对象，必须 urlopen 后自己写盘
    with urllib.request.urlopen(
            urllib.request.Request(
                url, headers={"Authorization": "Bearer " + TOK}),
            timeout=300) as resp:
        open(out, "wb").write(resp.read())
    print("下载 -> %s" % out, flush=True)

    import numpy as np
    import librosa
    import soundfile as sf
    import parselmouth
    sys.path.insert(0, "/mnt/tts/diag/x8197")
    from fix_chain import load16

    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", out,
                    "-ar", "24000", "-ac", "1", t], check=True)
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=24000, n_fft=2048)
    tot = S.sum() + 1e-12
    print("\n%-12s %9s %9s %8s" % ("维度", "手动生成", "EL均值", "z"))
    J = 0.0
    for lo, hi in BANDS:
        k = "b%d_%d" % (lo, hi)
        v = float(S[(f >= lo) & (f < hi)].sum() / tot)
        m, s = BASE_BANDS[k]
        z = (v - m) / s
        J += z * z
        print("%-12s %9.4f %9.4f %+8.2f" % (k, v, m, z))
    print("频段 J = %.1f" % J)

    y16, _ = load16(out)
    fl = float(librosa.feature.spectral_flatness(
        y=y16[:60 * 16000], hop_length=256)[0].mean())
    print("flat(16k) = %.4f   (EL 0.0461)" % fl)

    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", out,
                    "-ar", "16000", "-ac", "1", t], check=True)
    snd = parselmouth.Sound(t)
    os.unlink(t)
    pit = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    fm = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                             maximum_formant=5000)
    ts = np.arange(0.05, snd.duration - 0.05, 0.01)
    F0, F1, F2 = [], [], []
    for tt in ts:
        f0 = pit.get_value_at_time(tt)
        if f0 is None or np.isnan(f0) or f0 <= 0:
            continue
        a = fm.get_value_at_time(1, tt)
        b = fm.get_value_at_time(2, tt)
        if a is None or b is None or np.isnan(a) or np.isnan(b) or a <= 0 \
                or b <= 0:
            continue
        F0.append(f0)
        F1.append(a)
        F2.append(b)
    print("F0=%.1f F1=%.1f F2=%.1f   (EL 186.5 / 409.8 / 1555.4)"
          % (np.median(F0), np.median(F1), np.median(F2)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
