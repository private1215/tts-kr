# -*- coding: utf-8 -*-
"""head_check.py —— 细查文件头 1.6s（员工报「0秒 开头 怪音」）。

已知：10ms 包络显示 0.07~0.28s 有一个 ~210ms 的孤立突起（峰值 -26dBFS），
随后 0.28~1.25s 掉到 -50dB（接近静音），1.25s 之后才进入正常语音。
这和学生反馈的「开头怪音」位置吻合 → 需要判定这个突起是什么：
  · 浊音（有谐波）→ 是一个被切断/孤立的音节
  · 噪声（无谐波、宽带）→ 爆音/咔哒
  · 有谐波且后面接静音 → 首字被「咬掉」只剩瞬态
"""
import os
import subprocess

import numpy as np
import soundfile as sf

MP3 = "/mnt/tts/output/202610190902/202610190902-1.mp3"
WORK = "/mnt/tts/diag/v0902"
SR = 16000


def to16k(path, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", out], check=True)
    return out


def desc(x, sr, t0, t1):
    a, b = int(t0 * sr), int(t1 * sr)
    s = x[a:b]
    if len(s) < 32:
        return "太短"
    S = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    fr = np.fft.rfftfreq(len(s), 1.0 / sr)
    tot = S.sum() + 1e-12
    zcr = float(np.mean(np.abs(np.diff(np.sign(s))) > 0))
    # 谐波性：谱峰个数（浊音有密集等距峰）
    pk = S / (tot + 1e-12)
    top = np.sort(pk)[::-1][:12].sum()
    f0 = fr[np.argmax(S)]
    return ("峰值%.1fdB  主频%.0fHz  zcr=%.3f  低(<400)%.2f  "
            "中(400-3k)%.2f 高(>3k)%.2f  谱集中度%.3f"
            % (20 * np.log10(np.max(np.abs(s)) + 1e-12), f0, zcr,
               S[fr < 400].sum() / tot,
               S[(fr >= 400) & (fr < 3000)].sum() / tot,
               S[fr >= 3000].sum() / tot, top))


def main():
    wav = os.path.join(WORK, "x.wav")
    if not os.path.exists(wav):
        to16k(MP3, wav)
    x, sr = sf.read(wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    hop = int(0.01 * sr)
    print("=== 文件头 1.8s 的 10ms 包络（dBFS）===")
    vals = []
    for i in range(int(1.8 / 0.01)):
        seg = x[i * hop:(i + 1) * hop]
        vals.append(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)
                    if len(seg) else -120)
    for k in range(0, len(vals), 60):
        print("  %5.2fs " % (k * 0.01) +
              " ".join("%5.0f" % v for v in vals[k:k + 60]))

    print("\n=== 分段特征 ===")
    for t0, t1, note in [
        (0.00, 0.07, "数字静音（前导）"),
        (0.07, 0.28, "孤立突起 A"),
        (0.28, 0.60, "突起后的衰减段"),
        (0.60, 1.24, "接近静音的段 B"),
        (1.24, 1.60, "正常语音起点 C"),
        (1.60, 2.20, "正常语音 D"),
        (3.90, 4.40, "对照：5s 附近的正常语音"),
    ]:
        print("  %.2f~%.2fs  %-16s %s" % (t0, t1, note, desc(x, sr, t0, t1)))

    print("\n=== 孤立突起 A 的 20ms 子段（看是不是被切断的音节）===")
    t = 0.06
    while t < 0.34:
        print("  %.3f~%.3f  %s" % (t, t + 0.02, desc(x, sr, t, t + 0.02)))
        t += 0.02


if __name__ == "__main__":
    main()
