# -*- coding: utf-8 -*-
"""🔴 验证「电音感」是否来自**合成底噪**（受控实验的行为侧验证）

## 背景（2026-10-15 第24轮反馈）
3 条独立证据都指向「电音感」：
  1. 「像真人录的还是电子合成」→ **③ 明显像电子合成的 2/2**
  2. 优先级题 **2/2 排第一**
  3. 고구마 **连续两轮**自由文本都写「电音严重」

代码层最可能的来源：`app/roomtone.py` 的 `gen_noise()` 用
**FFT 塑形合成粉噪**当底噪——纯数学信号、无真实录音痕迹、平稳无起伏，
天然带"电音感"。EL 的底噪来自真实录音环境。

## 本脚本干什么
**受控实验**：同文案、同 seed、唯一变量 = `TTS_ROOMTONE=0`，各生成一份。
然后测**行为差异**（不读配置、不靠推断）：

| 指标 | 关底噪后应该怎么变 |
|---|---|
| **静音段电平** | 明显下降（接近数字真零）—— 这是底噪关掉的直接证据 |
| 平坦度 flat | 变化（合成粉噪被去掉） |
| 各频带 | 低频（60~150）变化最明显 |

🔴 **为什么必须测静音段**：这是唯一能证明「底噪真的关掉了」的判据。
配置文件改了不等于生效——我这次就踩了：改完 roomtone.conf 直接 restart，
systemd 没 daemon-reload，**配置根本没生效**，
靠 `strings /proc/<MainPID>/environ` 才看出来。

## 🔴 三条铁律（复用）
1. **只测长文案**（≥180s）
2. 语速必须取自**同一条 job 的 chunks[]**（分子分母不能来自不同文本）
3. 短片段不能当验收

## 用法
    python verify_roomtone_off.py <开底噪版> <关底噪版>
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

SR = 16000
LONG_MIN = 180
PROBE = "/mnt/tts/diag/x8197/probe_g.py"
VENV = "/mnt/tts/projects/qwen3-tts/venv/bin/python3"


def to16(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    import soundfile as sf
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y


def dur_of(p):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", p], capture_output=True,
        text=True).stdout.strip() or 0)


def silent_floor(y):
    """静音段电平：按每 100ms 分段，取 RMS 最低的那些段的中位。

    这就是「底噪电平」的直接度量——真人录音也有底噪，
    但合成的粉噪是**平稳的**，且关掉后会掉到接近数字真零。
    """
    n = SR // 10
    m = len(y) // n
    if m < 10:
        return None
    seg = np.array([np.sqrt(np.mean(y[i * n:(i + 1) * n] ** 2)) + 1e-12
                    for i in range(m)])
    db = 20 * np.log10(seg)
    floor = float(np.percentile(db, 5))          # 最低的 5% 段
    p50 = float(np.median(db))
    # 时间起伏：底噪段的 RMS 标准差（合成的是平稳的，真人有起伏）
    q = db[db < np.percentile(db, 20)]            # 最低 20% 的段
    wobble = float(q.std())
    return {"floor_db": round(floor, 1),
            "median_db": round(p50, 1),
            "floor_wobble_db": round(wobble, 2)}


def bands(y):
    import librosa
    y24 = to16(y) if False else None
    return None


def probe(p):
    out = subprocess.run([VENV, PROBE, "male", p],
                         capture_output=True, text=True, timeout=1200).stdout
    r = {}
    for line in out.splitlines():
        if "Σz²=" in line:
            try:
                r["sigz2"] = float(line.split("Σz²=")[1].split()[0])
            except (IndexError, ValueError):
                pass
        for key, nm in (("150~300", "b150_300"), ("4k~8k", "b4000_8000"),
                        ("平坦度", "flat"), ("60~150", "b60_150")):
            if key in line and "我们" in line:
                p2 = line.split()
                try:
                    r[nm] = float(p2[3])
                except (IndexError, ValueError):
                    pass
    return r


def main():
    if len(sys.argv) < 3:
        sys.exit("用法: verify_roomtone_off.py <开底噪版> <关底噪版>")
    files = sys.argv[1:3]
    res = []
    for f in files:
        dd = dur_of(f)
        if dd and dd < LONG_MIN:
            print("🔴 %s 只有 %.0fs（<%ds），短文案不能用于验证"
                  % (os.path.basename(f), dd, LONG_MIN))
            return
        y = to16(f)
        row = {"file": os.path.basename(f), "dur": round(dd, 1)}
        row.update(silent_floor(y))
        row.update(probe(f))
        res.append(row)
        print("\n%s (%.1fs)" % (row["file"], row["dur"]))
        print("  静音段电平 floor=%s dB  中位=%s dB  底噪起伏σ=%s dB"
              % (row.get("floor_db"), row.get("median_db"),
                 row.get("floor_wobble_db")))
        print("  Σz²=%s  60~150=%s  150~300=%s  4k~8k=%s  flat=%s"
              % (row.get("sigz2"), row.get("b60_150"), row.get("b150_300"),
                 row.get("b4000_8000"), row.get("flat")))

    if len(res) == 2:
        a, b = res
        print("\n=== 对照（A=%s / B=%s）===" % (a["file"], b["file"]))
        d = {}
        for k, lab in (("floor_db", "静音段电平 dB"),
                       ("floor_wobble_db", "底噪起伏σ dB"),
                       ("sigz2", "Σz²"), ("b60_150", "60~150"),
                       ("flat", "平坦度")):
            if a.get(k) is None or b.get(k) is None:
                continue
            d[lab] = round(b[k] - a[k], 2)
            print("  %-16s %8s → %-8s  Δ=%+.2f"
                  % (lab, a.get(k), b.get(k), d[lab]))
        # 判定：底噪真的关掉了吗
        ok = (a.get("floor_db") is not None and b.get("floor_db") is not None
              and b["floor_db"] < a["floor_db"] - 6)
        print("\n%s 底噪%s"
              % ("✅" if ok else "🔴",
                 "确实被关掉了（静音段掉 >6dB）" if ok
                 else "**没有明显变化**，实验无效——查 TTS_ROOMTONE 是否真生效"))
        out = "/mnt/tts/diag/muffle/verify_roomtone.json"
        json.dump({"rows": res, "delta": d, "passed": ok},
                  open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("-> %s" % out)


if __name__ == "__main__":
    main()