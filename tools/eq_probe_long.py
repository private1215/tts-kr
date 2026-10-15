# -*- coding: utf-8 -*-
"""🔴 候选 EQ 受控实验：262Hz + 185Hz 补频（长文案口径）。

## 背景（2026-10-15 排查结论）
修好验收口径后（全篇 + 稳健σ + EL 自检合格线），重测发现一个**此前完全没看见的事实**：

| 链路 | Σz²（全篇口径） | 150~300 |
|---|---|---|
| **老链路 r26a/r26d**（2026-10-13） | **27.5 ~ 61.9**（在 EL 分布内） | 0.2565（EL=0.2541，几乎完美） |
| **当前链路**（VoxCPM2 + 分窗变调） | **863 ~ 948**（远超 EL 上界 240） | 0.1838（−28σ） |

**当前链路比老链路差 20 倍。** 唯一系统性差异是**分窗变调**：
`asetrate` 把整个频谱搬高 R 倍 → 150~300 的能量被搬到 300~600。
实测证据：150~300 掉 28%（0.2565→0.1838），300~500 涨 9%（0.3559→0.3885），
比值从 0.720 掉到 0.473。

⚠️ 记忆里「变调后频谱被整体搬高，男声 EQ 高频档要从 −4dB 加深到 −9.5dB」
——**当时只处理了高频（变亮），完全没处理低频被搬走（变闷）**。
这就是「闷」的真实成因，而且它是**变调的必然代价，不是引擎缺陷**。

## 本脚本做什么
在**长文案**（≥3 分钟，见用户铁律：短文案验收没意义）上，
逐档扫描 150~300 覆盖的频段（185Hz=4b / 262Hz=5b），输出收益曲线。
**只加 EQ、不动其他**，确保是单变量。

## 🔴 三条硬约束（历史上都栽在这）
1. **必须长文案**（≥180s）。短切片本来就漂亮，用它验收等于没验收。
2. **必须端到端重生成验证**。底噪斜率那次离线 Σz² 17.3→ 端到端 32.4，方向反了。
3. **必须看沿时间轴稳定性**（`probe_g --segments`），不只看全篇 Σz²。
   4k~8k 沿时间轴极差：我们 94.9% vs EL 31.9% —— 这才是长篇的真差距。

## 用法
    python eq_probe_long.py <长篇成品.mp3> [...]
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = "/mnt/tts/diag/x8197"
VENV = "/mnt/tts/projects/qwen3-tts/venv/bin/python3"
LONG_MIN = 180

# 候选集：逐档加，避免一次性堆多档（历史教训：低频补太多会抬底噪）
CANDIDATES = [
    ("baseline",          None),
    ("5b=1.178",          "superequalizer=5b=1.178"),           # 现役档位值
    ("5b=1.178:4b=1.138", "superequalizer=5b=1.178:4b=1.138"),
    ("5b=1.300:4b=1.200", "superequalizer=5b=1.300:4b=1.200"),  # 候选甜点
    ("5b=1.300",          "superequalizer=5b=1.300"),
    ("5b=1.450:4b=1.250", "superequalizer=5b=1.450:4b=1.250"),  # 过量对照
]


def dur(path):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True,
        text=True).stdout.strip() or 0)


def probe(path, segments=False):
    cmd = [VENV, "%s/probe_g.py" % DIAG, "male", path]
    if segments:
        cmd.append("--segments")
    out = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=1200).stdout
    return out


def sigz2(out):
    for line in out.splitlines():
        if "Σz²=" in line:
            return float(line.split("Σz²=")[1].split()[0])
    return None


def band(out, key):
    for line in out.splitlines():
        if key in line and "我们" in line:
            p = line.split()
            try:
                return float(p[3]), float(p[-1])
            except (IndexError, ValueError):
                return None, None
    return None, None


def stability(out):
    """从 --segments 输出里抽 150~300 与 4k~8k 的极差%"""
    res = {}
    grab = False
    for line in out.splitlines():
        if "长文案稳定性" in line:
            grab = True
            continue
        if grab:
            if line.strip().startswith("跨窗"):
                break
            p = line.split()
            # 输出是 4 列：维度 中位 极差% CV%
            if len(p) >= 4 and p[0] in ("150~300", "4k~8k", "300~500"):
                try:
                    res[p[0]] = float(p[2].rstrip("%"))
                except ValueError:
                    pass
    return res


def main():
    files = sys.argv[1:]
    if not files:
        sys.exit("用法: eq_probe_long.py <长篇成品.mp3> [...]")
    rows = []
    for f in files:
        d = dur(f)
        if d < LONG_MIN:
            print("⚠️ 跳过 %s（仅 %.0fs < %ds，短文案验收没意义）"
                  % (os.path.basename(f), d, LONG_MIN))
            continue
        print("\n" + "=" * 74)
        print("%s  (%.0fs)" % (os.path.basename(f), d))
        base_out = probe(f)
        base_seg = probe(f, segments=True)
        b_stab = stability(base_seg)
        b_j = sigz2(base_out)
        b_lo, b_loz = band(base_out, "150~300")
        b_hi, _ = band(base_out, "4k~8k")
        print("  %-22s Σz²=%8s  150~300=%.4f(%s)  4k~8k=%.4f  稳定性=%s"
              % ("baseline(不动)", b_j, b_lo or 0, b_loz, b_hi or 0,
                 b_stab or {}))
        rows.append({"file": os.path.basename(f), "cand": "baseline",
                     "sigz2": b_j, "b150_300": b_lo, "b150_300_z": b_loz,
                     "b4000_8000": b_hi, "stability": b_stab})
        for nm, af in CANDIDATES:
            if af is None:
                continue
            t = tempfile.mktemp(suffix=".wav")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", f,
                            "-af", af, "-ar", "16000", "-ac", "1", t],
                           check=True)
            out = probe(t)
            seg = probe(t, segments=True)
            st = stability(seg)
            j = sigz2(out)
            lo, loz = band(out, "150~300")
            hi, _ = band(out, "4k~8k")
            d1 = (j - b_j) if (j and b_j) else None
            print("  %-22s Σz²=%8s (%+7.1f)  150~300=%.4f(%s)  4k~8k=%.4f  稳定性=%s"
                  % (nm, j, d1 or 0, lo or 0, loz, hi or 0, st or {}))
            rows.append({"file": os.path.basename(f), "cand": nm,
                         "sigz2": j, "delta": d1, "b150_300": lo,
                         "b150_300_z": loz, "b4000_8000": hi, "stability": st})
            os.unlink(t)
    out = "/mnt/tts/diag/muffle/eq_probe_long.json"
    json.dump(rows, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n-> 写入", out)


if __name__ == "__main__":
    main()