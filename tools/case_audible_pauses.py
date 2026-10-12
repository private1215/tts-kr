# -*- coding: utf-8 -*-
"""重测每条案例的**可听停顿**分布，写回 cases.json 的 pause_samples_audible。

🔴 为什么要有这个工具（2026-10-12 卡顿事故）
--------------------------------------------
生产管线 PBA reshuffle 的目标分布取自 cases.json 里的 `pause_samples`——
那是入库时用 prosody_report.sent_windows 口径量的。实测发现这个口径的
「停顿」大部分**不是静音**：女声标杆 00b6a45ad6aa 的 154 个「停顿」里只有
2 个低于 -45dB，中位 RMS 是 -40dB——那是句间**有声的轻音**（呼吸/气声/
室底），不是静默。 reshuffle 用 _fit_seg 把我们的停顿拉到那个长度时，
填进去的是数字静音，成品于是出现案例根本没有的 0.6~0.9s 死寂断档：
    女声成品 126s 内 14 个 >=0.6s 间隙（案例 459s 只有 2 个）
而同一篇文案的**裸推理**输出只有 1 个长间隙、静音占比 2.7%（案例 3.3%）——
员工盲听 6/7 票选的就是裸推理那版。是后处理把好节奏改坏了。

所以目标分布必须换成**可听停顿**（能量口径），本工具就是干这个的。

口径（与 /tmp/gapscan2.py 诊断时完全一致，勿改）：
    ffmpeg 转 16k 单声道 → 20ms 帧 / 10ms 跳距 RMS → dB > -45dB 为有声
    → 内部静音段 >=0.25s 记一个停顿（首尾的起振/收尾不算）
    -45dB 是「安静房间里听不见」的经验界：案例句间轻音在 -40dB 左右
    （有声，不计），真静音在 -50dB 以下（计）。

🔴 为什么单独放一个文件，不写回 cases.json
----------------------------------------
cases.json 属 root，服务进程内存里也有一份并且会回写（_analyze_case 等），
外部写进去的字段有被覆盖/冲突的风险。可听停顿是**派生数据**，放独立文件
/mnt/tts/data/pause_targets_audible.json（{案例id: [停顿时长...]}），
server.py 的 pause_targets() 优先读它，读不到就回落旧的 pause_samples。
data 目录组可写，不需要 root。

输出：/mnt/tts/data/pause_targets_audible.json
server.py 的 pause_targets() 优先读它；读不到的案例自动跳过。
噪底 > -44dB 的案例（底噪太响，能量口径不可信）会打警告且不写字段。

用法：
    python tools/case_audible_pauses.py            # 全量重测并写回
    python tools/case_audible_pauses.py --dry-run  # 只看会测什么
    python tools/case_audible_pauses.py --id 00b6a45ad6aa   # 只测一条
新增案例入库后重跑一次即可；老案例数据不变，可随时重跑。
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")
CASES_JSON = os.environ.get("TTS_CASES_JSON", "/mnt/tts/data/cases.json")
OUT_JSON = os.environ.get(
    "TTS_PAUSE_AUDIBLE_JSON", "/mnt/tts/data/pause_targets_audible.json")

SR = 16000
FRAME = 0.02          # 20ms
HOP = 0.01            # 10ms
ABS_DB = -45.0        # 有声阈值（绝对）
MIN_GAP = 0.25        # 短于此不计停顿
FLOOR_WARN = -44.0    # 噪底高于此值打警告（口径不可信）


def _to16k(path):
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", str(SR), "-ac", "1", tmp], check=True)
    import soundfile as sf
    a, sr = sf.read(tmp)
    os.unlink(tmp)
    return a.astype(np.float64), sr


def audible_pauses(path):
    """返回 (停顿时长列表, 噪底dB, 音频时长s)。"""
    a, sr = _to16k(path)
    fl = int(FRAME * sr)
    hop = int(HOP * sr)
    n = max(1, (len(a) - fl) // hop)
    rms = np.array([np.sqrt(np.mean(a[i * hop:i * hop + fl] ** 2) + 1e-12)
                    for i in range(n)])
    db = 20 * np.log10(rms + 1e-10)
    floor = float(np.median(np.sort(db)[:max(1, len(db) // 5)]))
    voiced = db > ABS_DB
    gaps = []
    i = 0
    while i < n:
        if not voiced[i]:
            j = i
            while j < n and not voiced[j]:
                j += 1
            dur = (j - i) * hop / sr
            if i > 0 and j < n and dur >= MIN_GAP:
                gaps.append(round(dur, 3))
            i = j
        else:
            i += 1
    return gaps, floor, len(a) / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--id", default="")
    a = ap.parse_args()

    data = json.load(open(CASES_JSON, encoding="utf-8"))
    cases = data if isinstance(data, dict) else {c["id"]: c for c in data}
    print("案例库 %d 条 -> %s" % (len(cases), OUT_JSON))

    out = {}
    n_ok = n_warn = n_miss = 0
    for cid, c in sorted(cases.items()):
        if a.id and cid != a.id:
            continue
        wav = os.path.join(CASES_DIR, cid, "merged.wav")
        if not os.path.exists(wav):
            print("  %s 缺 merged.wav，跳过" % cid)
            n_miss += 1
            continue
        if a.dry_run:
            print("  会测 %s (%s)" % (cid, c.get("gender")))
            continue
        try:
            gaps, floor, dur = audible_pauses(wav)
        except Exception as e:
            print("  %s 失败: %r" % (cid, e))
            n_warn += 1
            continue
        long_n = sum(1 for g in gaps if g >= 0.6)
        if floor > FLOOR_WARN:
            print("  %s %-6s %6.1fs 可听停顿%4d 个 (>=0.6s: %2d) 噪底%6.1fdB"
                  " ⚠️ 噪底过高，不写字段"
                  % (cid, c.get("gender") or "?", dur, len(gaps), long_n,
                     floor))
            n_warn += 1
            continue
        out[cid] = gaps
        n_ok += 1
        print("  %s %-6s %6.1fs 可听停顿%4d 个 (>=0.6s: %2d) 噪底%6.1fdB"
              % (cid, c.get("gender") or "?", dur, len(gaps), long_n, floor))

    if not a.dry_run:
        tmp = OUT_JSON + ".tmp"
        json.dump(out, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
        os.replace(tmp, OUT_JSON)
        print("\n已写 %s：%d 条有可听停顿字段，%d 条警告，%d 条缺文件"
              % (OUT_JSON, n_ok, n_warn, n_miss))
        print("⚠️ 服务内存里的 cases 是启动时加载的，pause_targets() 读的是"
              "这个独立文件（每次调用现读，有 mtime 缓存），改 server.py 后"
              "重启即生效；重跑本工具不用重启。")


if __name__ == "__main__":
    main()
