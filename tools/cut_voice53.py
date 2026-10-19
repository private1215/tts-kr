# -*- coding: utf-8 -*-
"""cut_voice53.py —— 从案例库切出被误删的 elara / yo han 音色模版，写回 voices.json。

## 背景
2026-10-18 员工误删这两个音色（连 sources/*__full.wav 一起清掉了）。
用户纠正：**它们不是员工上传的录音**，原本的上传克隆音色早已淘汰，
这两个是**从案例库语音里切出来的** → 案例库（36 条 EL 原声）还在，能重建。

## 来源（已用双证据锁定）
| 音色 | 源 | 证据①时长 | 证据②文本 |
|---|---|---|---|
| elara | `a3d39a095bac/seg0.wav` | 173.087s = 备份 src_dur **173.09** | 独有词 `겉모습만` 全库唯一命中 |
| yo han | `799e4d7060ce/seg0.wav` | 261.878s = 备份 src_dur **261.88** | ref_text 命中 full.txt / seg1.txt |

## 🔴🔴 关键设计：不照抄旧 ref_text
备份里 yo han 的 ref_text 是 **191 字 / 8.66s = 22 字每秒** —— 物理上念不出来。
GS 硬坑 3：ref_text 比音频长 → 语速崩 + **把参考文本当正文反复念**（CER 0.66 事故，
正是员工报的「说话慢 + 乱说话」）。
→ **切窗后用窗内 ASR 结果重新生成对齐的 ref_text**（`--reftext-asr`），
   或由调用方显式给 `--ref-text`。默认会先打印 ASR 结果让你确认。

## 用法
    # 先 dry-run：切窗 + 转写 + 打印（不写 voices.json）
    venv/bin/python tools/cut_voice53.py --dry
    # 确认后落盘
    venv/bin/python tools/cut_voice53.py --apply

🔴 写 voices.json 后**必须重启 qwen-tts**（否则任务结束时内存音色表写回会冲掉）。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

CASES = "/mnt/tts/data/cases"
VOICES = "/mnt/tts/data/voices/voices.json"
SR_OUT = 44100

# (name, gender, engine, 源相对路径, start, end)
JOBS = [
    ("elara", "female", "qwen", "a3d39a095bac/seg0.wav", 14.36, 22.50),
    ("yo han", "male", "qwen", "799e4d7060ce/seg0.wav", 136.02, 144.68),
]


def asr(path):
    """🔴 两条都必须照做，否则 faster_whisper 打不开音频：
    ① 先降到 16k 单声道；② **传 numpy 数组而不是文件路径** —— 本机 av 版本与
    faster_whisper 不兼容，传路径会 `TypeError: open() got an unexpected keyword
    argument 'metadata_errors'`（asrseg53.py 就是传 ndarray 绕开的）。"""
    import subprocess
    import tempfile

    import os

    import numpy as np
    import soundfile as sf

    import faster_whisper
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", tmp], check=True)
    x, _ = sf.read(tmp, dtype="float32")
    os.unlink(tmp)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.ascontiguousarray(x)
    # 🔴 用 CPU：8 秒短片段 CPU 足够快；CUDA 在这台机器上（GPU1 已占 12.5G）
    #   初始化会长时间挂住（2026-10-19 实测卡了 4 分钟无输出）。
    dev = "cuda" if os.environ.get("TTS_ASR_GPU") == "1" else "cpu"
    ct = "float16" if dev == "cuda" else "int8"
    # 🔴🔴 `local_files_only=True` 必须加：否则 faster_whisper 会去连 HuggingFace
    #   校验模型，服务器无外网 → **静默挂死**（2026-10-19 卡了 4 分钟无输出）。
    #   模型缓存已在 ~/.cache/huggingface/hub/models--Systran--faster-whisper-small
    w = faster_whisper.WhisperModel("small", device=dev, compute_type=ct,
                                    local_files_only=True)
    segs, _ = w.transcribe(x, language="ko", beam_size=5)
    return " ".join(s.text for s in segs).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真的写入（默认 dry）")
    a = ap.parse_args()

    out = {}
    for name, gender, engine, rel, s, e in JOBS:
        src = "%s/%s" % (CASES, rel)
        if not os.path.exists(src):
            print("❌ %s 源不存在：%s" % (name, src))
            continue
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", src], capture_output=True, text=True
        ).stdout.strip())
        # 🔴 `sources/` 是 root 属主，zheng 写不进去。现有音色（EL_female.wav 等）
        #    就直接放在 voices/ 这一层，跟着这个约定走。
        dst = "/mnt/tts/data/voices/%s.wav" % name.replace(" ", "_")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                        "-ss", str(s), "-t", str(e - s),
                        "-ar", str(SR_OUT), "-ac", "1", dst], check=True)
        txt = asr(dst)
        cps = len(txt) / (e - s)
        print("== %s ==" % name, flush=True)
        print("   源    : %s (%.3fs)  %s ~ %s" % (rel, dur, s, e))
        print("   切片  : %s  %.2fs @ %dHz" % (dst, e - s, SR_OUT))
        print("   ASR   : %s" % txt)
        print("   %d 字 / %.2fs = **%.2f 字每秒**" % (len(txt), e - s, cps))
        if cps > 12:
            print("   ⚠️ 语速异常高，ref_text 可能偏长，慎用")
        print(flush=True)
        out[name] = dict(gender=gender, engine=engine, ref_audio=dst,
                         ref_text=txt, duration=round(e - s, 2),
                         origin="cases", src_case=rel.split("/")[0],
                         clip=[s, e])

    if not a.apply:
        print("\n[dry-run] 未写 voices.json。确认上面 ASR 无误后加 --apply")
        return 0

    bak = VOICES + ".bak-before-cutvoice53-" + datetime.now().strftime(
        "%Y%m%d_%H%M%S")
    shutil.copy2(VOICES, bak)
    d = json.load(open(VOICES))
    vs = d["voices"] if isinstance(d, dict) and "voices" in d else d
    isdict = isinstance(vs, dict)
    if isdict:
        vs = dict(vs)
    for name, v in out.items():
        if isdict:
            vs[name] = v
        else:
            vs = [x for x in vs if x.get("name") != name]
            vs.append(dict(v, name=name))
    tmp = VOICES + ".tmp"
    if isdict:
        d["voices"] = vs
        json.dump(d, open(tmp, "w"), ensure_ascii=False, indent=2)
    else:
        json.dump({"voices": vs}, open(tmp, "w"), ensure_ascii=False, indent=2)
    os.replace(tmp, VOICES)
    # 回读自检
    d2 = json.load(open(VOICES))
    v2 = d2["voices"] if isinstance(d2, dict) and "voices" in d2 else d2
    print("\n✅ 已写入（备份 %s），回读音色数 = %d：%s"
          % (os.path.basename(bak), len(v2),
             ", ".join(sorted(v2.keys() if isinstance(v2, dict)
                              else [x.get("name") for x in v2]))))
    print("🔴 现在必须重启 qwen-tts，否则内存音色表会写回冲掉新音色")
    return 0


if __name__ == "__main__":
    sys.exit(main())
