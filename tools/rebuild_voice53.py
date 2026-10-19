# -*- coding: utf-8 -*-
"""rebuild_voice53.py —— 从案例库反推并重建被误删的音色切片。

2026-10-18 员工误删 elara / yo han，连 `sources/*__full.wav` 一起清掉。
用户说明：这两个**不是**员工上传的录音（原始上传克隆音色早已淘汰），
而是从**案例库语音**里切出来的 → 案例库（37 条 EL 原声）没被删，可反推。

## 定位方法
① 归一化（去空白/标点）后，用 ref_text 的**前缀**在所有 case 的
   full.txt / seg*.txt 里找（前缀逐级缩短：40→30→24→20 字，
   因为转写/切分可能导致个别字不同）
② 命中后按**字符比例**粗估时间：
     t_in_seg = (idx / len(norm(seg.txt))) * seg_dur
     t_merged = sum(前面各 seg 时长) + t_in_seg
③ 用 whisper 在 ±20s 内以 1s 步长滑窗校验，挑与 ref_text 最像的窗口
   （字符比例估时会有几十秒误差，必须声学校验）
④ 切出等长切片，**重新用 ASR 生成对齐的 ref_text**
   🔴 旧的 ref_text 可能本身就是错的：yo han 191 字 / 8.66s = 22 字/秒，
      物理上念不出来（GS 硬坑 3：ref_text 比音频长 → 语速崩 + 乱念参考文本），
      照抄只会把坑带回来。

用法: venv/bin/python -u tools/rebuild_voice53.py --name "yo han" [--apply]
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile

CASES = "/mnt/tts/data/cases"
VOICES_BAK = "/mnt/tts/data/voices/voices.json.bak-before-omnimale53"
VOUT = "/mnt/tts/data/voices"


def norm(s):
    return re.sub(r"[\s\.,!?\"'·…\-\–—()\[\]【】]", "", s or "")


def dur(p):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", p],
        capture_output=True, text=True).stdout.strip())


def find_in_cases(ref):
    """返回 (case_id, seg文件名, 字符下标)。"""
    for plen in (40, 30, 24, 20):
        pre = norm(ref)[:plen]
        if len(pre) < 20:
            break
        for cid in sorted(os.listdir(CASES)):
            d = os.path.join(CASES, cid)
            if not os.path.isdir(d):
                continue
            segs = sorted(f for f in os.listdir(d)
                          if f.endswith(".wav") and f != "merged.wav")
            for sfn in segs:
                tf = os.path.join(d, sfn[:-4] + ".txt")
                if not os.path.exists(tf):
                    continue
                nt = norm(open(tf, encoding="utf-8").read())
                i = nt.find(pre)
                if i >= 0:
                    return cid, sfn, i, len(nt), plen, segs
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    bak = json.load(open(VOICES_BAK, encoding="utf-8"))
    src = bak.get(a.name)
    if not src:
        print("备份里没有 %s" % a.name)
        return 1
    ref = src.get("ref_text") or ""
    target_dur = float(src.get("dur") or 8.0)
    print("== %s ==\nref_text %d 字 / dur %.2fs → %.1f 字每秒"
          % (a.name, len(ref), target_dur, len(ref) / target_dur))

    hit = find_in_cases(ref)
    if not hit:
        print("❌ 案例库里定位不到（前缀 40/30/24/20 都不命中）"
              " → 无法从案例库重建")
        return 2
    cid, sfn, idx, ntot, plen, segs = hit
    d = os.path.join(CASES, cid)
    print("命中：case %s / %s，前缀 %d 字 @ 第 %d 字（该文 %d 字）"
          % (cid, sfn, plen, idx, ntot))
    base = sum(dur(os.path.join(d, s)) for s in segs if s < sfn)
    segd = dur(os.path.join(d, sfn))
    t0 = base + (idx / float(ntot)) * segd
    print("粗估位置 %.2fs（前序段 %.2fs + 段内 %.2fs）"
          % (t0, base, t0 - base))

    from faster_whisper import WhisperModel
    m = WhisperModel("small", device="cpu", compute_type="int8",
                     local_files_only=True)

    def clip_text(t):
        w = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i",
                        os.path.join(d, "merged.wav"), "-ss", str(t),
                        "-t", str(target_dur), "-ar", "16000", "-ac", "1",
                        w], check=True)
        segs_, _ = m.transcribe(
            __import__("soundfile").read(w, dtype="float32")[0],
            language="ko", vad_filter=False)
        os.unlink(w)
        return " ".join(s.text.strip() for s in segs_)

    best = (-1.0, t0, "")
    for off in range(-20, 21, 2):
        t = max(0.0, t0 + off)
        txt = clip_text(t)
        sc = difflib.SequenceMatcher(None, norm(ref), norm(txt)).ratio()
        if sc > best[0]:
            best = (sc, t, txt)
    # 在最佳点附近再细搜 1s
    bt = best[1]
    for off in (-1.5, -1, -0.5, 0.5, 1, 1.5):
        t = max(0.0, bt + off)
        txt = clip_text(t)
        sc = difflib.SequenceMatcher(None, norm(ref), norm(txt)).ratio()
        if sc > best[0]:
            best = (sc, t, txt)
    sc, t, txt = best
    print("\n最佳窗口 t=%.2fs  与旧 ref_text 相似度 %.4f" % (t, sc))
    print("该窗口实际内容：\n  %s" % txt)
    if sc < 0.6:
        print("⚠️ 相似度偏低，需人工确认是否同一段")

    out = os.path.join(VOUT, "%s.wav" % a.name)
    if not a.apply:
        print("\n[DRY-RUN] 加 --apply 才写 %s" % out)
        return 0
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i",
                    os.path.join(d, "merged.wav"), "-ss", str(t),
                    "-t", str(target_dur), "-ar", "44100", "-ac", "1", out],
                   check=True)
    print("✅ 写出 %s（%.2fs）" % (out, dur(out)))
    print("建议 ref_text（用这段的 ASR 结果，别照抄旧的）：\n%s" % txt)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
