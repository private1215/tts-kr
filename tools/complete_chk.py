# -*- coding: utf-8 -*-
"""完整性回检：转录成品，看**原文有多少真的念出来了**。

为什么不能只看语速（时长）：
    案例原声 60 秒的文案，GS 女声只出了 33.6 秒。两种可能：
      (a) 语速真的快 1.79 倍 —— 听感是"赶"，但内容完整；
      (b) 中间吞字/截断     —— 内容缺失，直接不合格。
    只看时长分不出来，只有 ASR 转录 + 覆盖率能定性。

覆盖率怎么算（不用 CER 而是 LCS 覆盖率）：
    CER 会被"ASR 自己认错字"污染，把**转录噪声**算成**生成缺陷**。
    这里用 difflib 的匹配块总长 / 原文长度 —— 只统计"原文里有多少字符
    在转录里找到了对应"，ASR 多认的字不惩罚、少认的字才扣。
    判据：覆盖率 < 0.85 ⇒ 基本可判定吞字/截断。

用法：
    python tools/complete_chk.py <原文txt> <音频1> [音频2 ...]
"""
import difflib
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import asr_check as ASR                          # noqa: E402

LANG = os.environ.get("TTS_ASR_LANG", "ko")


def _to_wav(path):
    """一律重采样到 16k mono。

    ⚠️ 2026-10-12 修：原来 wav 直接返回原路径，而 GS 输出是 24k，
    脚本按 16k 算时长 → 33.6s 显示成 67.2s，误判成"内容超长"。
    覆盖率/CER 不受影响（transcribe 内部会重采样），但时长列一直错。
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", tmp.name], check=True)
    return tmp.name, tmp.name


def coverage(ref, hyp):
    """原文字符被转录覆盖的比例（0~1）。"""
    if not ref:
        return None
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return matched / float(len(ref))


def main():
    ref_path = sys.argv[1]
    ref = ASR.normalize(open(ref_path, encoding="utf-8").read())
    print("原文 %d 字（归一化后）" % len(ref))
    print("%-34s %7s %7s %7s %7s  %s" %
          ("样本", "覆盖率", "CER", "转录字", "冗余率", "时长s"))
    print("-" * 88)
    for p in sys.argv[2:]:
        if not os.path.exists(p):
            print("%-34s 缺失" % os.path.basename(p)[:34])
            continue
        w, tmp = _to_wav(p)
        try:
            hyp = ASR.transcribe(w, language=LANG)
            h = ASR.normalize(hyp)
            cov = coverage(ref, h)
            c = ASR.cer(ref, h)
            import soundfile as sf
            d = len(sf.read(w)[0]) / 16000.0
            # 冗余率：转录比原文长多少。> 0.3 基本可判定「重复念/跑飞」
            # （线上 yohan_male 曾测出转录 643 字 vs 原文 343 字 = 冗余 0.87）
            red = (len(h) - len(ref)) / float(len(ref)) if ref else None
            flag = ""
            if cov is not None and cov < 0.85:
                flag += " ⚠吞字"
            if red is not None and red > 0.30:
                flag += " ⚠重复"
            print("%-34s %7.3f %7.3f %7d %+7.2f  %.1f%s" %
                  (os.path.basename(p)[:34], cov or -1, c, len(h),
                   red or 0, d, flag))
        except Exception as ex:
            print("%-34s ERR %s" % (os.path.basename(p)[:34], repr(ex)[:70]))
        finally:
            if tmp:
                os.unlink(tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
