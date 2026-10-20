# -*- coding: utf-8 -*-
"""q4_control2.py —— 对照实验 2：ASR 增删到底是「TTS 多念字」还是「ASR 听错」。

q4_inserts.py 用能量判据太粗（把 ASR 听错也标成「真多念」）。
本脚本用**对照实验定罪**：把长音频里出问题的原句单独合成，再 ASR。
  干净单句也出现同样的 insert → ASR 伪影（与长音频/上下文无关）；
  干净单句正常 → 长音频里那个 insert 可能是真多念，需进一步查。

覆盖员工反馈点附近的句子：
  5s/14s（급하게 카메라 켰습니다 / 왜 이 자리가 / 시월 이일…）
  58s（뚫어내야 됩니다. 거래대금은…）← ASR 插入原文没有的「그래서」
  183s（깔아주십시오. 그리고, 제 찐…）← ASR 插入「이따」
"""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
os.environ.setdefault("TTS_BASE", "http://127.0.0.1:8000")
from tools.eval5 import submit_job, wait_job, _get, login  # noqa: E402

TEXT = (
    # 5s / 14s 区域（员工报「磁音」「音一个字没了」）
    "오늘 우리 주주님들께, 진짜 피가 되고 살이 되는 중요한 말씀 드리려고, "
    "급하게 카메라 켰습니다. 와, 지금 로보티즈, 자리 기가 막히거든요. "
    "왜 이 자리가 중요하냐고요? 시월 이일, 오전 열한 시 삼십사분 기준, "
    "로보티즈 삼십만구천오백원. "
    # 58s 区域（员工报「额」；ASR 插入原文没有的「그래서」）
    "무조건 삼십일만육천원 고까지, 다시 뚫어내야 됩니다. "
    "거래대금은 이미 어제 수준 다 잡아먹었죠? "
    # 183s 区域（ASR 把「제 찐」听成「이따 재찐」）
    "댓글에 '독수리', 쫙 한 번 깔아주십시오. 그리고, 제 찐 구독자님들은 아시죠?"
)

TITLE = "insctrl-1019"


def main():
    login()
    r = submit_job(TEXT, "EL_female", TITLE)
    jid = r.get("id") or r.get("job_id") or (r.get("job") or {}).get("id")
    if not jid:
        print("提交失败:", json.dumps(r, ensure_ascii=False)[:300])
        return
    wait_job(jid, timeout=1800)
    st = json.loads(_get("/api/jobs/" + jid).decode("utf-8"))
    mp3 = None
    for f in st.get("files") or []:
        mp3 = f.get("path")
    src = os.path.join("/mnt/tts/output", mp3)
    dst = "/mnt/tts/diag/v0902/insctrl.mp3"
    shutil.copy(src, dst)
    wav = "/mnt/tts/diag/v0902/insctrl.16k.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", dst, "-ar", "16000",
                    "-ac", "1", wav], check=True)
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    x, sr = sf.read(wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    m = WhisperModel("Systran/faster-whisper-small", device="cpu",
                     compute_type="int8", local_files_only=True)
    segs, _ = m.transcribe(x, language="ko", vad_filter=False, word_timestamps=True)
    words = []
    for s in segs:
        for w in (s.words or []):
            words.append([round(w.start, 2), round(w.end, 2), w.word])
    json.dump(words, open("/mnt/tts/diag/v0902/insctrl_words.json", "w"),
              ensure_ascii=False)
    print("\n=== 对照音频 ASR 全文 ===")
    print("".join(w[2] for w in words))
    print("\n=== 原文 ===")
    print(TEXT)
    print("\n=== 词级 ===")
    for st_, en, tx in words:
        print("  %7.2f-%7.2f  %s" % (st_, en, tx))


if __name__ == "__main__":
    main()
