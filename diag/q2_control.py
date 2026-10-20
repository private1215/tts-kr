# -*- coding: utf-8 -*-
"""q2_control.py —— Q2 对照实验：把争议数字单独合成，排除「TTS 念错」vs「ASR 听错」。

原 job bf17cf796a40 未带 case_id，这里同样不带；voice=EL_female / mode=clone /
language=Korean，尽量复刻原生产条件。文本只含争议数字，每条独立成句。
"""
import json
import os
import sys
import time

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
os.environ.setdefault("TTS_BASE", "http://127.0.0.1:8000")
from tools.eval5 import submit_job, wait_job, _get, login  # noqa: E402

TEXT = (
    "로보티즈 장중 고점이 삼십일만육천원 찍고 살짝 눌려있는 상태입니다. "
    "거래량은 벌써 사십사만주 훌쩍 넘겼고요. "
    "오늘 시가가 삼십만오백원이고 장중 고점이 삼십일만육천원입니다. "
    "전일 대비 한 사쩜사퍼센트 정도 띄워주고 있습니다. "
    "직전 고점 뚫으려면 한 육천오백원 정도 남았죠. "
    "지난 유월에 사십구만이천원에서 쌍봉 맞고 줄줄 흘러서, "
    "칠월 말에 십오만천구백원까지 진짜 반토막 넘게 박살이 났었죠. "
    "어제 기준 로보티즈 삼십만구천오백원이고 거래대금은 이십팔만사천입니다."
)

TITLE = "numctrl-1019"


def main():
    login()
    r = submit_job(TEXT, "EL_female", TITLE)
    print("submit ->", r)
    jid = r.get("id") or r.get("job_id") or (r.get("job") or {}).get("id")
    if not jid:
        print("拿不到 id，响应：", json.dumps(r, ensure_ascii=False)[:500])
        return
    wait_job(jid, timeout=1800)
    st = json.loads(_get("/api/jobs/" + jid).decode("utf-8"))
    print("status =", st.get("status"), "files =", st.get("files"))
    mp3 = None
    for f in st.get("files") or []:
        mp3 = f.get("path")
    if not mp3:
        print("没拿到产物路径")
        return
    src = os.path.join("/mnt/tts/output", mp3)
    dst = "/mnt/tts/diag/v0902/numctrl.mp3"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    import shutil
    shutil.copy(src, dst)
    print("产物 ->", dst)

    # 16k wav
    wav = "/mnt/tts/diag/v0902/numctrl.16k.wav"
    import subprocess
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
    segs, _ = m.transcribe(x, language="ko", vad_filter=False,
                           word_timestamps=True)
    words = []
    for s in segs:
        for w in (s.words or []):
            words.append([round(w.start, 2), round(w.end, 2), w.word])
    json.dump(words, open("/mnt/tts/diag/v0902/numctrl_words.json", "w"),
              ensure_ascii=False)
    print("\n=== 对照音频 ASR（词级） ===")
    for st_, en, tx in words:
        print("  %7.2f-%7.2f  %s" % (st_, en, tx))
    print("\n=== 对照音频 ASR 全文 ===")
    print("".join(w[2] for w in words))


if __name__ == "__main__":
    main()
