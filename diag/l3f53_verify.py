# -*- coding: utf-8 -*-
"""l3f53_verify.py —— 女声接 EQ+clickfix 后的 **L3 全篇线上验收**。

## 为什么必须 L3
项目铁律：**片段口径与全篇口径结论可以相反**（第47轮栽两次）→ 只用 L3 下结论。
L1 只有 32.8s，中频 1.11×EL 不可信（短片段统计噪声大）。

## 判据（对合规标杆 cases/00b6a45ad6aa/merged.wav）
  1. 中频要从 0.88 抬到 **0.97~1.05**（修到位，且不过冲）；
  2. click 不能比修前差（修前 1.24×EL；预期修后 <<1）；
  3. 音色不能掉（修前 0.9204；clickfix 在男声上实测零代价）；
  4. 质心/杂音/cps/停顿 不退步。

用法: venv/bin/python -u /mnt/tts/diag/l3f53_verify.py
"""
import glob
import os
import subprocess
import sys
import time

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from tools.eval5 import login, submit_job, wait_job  # noqa: E402

PY = "/mnt/tts/projects/qwen3-tts/venv/bin/python"
T = "/mnt/tts/projects/qwen3-tts/tools"
EL = "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"
TITLE = "l3f53b"
CHARS = 3415


def find_out(title):
    hits = sorted(glob.glob("/mnt/tts/output/" + title + "*"),
                  key=os.path.getmtime)
    return hits[-1] if hits else None


def main():
    login()
    txt = open("/mnt/tts/diag/female_full.txt", encoding="utf-8").read().strip()
    print("文案 %d 字" % len(txt), flush=True)
    j = submit_job(txt, "EL_female", TITLE)
    print("job=%s" % j.get("id"), flush=True)
    s = wait_job(j.get("id"), timeout=7200)
    st = s.get("stats") or {}
    print("status=%s engine=%s cps=%s dur=%s"
          % (s.get("status"), s.get("engine"), st.get("cps"),
             st.get("audio_seconds")), flush=True)
    d = find_out(TITLE)
    if not d:
        print("❌ 找不到产物 %s" % TITLE)
        return 1
    fs = sorted(glob.glob(os.path.join(d, "*.mp3")))
    if not fs:
        print("❌ 没有 mp3")
        return 1
    mp3 = fs[-1]
    print("产物：%s" % mp3, flush=True)
    print("\n===== 客观指标 vs 合规 EL 标杆 =====", flush=True)
    subprocess.run([PY, "-u", T + "/measure53.py", mp3, EL, str(CHARS)])
    print("\n===== 音色 vs 合规 EL 标杆 =====", flush=True)
    subprocess.run(["/mnt/tts/projects/cosy/venv/bin/python", "-u",
                    T + "/spkcmp53.py", mp3, EL])
    print("\n对照（修前 l3f53-1.mp3）：click 1.24× / 中频 0.88 / 质心 0.97 / "
          "杂音 1.27 / 音色 0.9204", flush=True)
    print("成品：%s" % mp3, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
