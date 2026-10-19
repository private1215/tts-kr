# -*- coding: utf-8 -*-
"""smoke_clickfix53b.py —— L1 冒烟：确认 clickfix53b 在生产链上真的跑了。

只验「改动是否生效」，不验质量（质量要用 L3 全篇）。
用法: cd /mnt/tts/projects/qwen3-tts && venv/bin/python -u /mnt/tts/diag/smoke_clickfix53b.py
"""
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from tools.eval5 import login, submit_job, wait_job  # noqa: E402

TXT = ("안녕하세요. 제주반도체 주주 여러분, 반갑습니다. 정민철 프로입니다. "
       "오늘 장중 흐름을 보면 바닥권 탈출 이후 매수세가 붙으면서 고점을 "
       "찔러본 뒤에도 시가 위를 꾹꾹하게 방어하고 있습니다. 현재 주가도 "
       "시가 위에서 잘 버텨주고 있죠. 저점을 깨지 않고 우상향 흐름을 "
       "유지한다는 건 긍정적이지만, 전고점을 건드리는 것과 그 악성 매물대를 "
       "뚫어내고 안착하는 건 완전히 다른 문제입니다.")

login()
print("chars:", len(TXT), flush=True)
j = submit_job(TXT, "EL_male", "smoke-cf53b")
st = wait_job(j.get("id"), timeout=1800)
print("status:", st.get("status"), "| engine=", st.get("engine"))
print("stats:", st.get("stats"))
