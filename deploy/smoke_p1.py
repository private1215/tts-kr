#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1 上线冒烟：男声(EL_male→voxcpm) 与 女声(EL_female→omnivoice) 各跑一单。

判活标准（不是 /health，那路由不存在）：
  1. 任务能进队列
  2. 状态走到 done
  3. 成品 mp3 存在且时长合理（>3s）
  4. 日志里出现 fix_chain 完成 + 引擎=新引擎
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"
TOK = os.environ.get("TTS_DEBUG_TOKEN", "smoke-8f3a2c91")

TEXT_M = ("안녕하세요, 한화솔루션 주주 여러분. 반갑습니다. "
          "오늘 종가는 이만삼천오백원, 전일 대비 천원 올랐습니다. "
          "거래량은 삼십오만사천주 정도 터졌고요. "
          "외인 수급을 보시면, 이천팔백주 순매도 포지션입니다. "
          "지금 구간에서 추격매수는 금물이고, 이만칠천오백원 매물대를 "
          "거래량 실어서 뚫어주는지 먼저 확인하셔야 됩니다.")
TEXT_F = ("안녕하세요, 리노공업 주주 여러분. 반갑습니다. "
          "오늘 시장 변동성이 크게 나왔습니다. "
          "단기적으로는 팔천육백팔십원 저항선을 뚫어내는지, "
          "눌릴 때 칠천삼백원을 지켜주는지 이 두 가지만 보시면 돼요. "
          "장 초반에 반짝 튄다고 덜컥 따라붙지 마시고요.")


def post(path, data):
    body = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Authorization": "Bearer " + TOK})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def get(path):
    req = urllib.request.Request(
        BASE + path, headers={"Authorization": "Bearer " + TOK})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def run(voice, text, tag):
    print("\n===== %s (voice=%s) %d 字 =====" % (tag, voice, len(text)))
    try:
        r = post("/api/jobs", {"voice": voice, "text": text,
                               "mode": "clone", "language": "Korean",
                               "title": "P1-smoke-" + tag,
                               "emotion": "standard"})
    except Exception as e:
        print("提交失败:", repr(e)[:300])
        return None
    jid = r.get("id") or r.get("job", {}).get("id")
    print("job id =", jid, "| resp keys:", list(r)[:8])
    if not jid:
        print(r)
        return None
    t0 = time.time()
    last = ""
    while time.time() - t0 < 900:
        try:
            j = get("/api/jobs/" + jid)
        except Exception as e:
            print("轮询失败:", repr(e)[:200])
            break
        st = j.get("status")
        if st != last:
            print("  %.0fs status=%s stage=%s progress=%s"
                  % (time.time() - t0, st, j.get("stage"),
                     j.get("progress")))
            last = st
        if st in ("done", "error", "failed"):
            print("  最终:", json.dumps({k: j.get(k) for k in
                                        ("status", "stats", "error", "files")},
                                       ensure_ascii=False)[:600])
            return j
        time.sleep(4)
    print("  超时")
    return None


if __name__ == "__main__":
    out = {}
    for voice, text, tag in (("EL_male", TEXT_M, "male-voxcpm"),
                             ("EL_female", TEXT_F, "female-omnivoice")):
        out[tag] = run(voice, text, tag)
    print("\n===== 汇总 =====")
    for tag, j in out.items():
        if not j:
            print(tag, "FAILED")
            continue
        st = j.get("stats") or {}
        print("%-18s status=%s engine=%s 时长=%.1fs cps=%.2f files=%s"
              % (tag, j.get("status"),
                 (st.get("plan") or {}).get("engine"),
                 st.get("audio_seconds") or 0, st.get("cps") or 0,
                 len(j.get("files") or [])))
