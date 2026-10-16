#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gs_voice_f8.py —— 注册**语速更快**的男声 GS 参考音色，用于把 GS 输出
从 cps 6.92 提到接近 EL 的 8.23。

## 背景
男声 GS 质量全面优于 voxcpm（静音 4-8k 11.423≈EL 10.968、click 0.517% vs 1.175%），
但**语速慢 16%**（cps 6.92 vs EL 8.23）。

## 依据
**GS 输出语速 ≈ 参考音频隐含语速（ref_text 字数 ÷ 音频时长）**
  现役 yohan_gs10 = 7.55s / 51 字 = 6.75 字/s -> 输出 6.92 ✅
新参考 cand2 = 6.20s / 48 字(原稿口径) = **7.74 字/s** -> 预期输出 ~7.9

## 🔴 ref_text 必须用**原稿**，不能直接用 ASR 输出
ASR 把「켜둬도」误识成「켜도도」、把「주린이」误识成「주리니」，
字数也多算了 4 个（52 vs 48）→ 隐含语速被高估成 8.39（实际 7.74）。
ref_text 写错会让 GS 学错发音，也会让语速规律算偏。

用法：
  python3 gs_voice_f8.py add      # 注册 yohan_gs_f8 并 reload
  python3 gs_voice_f8.py del      # 删除
"""
import json
import os
import shutil
import sys

import requests

BASE = "http://127.0.0.1:8000"
VJ = "/mnt/tts/data/voices/voices.json"
BAK = VJ + ".bak-before-f8"

# 🔴 原稿核对过的文本（cases/6cdd25bc75f2/full.txt 第 13~14 行）
REF_TEXT = ("이 채널 하나만 알람 켜둬도 계좌 색깔이 바뀐다고들 하세요. "
            "주린이분들도 바로 따라 하실 수 있게,")
SRC = "/mnt/tts/diag/refspd48/cand2.wav"        # 6.20s / 48字 / 7.74 字/s
DST = "/mnt/tts/data/voices/yohan_gs_f8.wav"
KEY = "yohan_gs_f8"


def login():
    s = requests.Session()
    for pw in ("admin", "smoke-8f3a2c91", "123456"):
        r = s.post(f"{BASE}/api/login", json={"username": "admin", "password": pw})
        if r.status_code == 200:
            return s
    s.headers.update({"Authorization": "Bearer smoke-8f3a2c91"})
    return s


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "add"
    v = json.load(open(VJ, encoding="utf-8"))

    if cmd == "del":
        v.pop(KEY, None)
        json.dump(v, open(VJ, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("已删除", KEY)
    elif cmd == "add":
        if not os.path.exists(SRC):
            print("缺素材：%s" % SRC)
            return 1
        if not os.path.exists(BAK):
            shutil.copy(VJ, BAK)
            print("备份 ->", BAK)
        shutil.copy(SRC, DST)
        import soundfile as sf
        dur = sf.info(DST).duration
        if not (3.0 <= dur <= 10.0):
            print("❌ 时长 %.2fs 超出 GS 的 3~10s 限制" % dur)
            return 1
        v[KEY] = {
            "path": DST,
            "ref_text": REF_TEXT,
            "sr": 24000,
            "dur": round(dur, 2),
            "engine": "gptsovits",
            "gender": "male",
            "bench_case": "6cdd25bc75f2",
            "note": ("第47轮 v1.2 实验：语速更快的 GS 参考。"
                     "%.2fs / %d字 / %.2f字/s（原稿口径），音色×EL_male 0.9184。"
                     "目标把 GS 输出 cps 从 6.92 提到 ~7.9（EL 8.23）。"
                     % (dur, len(REF_TEXT), len(REF_TEXT) / dur)),
            "cps_lock": None,
            "cps": None,
        }
        json.dump(v, open(VJ, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("已注册 %s：%.2fs / %d字 / %.2f字/s" %
              (KEY, dur, len(REF_TEXT), len(REF_TEXT) / dur))
    else:
        print("用法：add | del")
        return 1

    s = login()
    r = s.post(f"{BASE}/api/voices/reload")
    print("/api/voices/reload ->", r.status_code, r.text[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
