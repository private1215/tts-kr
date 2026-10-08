# -*- coding: utf-8 -*-
"""临时给 voices.json 加一个 engine=cosyvoice3 的音色，供 Cosy A/B 使用。

为什么是"临时"
--------------
引擎由音色的 `engine` 字段决定，而 /api/jobs **不接受 engine 参数**。
要跑 CosyVoice3 只有两条路：改 emotion（会带情感指令，不公平）或加一个
engine=cosyvoice3 的音色。我们选后者，且**不改 EL_male 本身**——员工正在
用平台，改了就直接影响线上。

跑完 A/B 后用 `tools/del_cosy_voice.py` 删掉，恢复原状。

用法（服务器上，需要 sudo 因为 voices.json 是 root 属主）：
    echo <pw> | sudo -S -p '' /mnt/tts/projects/qwen3-tts/venv/bin/python3 \
        tools/add_cosy_voice.py
"""
import json
import shutil
import os

P = "/mnt/tts/data/voices/voices.json"
BAK = P + ".bak-cosyab"
KEY = "EL_male__COSY"


def main():
    if not os.path.exists(BAK):
        shutil.copy2(P, BAK)
        print("已备份 ->", BAK)
    d = json.load(open(P, encoding="utf-8"))
    if KEY in d:
        print("已存在，跳过:", KEY)
        return
    src = d["EL_male"]
    d[KEY] = dict(src, engine="cosyvoice3")
    json.dump(d, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("已添加", KEY, "-> engine =", d[KEY]["engine"])
    print("当前音色:", list(d))


if __name__ == "__main__":
    main()
