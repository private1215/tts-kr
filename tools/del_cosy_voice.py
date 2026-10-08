# -*- coding: utf-8 -*-
"""删掉 `add_cosy_voice.py` 加的临时音色，让平台恢复原状（员工零感知）。

用法（服务器上，需要 sudo）：
    echo <pw> | sudo -S -p '' /mnt/tts/projects/qwen3-tts/venv/bin/python3 \
        tools/del_cosy_voice.py
"""
import json

P = "/mnt/tts/data/voices/voices.json"
KEY = "EL_male__COSY"


def main():
    d = json.load(open(P, encoding="utf-8"))
    if KEY not in d:
        print("不存在，无需删除:", KEY)
    else:
        del d[KEY]
        json.dump(d, open(P, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("已删除", KEY)
    print("当前音色:", list(d))


if __name__ == "__main__":
    main()
