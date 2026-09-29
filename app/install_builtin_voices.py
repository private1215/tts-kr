# -*- coding: utf-8 -*-
"""安装内置音色：把 make_builtin_voices.py 产出的音频与元数据写进 voices.json。

必须 sudo 运行（VOICE_DIR 属 root:root）：
    sudo ./venv/bin/python app/install_builtin_voices.py --src /tmp/builtin

拆成两步的原因：制作阶段要跑 Whisper、耗时长，用普通用户跑即可；
只有最后写 voices.json 和拷音频才需要 root。避免整段流程都以 root 运行
（root 下 HF 缓存路径不同，会重新去找模型）。
"""
import os
import sys
import json
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.environ.get("TTS_BASE", "").strip() or "/mnt/tts"
VOICE_DIR = os.path.join(BASE, "data", "voices")
VOICES_JSON = os.path.join(VOICE_DIR, "voices.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/builtin")
    ap.add_argument("--drop", default="", help="要一并删除的音色名（逗号分隔）")
    args = ap.parse_args()

    plan_path = os.path.join(args.src, "install_plan.json")
    if not os.path.exists(plan_path):
        print(f"找不到 {plan_path}，先跑 make_builtin_voices.py")
        return 1
    plan = json.load(open(plan_path, encoding="utf-8"))

    data = {}
    if os.path.exists(VOICES_JSON):
        try:
            data = json.load(open(VOICES_JSON, encoding="utf-8"))
        except Exception as e:
            print("读取现有 voices.json 失败，按空处理:", e)
    if isinstance(data, list):
        data = {}

    for name in [n for n in args.drop.split(",") if n.strip()]:
        name = name.strip()
        if name in data:
            v = data.pop(name)
            for p in (v.get("path"), v.get("src_path")):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                        print(f"  已删除 {p}")
                    except OSError as e:
                        print(f"  删除失败 {p}: {e}")
            print(f"  已移除音色「{name}」")

    os.makedirs(VOICE_DIR, exist_ok=True)
    for vid, rec in plan.items():
        local = rec.pop("_local_wav", None)
        dst = rec.get("path") or os.path.join(VOICE_DIR, f"{vid}.wav")
        if local and os.path.exists(local):
            shutil.copyfile(local, dst)
            os.chmod(dst, 0o644)
            print(f"  {vid}: {dst} ({rec.get('dur')}s)")
        elif not os.path.exists(dst):
            print(f"  ! {vid} 缺少音频文件，跳过")
            continue
        rec["path"] = dst
        data[vid] = rec

    tmp = VOICES_JSON + ".tmp"
    json.dump(data, open(tmp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    os.replace(tmp, VOICES_JSON)
    os.chmod(VOICES_JSON, 0o644)
    print(f"\n✓ 已写入 {VOICES_JSON}")
    for k, v in data.items():
        print(f"   {k}: {v.get('dur')}s  builtin={v.get('builtin')}  "
              f"ref_text {len(v.get('ref_text') or '')} 字")
    return 0


if __name__ == "__main__":
    sys.exit(main())
