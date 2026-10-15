#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""declick 接入复原（2026-10-15 第30轮，用于修复 rsync 误覆盖）

## 事故
本地仓库是**旧快照**，我误跑了一次整项目 rsync，
把 `app/server.py` 覆盖成了不含 declick 的旧版（declick 集成丢失）。

## 本脚本
1. 补 `import declick`（原来在 `import roomtone` 之后）
2. 调 `apply_declick2.py` 的逻辑，把处理插到**所有路径必经处**

幂等：已存在就跳过。
"""
import ast
import os
import sys

P = "/mnt/tts/projects/qwen3-tts/app/server.py"

ANCHOR = ("                if _rt_db is None:\n"
          "                    # 没开底噪：一条命令走完，与 r21 及之前**完全一致**\n"
          "                    subprocess.run(c + _tail, check=True, timeout=600)\n"
          "                    return")

INSERT = '''                # 🔴 2026-10-15 第28轮 declick（去咔）：给极低电平段补环境声。
                #    必须放在这里（而不是 roomtone 之后）：OmniVoice/voxCPM 命中
                #    NEW_ENGINES → `_legacy_post=False` → 下面 `if _rt_db is None`
                #    的快路径会整条绕过 roomtone，插在它后面对它们**完全无效**
                #    （实测残留 2 处硬切，最陡 -91.0dB）。这里两条分支都会经过。
                #    只补真正的数字零段（<-100dBFS），不改时长、不改可听内容。
                if declick.ENABLED:
                    _dc_wav = os.path.join(folder_path, f".dc_{jid}_{ci}.wav")
                    subprocess.run(c + ["-ac", "1", "-ar", str(OUT_SR),
                                        "-c:a", "pcm_f32le", _dc_wav],
                                   check=True, timeout=600)
                    _dx, _dsr = sf.read(_dc_wav, dtype="float32")
                    if getattr(_dx, "ndim", 1) > 1:
                        _dx = _dx.mean(axis=1)
                    _dx = declick.apply(_dx, _dsr)
                    sf.write(_norm_wav, _dx, _dsr, subtype="FLOAT")
                    try:
                        os.unlink(_dc_wav)
                    except OSError:
                        pass
                    # 之后统一从 _norm_wav 重走 libmp3lame
                    c = ["ffmpeg", "-y", "-loglevel", "error", "-i", _norm_wav]
                    subprocess.run(c + _tail, check=True, timeout=600)
                    return

'''


def main():
    src = open(P, encoding="utf-8").read()

    # 1) import
    if "import declick" not in src:
        if "import roomtone" in src:
            src = src.replace(
                "import roomtone",
                "import roomtone\n"
                "import declick            # 去咔：给数字零的停顿段补环境声（r28）",
                1)
            print("✅ 补 import declick")
        else:
            print("🔴 找不到 import roomtone 锚点")
            return 1
    else:
        print("· import declick 已存在")

    # 2) 插入
    if "_dc_wav" in src:
        print("· declick 调用已存在，跳过")
    elif ANCHOR not in src:
        print("🔴 锚点没找到")
        return 1
    else:
        src = src.replace(ANCHOR, INSERT + ANCHOR, 1)
        print("✅ declick 调用已插入到所有路径必经处")

    ast.parse(src)
    open(P, "w", encoding="utf-8").write(src)
    print("✅ 语法通过，已写回 " + P)
    return 0


if __name__ == "__main__":
    sys.exit(main())
