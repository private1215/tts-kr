#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""declick 接入 v2：处理「快路径绕过」问题。

## 实测到的坑
`server.py:1600  _legacy_post = _engine_name not in NEW_ENGINES`
`NEW_ENGINES = ("voxcpm", "omnivoice")`
→ **OmniVoice 走 `_legacy_post=False`，命中 1988 行的快路径
   `if _rt_db is None:` → 一条命令写 mp3，完全绕过 roomtone/declick。**

第一次我把 declick 插在 1997 行（roomtone 之后）——**对 OmniVoice 完全无效**，
实测成品仍残留 2 处硬切（最陡 −91.0dB @52.44s，峰值 0.000570 → 0.000000）。

## 本脚本的做法（不碰快路径，改为给快路径也补一次）
在 `_encode_mp3` 内、快路径**之前**插入：
    1. 先把整条 af 链的结果落成临时 wav
    2. 对该 wav 做 declick（补极安静段的环境声）
    3. 用**已处理过的 wav** 作为 `-i` 重新走 libmp3lame

⚠️ 与旧分支的区别：
   - 旧分支读 `_norm_wav`（roomtone+peakfit 后的中间态）
   - 新插入读**同一条 af 链的输出**，所以**无论是否 legacy_post 都会经过**

### 为什么不能简单地"把 omnivoice 从 NEW_ENGINES 去掉"
那样会让 OmniVoice 也走 peakfit/TONE_EQ 等 legacy 处理，
而这些是**按 VoxCPM2 频谱拟合的**，对 OmniVoice 可能失配（另一个待查项）。
→ 保持 NEW_ENGINES 不动，只给快路径补 declick。
"""
import ast
import sys

P = "/mnt/tts/projects/qwen3-tts/app/server.py"

BAD = """                # 🔴 2026-10-15 第28轮：员工反馈「还是存在声音卡顿」。
                #    实测 EL 标杆硬切 0 处，我们 10 处（旧引擎 22 处），
                #    原因是我们输出的停顿电平低到 -85~-146dBFS（近乎绝对零），
                #    而真人停顿是有环境声的。补一层 -70dB 极轻噪声即可消除。
                #    必须在 roomtone 之后、libmp3lame 之前（陷阱 #18）。
                    if declick.ENABLED:
                    _nx = declick.apply(_nx, _nsr)"""

ANCHOR = ("                if _rt_db is None:\n"
          "                    # 没开底噪：一条命令走完，与 r21 及之前**完全一致**\n"
          "                    subprocess.run(c + _tail, check=True, timeout=600)\n"
          "                    return")

INSERT = '''                # 🔴 2026-10-15 第28轮 declick（去咔）：给极低电平段补环境声。
            #    必须放在这里（而不是 roomtone 之后）：OmniVoice/voxCPM 命中
            #    NEW_ENGINES → `_legacy_post=False` → 下面 `if _rt_db is None`
            #    的快路径会整条绕过 roomtone，插在它后面对它们**完全无效**
            #    （实测残留 2 处硬切，最陡 -91.0dB）。这里两条分支都会经过。
            #    只补极安静段（<−60dBFS），不改时长、不改可听内容。
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
    if BAD in src:
        src = src.replace(BAD + "\n", "", 1)
        print("撤掉第一次插错位置的调用")
    if "_dc_wav" in src:
        print("已接过（v2 位置），跳过")
        return 0
    if ANCHOR not in src:
        print("🔴 锚点没找到")
        return 1
    src = src.replace(ANCHOR, INSERT + ANCHOR, 1)
    ast.parse(src)
    open(P, "w", encoding="utf-8").write(src)
    print("✅ declick v2 已插入到所有路径必经处，语法通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())