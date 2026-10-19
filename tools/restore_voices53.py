# -*- coding: utf-8 -*-
"""restore_voices53.py —— 恢复 2026-10-18 员工误删的音色（B 级数据变更）。

事故全貌（service.log 铁证，时间均为 UTC，IP 192.168.1.107）：
  09:01:21  删除音色 elara，清理文件: ['elara.wav', 'elara__full.wav']
  09:01:44  删除音色 SOLA_female，清理文件: ['SOLA_female.wav']
  09:01:56  删除音色 SOLA_gs10，清理文件: 无（wav 在 VOICE_DIR 之外，幸存）
  09:02:40  删除音色 yo han，清理文件: ['yo han.wav', 'yo han__full.wav']
另有 5 个音色的 engine 被逐个改成 cosyvoice3（已在前一步修回）。

可恢复性判定：
  · SOLA_gs10  ✅ 已恢复（wav 幸存于 /mnt/tts/projects/gpt-sovits/out/）
  · SOLA_female ✅ **可完整重建** —— 它本来就是从案例 9742074652c4 的
    merged.wav 切 [172.5, 187.5] 得到的（origin=cases / src_case / clip
    字段都在备份里），案例库没被删 → 按原参数重新切一遍即可，字节级等价。
  · elara / yo han ❌ **不可恢复** —— 它们的 src 是用户上传的原始录音
    （sources/*__full.wav），删除时连源带切片一起清掉了，全盘无副本。
    硬造一个参考音频会让员工选到「同名不同人」的嗓音（比缺失更糟），
    因此**不恢复**，留待用户重新上传原录音。

用法（先 dry-run 看判定，再 --apply 落盘）：
  python3 tools/restore_voices53.py [--apply]
"""
import json
import os
import shutil
import subprocess
import sys
import time

VOICES = "/mnt/tts/data/voices/voices.json"
BAK = VOICES + ".bak-before-omnimale53"
CASE = "/mnt/tts/data/cases/9742074652c4/merged.wav"

APPLY = "--apply" in sys.argv


def log(*a):
    print(*a, flush=True)


def main():
    v = json.load(open(VOICES, encoding="utf-8"))
    b = json.load(open(BAK, encoding="utf-8"))

    # ---- SOLA_female ----
    if "SOLA_female" in v:
        log("SKIP  SOLA_female 已存在")
    else:
        src = b["SOLA_female"]
        clip = src.get("clip")
        dur = src.get("dur")
        log("RESTORE SOLA_female: 源案例 %s  clip=%s dur=%s sr=%s engine=%s"
            % (src.get("src_case"), clip, dur, src.get("sr"),
               src.get("engine")))
        if not os.path.exists(CASE):
            log("  FAIL 案例文件不存在 %s —— 无法重建" % CASE)
        elif not clip:
            log("  FAIL 备份里没有 clip 字段，不敢猜切点")
        else:
            out = src["path"]
            t0, t1 = clip
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", CASE,
                   "-ss", str(t0), "-to", str(t1),
                   "-ar", str(src.get("sr", 24000)), "-ac", "1", out]
            log("  CMD " + " ".join(cmd))
            if APPLY:
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    log("  FAIL ffmpeg: %s" % (r.stderr or "")[-300:])
                else:
                    real = float(subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries",
                         "format=duration", "-of", "default=nw=1:nk=1", out],
                        capture_output=True, text=True).stdout.strip())
                    log("  OK   %s  %.2fs（期望 %.1fs）" % (out, real, dur))
                    v["SOLA_female"] = src
                    log("  配置条目已回填（engine=%s gender=%s）"
                        % (src.get("engine"), src.get("gender")))

    # ---- elara / yo han ----
    for name in ("elara", "yo han"):
        if name in v:
            log("SKIP  %s 已存在" % name)
            continue
        src = b.get(name)
        if not src:
            log("MISS  %s 不在备份里" % name)
            continue
        need = [src.get("path"), src.get("src_path")]
        alive = [p for p in need if p and os.path.exists(p)]
        log("NORESTORE %s：参考音频已随删除清掉（需要 %s，现存 %s）→ 需用户重新上传原录音"
            % (name, need, alive or "无"))

    if not APPLY:
        log("\n[DRY-RUN] 未落盘。确认无误后加 --apply")
        return

    # 备份 + 原子写
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bak = VOICES + ".bak-before-restore53-" + stamp
    shutil.copy2(VOICES, bak)
    log("备份 -> %s" % bak)
    tmp = VOICES + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(v, f, ensure_ascii=False, indent=2)
    os.replace(tmp, VOICES)
    log("已写入 %s（%d 个音色）" % (VOICES, len(v)))
    # 语法/完整性自检
    chk = json.load(open(VOICES, encoding="utf-8"))
    log("自检：读回 %d 个音色 %s" % (len(chk), sorted(chk)))


if __name__ == "__main__":
    main()
