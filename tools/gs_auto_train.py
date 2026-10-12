# -*- coding: utf-8 -*-
"""上传案例 → 自动触发 GPT-SoVITS 微调（补上「任意文案生成」链路最大缺口）。

🔴 缺口是什么
-------------
上传新案例后，系统只会做「拆解 → 生成克隆参考音频」，**不会训练微调权重**。
结果是新音色只能走零样本，而实测零样本天花板 0.80~0.88，
拿不到微调的 0.93+（2026-10-12 长文案对照坐实）。
也就是说：**案例库每多一个人，就多一个拿不到最佳链路的音色。**

🔴 硬约束（每条都踩过坑）
-------------------------
1. **测试区必须留出来**：hold 案例整条不进训练集，否则评测变成开卷考试。
   默认 hold = 该组按 id 排序的第一条，也可用 --hold 指定。
2. **显存是串行的**：3090 上 Cosy 常驻 ~2.2G，训练峰值约 10G。
   开训前必须检查空闲显存，不够就排队等，不要硬上（会连累员工的 Cosy 任务）。
3. **同一时刻只允许一个训练**（文件锁），两个训练并行必 OOM。
4. 训练完必须让 GS 服务 `/unload`，否则它内存里还挂着旧权重，
   新权重文件躺在那儿不生效（这个坑完全静默，员工只会觉得"没变化"）。

用法
----
    # 训某一组
    python tools/gs_auto_train.py --spk SOLA
    # 训全部有足够素材的组
    python tools/gs_auto_train.py --all
    # 只看会训什么，不真训
    python tools/gs_auto_train.py --all --dry-run

接入上传链路：app/server.py 的 /api/cases/extract-voices 末尾调用
`gs_auto_train.maybe_train(added_cids)`，由 GS_AUTO_TRAIN=1 开关控制（默认关，
避免未经评估就占用 3090）。
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GS = os.environ.get("GS_DIR", "/mnt/tts/projects/gpt-sovits")
PROJ = os.environ.get("TTS_PROJ", "/mnt/tts/projects/qwen3-tts")
VENV = os.path.join(PROJ, "venv", "bin", "python")
LOCK = os.path.join(GS, ".auto_train.lock")
LOG = os.path.join(GS, "auto_train.log")

GPU_UUID = "GPU-b67eee2c-c672-6915-910d-617c62abebee"   # 3090
MIN_FREE_GIB = 8.0        # 低于此值排队等（Cosy 常驻 2.2G，训练峰值 ~10G）
WAIT_TIMEOUT = 3600.0     # 等显存最多 1 小时，超时就放弃本次
MIN_CASES = 3             # 少于这么多样例不值得训
MAX_MINUTES = 160.0       # 训练音频上限（分钟），防训练过久


def log(msg):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def gpu_free_gib():
    """3090 空闲显存（GiB）。查不到返回 None（不阻塞，交给后续判断）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits", "-i", "1"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        # 用 UUID 更稳，-i 1 在驱动异常时可能指错卡；上面失败再试 UUID
        if not out:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits",
                 "--id", GPU_UUID],
                capture_output=True, text=True, timeout=20).stdout.strip()
        vals = [float(x) for x in out.splitlines() if x.strip()]
        return vals[0] / 1024.0 if vals else None
    except Exception as e:
        log("⚠️ 查显存失败：%r" % e)
        return None


def wait_vram():
    """等 3090 空闲显存达标。返回 True=可以开训。"""
    t0 = time.time()
    while True:
        free = gpu_free_gib()
        if free is None:
            log("查不到显存，按可以开训处理（风险自负）")
            return True
        if free >= MIN_FREE_GIB:
            log("3090 空闲 %.1f GiB（阈值 %.1f）→ 开训"
                % (free, MIN_FREE_GIB))
            return True
        if time.time() - t0 > WAIT_TIMEOUT:
            log("等显存超时（%.0f 分钟），放弃本次训练" % (WAIT_TIMEOUT / 60))
            return False
        log("3090 空闲仅 %.1f GiB < %.1f，%.0fs 后再看"
            % (free, MIN_FREE_GIB, 60))
        time.sleep(60)


def run(cmd, cwd=None, timeout=7200):
    log("RUN: %s" % " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True)
    tail = (p.stdout or "")[-2000:]
    for line in tail.splitlines():
        log("   | %s" % line)
    return p.returncode


def list_groups():
    """案例库里所有说话人组及案例数。"""
    sys.path.insert(0, HERE)
    import mkcase_voices as mk
    cases = mk.load_json(os.path.join(
        os.environ.get("TTS_BASE", "/mnt/tts"), "data", "cases.json"), {})
    grp, _meta = mk.embed_groups(cases, 0.75)
    out = []
    for (nm, gd), cids in grp.items():
        out.append((nm, len(cids)))
    return sorted(out, key=lambda x: -x[1])


def train_one(spk, max_minutes=MAX_MINUTES, hold="", dry=False):
    """跑完 prep → preprocess → train → unload。返回 True=成功。"""
    log("=" * 60)
    log("开始训练说话人组：%s" % spk)
    rc = run([VENV, os.path.join(HERE, "gs_prep_group.py"),
              "--spk", spk, "--max-minutes", str(max_minutes)]
             + (["--hold", hold] if hold else []), cwd=PROJ)
    if rc != 0:
        log("❌ 数据准备失败 %s（rc=%d）" % (spk, rc))
        return False
    for step in ("preprocess", "train"):
        rc = run(["bash", os.path.join(ROOT, "deploy", "gs_run_group.sh"),
                  spk, step], cwd=PROJ, timeout=14400)
        if rc != 0:
            log("❌ %s 失败 %s（rc=%d）" % (step, spk, rc))
            return False
    # 🔴 必须让 GS 服务丢掉内存里的旧权重，否则新权重不生效（静默失败）
    try:
        import urllib.request
        urllib.request.urlopen(
            urllib.request.Request("http://127.0.0.1:8200/unload",
                                   data=b"", method="POST"), timeout=30)
        log("✅ 已通知 GS 服务 unload，下次请求将加载新权重")
    except Exception as e:
        log("⚠️ GS 服务 unload 失败（不影响训练产物）：%r" % e)
    log("✅ 训练完成：%s" % spk)
    return True


def maybe_train(added_cids=None, dry=False):
    """上传链路调用入口：素材够就训，不够就跳过。"""
    if os.environ.get("GS_AUTO_TRAIN", "0") != "1":
        log("GS_AUTO_TRAIN 未开启，跳过自动训练")
        return []
    if os.path.exists(LOCK):
        log("已有训练在跑（锁文件存在），跳过")
        return []
    open(LOCK, "w").close()
    try:
        groups = list_groups()
        done = []
        for nm, n in groups:
            if n < MIN_CASES:
                log("组 %s 只有 %d 条（< %d），跳过"
                    % (nm, n, MIN_CASES))
                continue
            if dry:
                log("[dry-run] 会训练 %s（%d 条案例）" % (nm, n))
                continue
            if not wait_vram():
                break
            if train_one(nm):
                done.append(nm)
        return done
    finally:
        try:
            os.unlink(LOCK)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spk", default="", help="说话人组名，如 SOLA")
    ap.add_argument("--all", action="store_true", help="训所有素材够的组")
    ap.add_argument("--max-minutes", type=float, default=MAX_MINUTES)
    ap.add_argument("--hold", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.dry_run:
        for nm, n in list_groups():
            flag = "→ 会训" if n >= MIN_CASES else "→ 跳过(素材不足)"
            print("  %-10s %3d 条  %s" % (nm, n, flag))
        return 0
    if a.spk:
        return 0 if train_one(a.spk, a.max_minutes, a.hold) else 1
    if a.all:
        return 0 if maybe_train() is not None else 1
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
