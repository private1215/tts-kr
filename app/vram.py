"""双引擎显存串行 —— 12G 卡上「两个引擎共存」的唯一方式。

Qwen3-TTS 常驻 ~4.2 GiB、峰值实测 **10.52 GiB**（开保真增强时，旧基线 6.97 已过时）；
CosyVoice3 加载需 ~3.2 GiB。两者**不能同时驻留**，否则加载阶段直接 CUDA OOM：

    _pickle.UnpicklingError: Weights only load failed ... WeightsUnpickler error:
    CUDA out of memory. Tried to allocate 20.00 MiB. GPU

所以「两个引擎都要能用」的正确姿势是**串行共存**：跑某个引擎之前，把另一个从显存
彻底赶出去，并且**等显存真的释放**再往下走。生成路径与选型路径共用这套逻辑，
避免两处各写一份、其中一份悄悄失效（2026-09-29 就是这个原因导致的 P0：
选型路径有串行、生成路径没有 → 情感任务 100% 失败）。

⚠️ 关键坑（改这里必看）
    `del model + empty_cache()` **不够**。跨进程场景（CosyVoice3 走 HTTP /unload）
    返回后，nvidia-smi 仍显示该进程持有 5~7.6 GiB —— torch 缓存引用环没回收。
    必须 gc.collect() → synchronize() → empty_cache() → gc.collect() → empty_cache()
    （已在 cosy_server._release_model 实现），并以 torch.cuda.mem_get_info() 的
    **整卡空闲量**做轮询验证。
"""
import logging
import os
import time

log = logging.getLogger("tts")

# 各引擎加载所需的整卡空闲显存（GiB）= 实测加载量 + 余量
MIN_FREE_FOR_QWEN = float(os.environ.get("TTS_MIN_FREE_QWEN", "6.0"))
MIN_FREE_FOR_COSY = float(os.environ.get("TTS_MIN_FREE_COSY", "4.5"))
WAIT_TIMEOUT = float(os.environ.get("TTS_VRAM_WAIT_TIMEOUT", "90"))


def wait_free_vram(min_free_gib, timeout=WAIT_TIMEOUT):
    """轮询等整卡空闲显存达标，返回达标时的 GiB（拿不到信息返回 None）。

    超时**只告警不阻断**：后面还有懒加载兜底，不能让任务卡死在等显存上。
    实测选型场景里两次超时（2.9 / 1.9 GiB）后仍然跑成功了 —— 轮询是兜底不是硬门槛。
    """
    try:
        import torch
    except Exception:
        return None
    t0 = time.time()
    free = 0.0
    try:
        while True:
            free_b, _total = torch.cuda.mem_get_info()
            free = free_b / 1024 ** 3
            if free >= min_free_gib:
                return free
            if time.time() - t0 >= timeout:
                break
            time.sleep(2.0)
        log.warning("等待空闲显存 >= %.1f GiB 超时（%.0fs，当前 %.1f GiB），仍继续",
                    min_free_gib, timeout, free)
        return free
    except Exception as e:
        log.warning("查询显存失败（跳过等待）: %s", repr(e)[:120])
        return None


def release_cosy():
    """卸掉 CosyVoice3（独立进程，走 HTTP /unload；服务端已做同步释放）。"""
    try:
        import cosy_client
        return cosy_client.unload()
    except Exception as e:
        log.warning("卸载 CosyVoice3 失败: %s", repr(e)[:120])
        return None


def release_qwen():
    """卸掉 Qwen（同进程，直接 unload；卸载后下次生成经 _ensure_variant 自恢复）。"""
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import server
        eng = getattr(server, "engine", None)
        if eng is not None:
            eng.unload()
    except Exception as e:
        log.warning("卸载 Qwen 失败: %s", repr(e)[:120])


COSY_UNIT = os.environ.get("TTS_COSY_UNIT", "/etc/systemd/system/cosy-tts.service")


def _cosy_gpu_isolated():
    """CosyVoice3 是不是跑在**另一张卡**上（独立进程）。

    🔴 这段串行逻辑的前提是「两个引擎抢同一张 12G 卡」，那个前提**在 Cosy 被迁到
    3090 之后就不成立了**（2026-10-12 实测：nvidia-smi 显示 cosy_server 进程占
    GPU-b67eee2c（3090）5.3 GiB，主服务占 GPU-94853612（3060）4.2 GiB，两卡隔离）。
    前提不成立却还在跑，后果是**每次情感任务都白付一次 Qwen 卸载 + 重载**：
    员工点「激昂」→ 卸 Qwen → 等 3060 空闲（最多 90s）→ Cosy 在 3090 上生成（根本
    不用 3060）→ 收尾再重载 Qwen。纯浪费，还让情感任务明显变慢。

    判断方式（保守优先，拿不到信息就按「不隔离」处理，保持原有安全行为）：
      1. 环境变量 TTS_COSY_GPU_ISOLATED=1/0 显式指定，最优先
      2. 否则读 cosy 单元的 CUDA_VISIBLE_DEVICES，和本进程自己的比；不同 -> 隔离
      3. 读不到 -> False（继续串行，宁可慢也不要 OOM）
    """
    forced = os.environ.get("TTS_COSY_GPU_ISOLATED", "").strip().lower()
    if forced in ("1", "true", "yes"):
        return True
    if forced in ("0", "false", "no"):
        return False
    try:
        mine = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        theirs = ""
        with open(COSY_UNIT, encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("Environment=CUDA_VISIBLE_DEVICES="):
                    theirs = s.split("=", 1)[1].strip()
                    break
        if not mine or not theirs:
            return False
        return mine != theirs
    except Exception as e:
        log.warning("判断 Cosy 是否异卡失败（按同卡处理，继续串行）: %s", repr(e)[:120])
        return False


def switch_to(engine_name, log_prefix=""):
    """跑 engine_name 之前，把另一个引擎赶出显存并等显存真的释放。幂等，可重复调用。

    两个引擎**不在同一张卡**时直接跳过 —— 这时候卸载本地引擎没有任何意义，
    只会白白付出加载开销（详见 _cosy_gpu_isolated 的说明）。
    """
    if _cosy_gpu_isolated():
        log.info("%s引擎按 GPU 隔离运行（Qwen 与 Cosy 不在同一张卡），跳过显存串行切换",
                 log_prefix)
        return None
    if engine_name == "cosyvoice3":
        release_qwen()
        free = wait_free_vram(MIN_FREE_FOR_COSY)
    else:
        release_cosy()
        free = wait_free_vram(MIN_FREE_FOR_QWEN)
    if free is not None:
        log.info("%s切换引擎 -> %s，整卡空闲显存 %.1f GiB", log_prefix, engine_name, free)
    return free


def restore_qwen(warm=True):
    """收尾：卸掉 CosyVoice3，把 Qwen 恢复到常驻稳态（下个任务零加载等待）。

    GPU 隔离时 Qwen 压根没被卸过，所以整段都可以跳过 —— 连 warmup 都不用做。
    """
    if _cosy_gpu_isolated():
        return None
    switch_to("qwen")
    if not warm:
        return
    try:
        import server
        eng = getattr(server, "engine", None)
        if eng is not None:
            log.info("收尾：重新加载 Qwen 恢复常驻稳态")
            eng.warmup()
    except Exception as e:
        # 不致命：下次生成会懒加载兜底
        log.warning("重载 Qwen 失败（下次生成将懒加载）: %s", repr(e)[:160])
