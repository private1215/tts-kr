# -*- coding: utf-8 -*-
"""3090 带载压力测试：可控时长 + 可控显存/算力占用，用于复现/排除掉卡。

背景
----
2026-10-07 03:13 训练到第 30 分钟时 3090 掉卡：

    pcieport 0000:00:03.0: AER: Uncorrectable (Fatal), type=Transaction Layer
      device [8086:2f08] error status/mask=00000020/00000000
      [ 5] SDES   (First)          <- Surprise Down：设备从链路上消失
    NVRM: Xid (PCI:0000:04:00): 79, GPU has fallen off the bus.

前面**没有任何**可纠正错误（无 BadTLP / BadDLLP / RxErr / Replay），
所以这不是「链路裕量不足」，更像供电/接触在持续大电流下撑不住。
本脚本用来在**限定功耗**下长时间带载，验证「不满负荷」是否能稳定。

用法（在服务器上，qwen 环境即可）：
    sudo nvidia-smi -i 1 -pl 250            # 先限功耗
    python tools/gpu_stress.py --minutes 35 --gb 14 --uuid GPU-b67eee2c-...
"""
import os
import sys
import time
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=35.0)
    ap.add_argument("--gb", type=float, default=14.0, help="显存占用 GB")
    ap.add_argument("--uuid", default="GPU-b67eee2c-c672-6915-910d-617c62abebee")
    ap.add_argument("--log", default="/tmp/gpu_stress.log")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.uuid
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return 1
    print("device:", torch.cuda.get_device_name(0), flush=True)

    n = int(a.gb * 1024 ** 3 / 4)
    bufs = []
    try:
        while n > 0:
            try:
                bufs.append(torch.empty(n, dtype=torch.float32, device="cuda"))
                break
            except torch.OutOfMemoryError:
                n = int(n * 0.8)
    except Exception as e:
        print("分配失败", repr(e)[:200])
        return 2
    print(f"常驻显存 {n * 4 / 1024 ** 3:.1f} GB", flush=True)
    bufs[0].zero_()

    # 大矩阵乘 + 卷积混合，尽量贴近训练的实际负载（tensor core + 显存带宽）
    m, k, nn = 4096, 4096, 4096
    w = torch.randn(m, k, device="cuda", dtype=torch.float16)
    conv = torch.randn(64, 64, 7, 7, device="cuda", dtype=torch.float16)
    img = torch.randn(8, 64, 96, 96, device="cuda", dtype=torch.float16)

    t0 = time.time()
    last = 0.0
    f = open(a.log, "a")
    it = 0
    try:
        while time.time() - t0 < a.minutes * 60:
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            for _ in range(4):
                x = torch.mm(x, w)
                x = torch.relu_(x)
            img2 = torch.conv2d(img, conv, padding=3)
            img = (img + img2 * 0.01).contiguous()
            torch.cuda.synchronize()
            it += 1
            el = time.time() - t0
            if el - last >= 30:
                last = el
                line = (f"{time.strftime('%H:%M:%S')} t={el/60:.1f}min it={it} "
                        f"mem={torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
                print(line, flush=True)
                f.write(line + "\n")
                f.flush()
            del x, img2
    except Exception as e:
        msg = f"!! 异常 {time.strftime('%H:%M:%S')} {repr(e)[:300]}"
        print(msg, flush=True)
        f.write(msg + "\n")
        return 3
    finally:
        f.close()
    print(f"完成 {a.minutes} 分钟，无异常", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
