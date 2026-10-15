#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分级验证：**短片过关 → 才允许上长篇**（2026-10-15 用户纠正后定案）。

## 用户原话
> **「可以短片先过关，然后优化长篇，直接上长篇文案生成语音好像不合理。」**

## 为什么必须有这道闸（实测，不是推理）
短片 = `P1-smoke-*` 的 18 秒成品，同一配置连续 7 次生成：

| 次 | Σz² |
|---|---|
| 2 | 190.6 |
| 3 | 235.9 |
| 4 | 753.2 |
| 5 | 425.3 |
| 6 | 426.8 |
| 7 | 650.0 |
| 8 | 527.4 |

**波动 4 倍**（190.6 ~ 753.2），中位约 477。
→ **短片的绝对分数完全不能用来判「过没过」**，它连自洽都做不到。

更糟的是**系统偏差**（第 4/6/8 次）：

| 维度 | 短片实测 | EL | 结论 |
|---|---|---|---|
| **4k~8k** | 0.0034 ~ 0.0070 | 0.0129 | **系统性低 40%，高频还没建立起来** |
| 150~300 | 0.2809 ~ 0.3044 | 0.2541 | 系统性偏高 10~20% |

→ 18 秒里说话人还没进入稳定状态，高频（齿音、气息）没铺开。
**拿短片的频谱数字去调参，等于照着假数据改**。

## 但短片不是没用 —— 它能回答另一个问题
| 阶段 | 长度 | 能不能判「过没过」 | 能回答什么 |
|---|---|---|---|
| **L1 快筛** | 15~30 秒 | ❌ 不能（波动4倍） | **改动有没有生效**（相对比较）、链路通不通、别崩 |
| **L2 短篇** | 60~120 秒 | ⚠️ 勉强（要跑 3 次取中位） | 大改动的方向筛查 |
| **L3 长篇** | ≥180 秒，**最好 ≥5 分钟** | ✅ 唯一可信 | 真正的验收 |

🔴 **L1 的正确用法是「A/B 相对比较」，不是「打分」**：
同一批短片里，改动前后各跑几次，比**中位数**的差异，不比单次的绝对值。

## 本脚本做什么
跑完 L1 后**自动判定闸门**：
- 只有**全部**满足才允许进 L2/L3
- 任一项不过就**明确报告"先别上长篇"**，并指出该修什么

## 用法
    python gate_l1.py --gender male            # 只跑 L1 并判定
    python gate_l1.py --gender male --text 600字  # 自定义长度
"""
import argparse
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
import uuid

BASE = "http://127.0.0.1:8000"
TOK = os.environ.get("TTS_DEBUG_TOKEN", "smoke-8f3a2c91")
VENV = "/mnt/tts/projects/qwen3-tts/venv/bin/python3"
PROBE = "/mnt/tts/diag/x8197/probe_g.py"
CASES = "/mnt/tts/data/cases"

# L1 闸门（实测推导，门槛故意宽松——短片只判「离谱」不判「精细」）
GATE = {
    "dur_min": 8.0,             # 太短说明生成有问题
    "dur_max": 120.0,
    "z2_max": 4000.0,           # 短片 Σz² 实测 190~753，放宽到 4000
    "hf_min_ratio": 0.45,       # 4k~8k / EL 比值下限（实测 0.26~0.54）
    "spread_max": 3.0,          # 3 次生成的 Σz² 中位数散布上限（实测2.2）
}


def post(path, data):
    body = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Authorization": "Bearer " + TOK})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


def wait(jid, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        req = urllib.request.Request(
            "%s/api/jobs/%s" % (BASE, jid),
            headers={"Authorization": "Bearer " + TOK})
        d = json.loads(urllib.request.urlopen(req, timeout=30).read())
        if d.get("status") in ("done", "error", "failed"):
            return d
        time.sleep(5)
    return d


def probe(path, gender):
    r = subprocess.run([VENV, PROBE, gender, path],
                       capture_output=True, text=True, timeout=900).stdout
    out = {"path": path}
    for line in r.splitlines():
        if "Σz²=" in line and "我们" not in line:
            try:
                out["sigz2"] = float(line.split("Σz²=")[1].split()[0])
            except (IndexError, ValueError):
                pass
        for key, nm in (("150~300", "b150_300"), ("4k~8k", "b4000_8000"),
                        ("平坦度", "flat"), ("发声占比", "voiced")):
            if key in line and "我们" in line:
                p = line.split()
                try:
                    out[nm] = float(p[3])
                except (IndexError, ValueError):
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", default="male", choices=["male", "female"])
    ap.add_argument("--voice", default="")
    ap.add_argument("--text", default="")
    ap.add_argument("--case", default="")
    ap.add_argument("--runs", type=int, default=3,
                    help="跑几次取中位（L1 散布大，必须≥3）")
    a = ap.parse_args()
    voice = a.voice or ("EL_male" if a.gender == "male" else "EL_female")

    # 取文案：优先指定 case，否则拼案例 seg0 的开头一段
    if a.text:
        text = a.text
    elif a.case:
        text = open("%s/%s/seg0.txt" % (CASES, a.case),
                    encoding="utf-8").read()
    else:
        t = open("%s/1a6c20a60481/seg0.txt" % CASES,
                 encoding="utf-8").read()
        text = t[:600]
    print("L1 快筛  性别=%s  音色=%s  文案 %d 字  跑 %d 次"
          % (a.gender, voice, len(text), a.runs))

    tag = "l1-%s-%s" % (a.gender, uuid.uuid4().hex[:6])
    res = []
    for i in range(a.runs):
        r = post("/api/jobs", {"text": text, "voice": voice,
                               "mode": "clone", "speed": "1.0",
                               "emotion": "standard", "seed": str(42 + i),
                               "title": "%s-%d" % (tag, i + 1)})
        d = wait(r["id"])
        if d.get("status") != "done":
            print("  第%d次 失败：%s" % (i + 1, d.get("status")))
            continue
        jid = r["id"]
        # 成品落在 /mnt/tts/output/<title>/<title>-1.mp3
        outroot = "/mnt/tts/output"
        want = "%s-%d" % (tag, i + 1)
        cand = next((d for d in os.listdir(outroot)
                     if d.startswith(want)
                     and os.path.isdir(os.path.join(outroot, d))), None)
        if not cand:
            print("  第%d次 找不到成品目录（找 %s*）" % (i + 1, want))
            continue
        fs = [f for f in os.listdir(os.path.join(outroot, cand))
              if f.endswith(".mp3")]
        if not fs:
            continue
        p = os.path.join(outroot, cand, fs[0])
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", p], capture_output=True,
            text=True).stdout.strip() or 0)
        m = probe(p, a.gender)
        m["dur"] = dur
        m["job"] = cand
        res.append(m)
        print("  第%d次 %5.1fs  Σz²=%8s  150~300=%s  4k~8k=%s"
              % (i + 1, dur, m.get("sigz2"),
                 m.get("b150_300"), m.get("b4000_8000")))

    if not res:
        print("\n❌ 全部失败，无法判定")
        return

    js = sorted(x["sigz2"] for x in res if x.get("sigz2"))
    med = js[len(js) // 2]
    spread = (js[-1] - js[0]) / max(js[0], 1e-9)
    hf = [x["b4000_8000"] / 0.0129 for x in res if x.get("b4000_8000")]
    hf_min = min(hf) if hf else 0

    print("\n=== L1 闸门判定 ===")
    checks = [
        ("时长在 %g~%gs" % (GATE["dur_min"], GATE["dur_max"]),
         all(GATE["dur_min"] <= x["dur"] <= GATE["dur_max"] for x in res),
         "生成异常（太短=截断，太长=参数没生效）"),
        ("Σz² 中位 %.1f < %g" % (med, GATE["z2_max"]),
         med < GATE["z2_max"],
         "短片分数虚高说明链路有问题，别急着上长篇"),
        ("4k~8k 最低比值 %.2f > %.2f" % (hf_min, GATE["hf_min_ratio"]),
         hf_min > GATE["hf_min_ratio"],
         "**高频还没建立起来**——18秒不够测高频，调它只会照着假数据改"),
        ("%d 次散布 %.2f < %.1f" % (len(js), spread, GATE["spread_max"]),
         spread < GATE["spread_max"],
         "连自洽都做不到，先修稳定性再谈其他"),
    ]
    ok = True
    for desc, passed, why in checks:
        print("  %s %s" % ("✅" if passed else "❌", desc))
        if not passed:
            print("       → %s" % why)
            ok = False

    out = {"gender": a.gender, "runs": res, "median_sigz2": med,
           "spread": spread, "hf_min_ratio": hf_min, "pass": ok}
    jp = "/mnt/tts/diag/muffle/l1_gate_%s.json" % a.gender
    json.dump(out, open(jp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n-> %s" % jp)
    if ok:
        print("\n✅ **L1 通过**，可以跑 L2/L3（60秒~5分钟）做真正的验收")
        print("   L1 只说明「链路正常」，不代表质量达标——质量必须用长篇判")
    else:
        print("\n🔴 **L1 未过，先别上长篇**。按上面的提示修完再跑。")


if __name__ == "__main__":
    main()