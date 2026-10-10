# -*- coding: utf-8 -*-
"""反馈题防作弊核对：**任何选项都不得与参照是同一条音频**。

为什么必须查：参照是案例真人原声。要是某个选项不小心就是它（或者退一步，
是它的重编码副本），员工一眼就能选出"最像本人"的那个 —— 这道题当场作废，
而我们还会拿着这个错误结论去改引擎。这种错在项目里是完全静默的。

查两件事：
  1. 每个选项 vs 参照的 campplus 余弦：接近 1.0 就是同一条
  2. 选项两两之间的余弦：接近 1.0 只说明"同一个人"，**不一定是错的**
     —— pause_* / tone_* 这类题本来就是拿同一条音频改一个参数生成的，
       所以脚本把它们和"泄露"分开报，不要混为一谈。

用法：
    python tools/fb_dupchk.py
"""
import os
import subprocess
import sys
import itertools

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from spkchk import read16, embed, CAMPPLUS  # noqa: E402

FB = os.environ.get("TTS_FB_DIR", "/mnt/tts/data/tone-ab")
LEAK_TH = 0.99     # 高于此 -> 疑似和参照是同一条音频


def load_items():
    import json
    p = os.path.join(FB, "items.json")
    return json.load(open(p, encoding="utf-8"))["items"]


def md5(path):
    r = subprocess.run(["md5sum", path], capture_output=True, text=True)
    return r.stdout.strip().split()[0] if r.stdout.strip() else ""


def main():
    sess = ort.InferenceSession(CAMPPLUS, providers=["CPUExecutionProvider"])
    cache = {}

    def emb(f):
        if f not in cache:
            cache[f] = embed(sess, read16(os.path.join(FB, f)))
        return cache[f]

    leak = 0
    for it in load_items():
        ref = (it.get("ref") or {}).get("file")
        opts = [(o["key"], o["file"], o.get("label", ""))
                for o in it.get("options", []) if o.get("file")]
        print()
        print("── %s  ［%s］" % (it.get("title", it["id"]), it.get("dim", "-")))
        print("   id=%s" % it["id"])

        if ref:
            e0 = emb(ref)
            h0 = md5(os.path.join(FB, ref))
            worst = 0.0
            for k, f, lab in opts:
                s = float(np.dot(e0, emb(f)))
                worst = max(worst, s)
                flag = ""
                if s >= LEAK_TH or md5(os.path.join(FB, f)) == h0:
                    flag = "  🔴 疑似与参照同一条！"
                    leak += 1
                print("      选项 %-3s %-22s vs 参照 %.4f%s" % (k, lab, s, flag))
            print("      -> 本项最高 %.4f（阈值 %s）%s"
                  % (worst, LEAK_TH, "❌ 有泄露" if worst >= LEAK_TH else "✅ 没泄露"))

        same = []
        for (k1, f1, _), (k2, f2, _) in itertools.combinations(opts, 2):
            s = float(np.dot(emb(f1), emb(f2)))
            if s >= LEAK_TH:
                same.append((k1, k2, s))
        if same:
            print("      ℹ️  以下选项两两相似度 >= %s：" % LEAK_TH)
            for k1, k2, s in same:
                print("         %s 与 %s：%.4f" % (k1, k2, s))
            print("         多数情况是**同一个人改一个参数**（停顿/句尾类题目本就如此），")
            print("         不是错误；但如果这是「不同引擎」的题，就要回头查出题环节。")

    print()
    print("=" * 60)
    print("泄露判定：%s" % ("🔴 有 %d 处需要修" % leak if leak else "✅ 没有选项与参照重复"))


if __name__ == "__main__":
    main()
