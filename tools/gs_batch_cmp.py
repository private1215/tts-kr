# -*- coding: utf-8 -*-
"""批量「同文案」对照生成：GS 微调(+分段) vs GS 零样本。

🔴 与 gs_infer_group.py 的区别
    gs_infer_group.py 的测试文本写死成 hold 案例前 60 秒（一段）。
    这里支持**多段文案**（同一个说话人、同一套 prompt），用来看结论
    是否只在特定文案上成立 —— 单一文案的对比是可能被偶然性污染的。

输入：
    --spk      说话人组名（决定用哪套微调权重）
    --prompt   prompt 音频（必须来自未训练区）
    --ptxt     prompt 对应文字稿
    --texts    逗号分隔的若干文案 txt 路径
输出：
    out/gs_<spk>_<tag>_ft_e<ep>[_seg]/... 见 --out-prefix

⚠️ 一律启用外部分段（默认 50 字/段）：整段喂给 GS 会触发自回归发散，
   女声实测覆盖率 0.15 → 0.96。除非显式 --no-seg 做反面参照。
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

REPO = "/mnt/tts/projects/gpt-sovits/repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(REPO)

import torch                                              # noqa: E402
import torchaudio                                         # noqa: E402


def _sf_load(path):
    w, sr = sf.read(path, dtype="float32")
    t = torch.from_numpy(np.asarray(w))
    if t.ndim == 1:
        t = t.unsqueeze(0)
    else:
        t = t.transpose(0, 1)
    return t, sr


torchaudio.load = _sf_load

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config   # noqa: E402

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")
import gs_infer_group as G                                   # noqa: E402

OUT = "/mnt/tts/projects/gpt-sovits/out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spk", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ptxt", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--epoch", default="40")
    ap.add_argument("--out-prefix", default="cmp")
    ap.add_argument("--seg", type=int, default=50)
    ap.add_argument("--no-seg", action="store_true")
    a = ap.parse_args()

    ptxt = open(a.ptxt, encoding="utf-8").read().strip()
    w = G.pick_weight(a.spk, a.epoch)
    texts = [t.strip() for t in a.texts.split(",") if t.strip()]

    for tp in texts:
        txt = open(tp, encoding="utf-8").read().strip()
        tag = os.path.basename(tp).replace("text_", "").replace(".txt", "")
        for mode, sov in (("ft", w), ("zs", G.PRETRAINED_SOVITS)):
            name = "%s_%s_%s_e%s%s" % (
                a.out_prefix, tag, mode, a.epoch,
                "" if a.no_seg else "_seg%d" % a.seg)
            dst = os.path.join(OUT, name + ".wav")
            if os.path.exists(dst):
                print("  跳过(已存在)", name, flush=True)
                continue
            tts = TTS(G.build(sov))
            try:
                if a.no_seg:
                    G.synth(tts, txt, a.prompt, ptxt, dst)
                else:
                    G.synth_segs(tts, txt, a.prompt, ptxt, dst,
                               max_chars=a.seg)
            finally:
                del tts
                torch.cuda.empty_cache()
            print("  ->", name, flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
