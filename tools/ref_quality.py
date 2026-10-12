# -*- coding: utf-8 -*-
"""候选参考切片横向打分：谁最"像这个人"，零样本上限就是它。

为什么需要这个
--------------
零样本克隆的音色天花板 = 参考音频本身对目标说话人的代表性。
现在线上 EL_male / EL_female 的 15s 参考是**当年从某一条案例随手切的**，
而且 voices.json 里记着 EL_male 对案例原声只有 0.9075 —— 也就是说光是
"参考自己"就已经先丢了 0.09。

所以别急着调引擎，先把"拿哪 15 秒当参考"这件事量化：
对每个候选切片，算它与**该说话人全部案例原声**的 campplus 余弦，取均值/中位/最低。

用法
----
    venv/bin/python tools/ref_quality.py            # 全量评估
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = "/mnt/tts/projects/qwen3-tts"
sys.path.insert(0, os.path.join(ROOT, "tools"))
import spkchk  # noqa: E402

CAMPPLUS = os.environ.get(
    "TTS_CAMPPLUS", "/mnt/tts/models/Fun-CosyVoice3-0.5B/campplus.onnx")
CASES = "/mnt/tts/data/cases"
VOICES = "/mnt/tts/data/voices/voices.json"

_SESS = None


def sess():
    global _SESS
    if _SESS is None:
        import onnxruntime
        _SESS = onnxruntime.InferenceSession(
            CAMPPLUS, providers=["CPUExecutionProvider"])
    return _SESS


def wav16(path):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", tmp.name], check=True)
    return tmp.name


def emb(path):
    w = wav16(path)
    try:
        return spkchk.embed(sess(), spkchk.read16(w))
    finally:
        os.unlink(w)


def cos(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def clip(src, start, dur, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start),
                    "-t", str(dur), "-i", src, "-ar", "16000", "-ac", "1", out],
                   check=True)
    return out


GROUPS = {
    "yohan(男)": ["799e4d7060ce", "fc31abeae1b0", "bfcbda560255", "1e15fae990d9",
                  "d172b8c4efb3", "31040ddc36e5", "8197d9ac8a35", "6cdd25bc75f2",
                  "ba24c0925311", "ed307f65ab58", "5bef616b57f7", "81c2d4fc5a99",
                  "1a6c20a60481"],
    "SOLA(女)": ["7717df971756", "b74265f0f848", "9eb2ed4904d3", "881e19326691",
                 "ab716584ab7e", "266a04225c01", "b029e7771a1c", "ea3154b3040f",
                 "ce1a3fda551a", "a79503d0b0ca", "557b83ff98c8", "19c3b37643d6",
                 "b5cee5cfa4f4", "1e9d7d7ad2cc", "481fefa98fea", "699ce016a078",
                 "6fccb199a4bf", "52727ec43154", "9742074652c4", "2e515c65c615",
                 "00b6a45ad6aa"],
}

# (显示名, 音频路径) —— 现役 / 质心法新挑 / 历史高分那支
CAND = {
    "yohan(男)": [
        ("现役 EL_male(15s,取自1a6c20a60481)", "/mnt/tts/data/voices/EL_male.wav"),
        ("现役 yohan_male(15s)", "/mnt/tts/data/voices/yohan_male.wav"),
        ("历史高分prompt(10s,取自1a6c20a60481)",
         "/mnt/tts/projects/gpt-sovits/out/prompt_yohan_1a6c20a6.wav"),
        ("质心挑源 81c2d4fc@50s", None),
        ("质心挑源 81c2d4fc@105s", None),
        ("质心挑源 6cdd25bc@80s", None),
    ],
    "SOLA(女)": [
        ("现役 EL_female(15s,取自00b6a45ad6aa)", "/mnt/tts/data/voices/EL_female.wav"),
        ("现役 SOLA_female(15s)", "/mnt/tts/data/voices/SOLA_female.wav"),
        ("历史高分prompt(10s,取自00b6a45ad6aa)",
         "/mnt/tts/projects/gpt-sovits/out/prompt_sola_00b6a45a.wav"),
        ("质心挑源 97420746@175s", None),
        ("质心挑源 a79503d0@70s", None),
        ("质心挑源 ea3154b3@190s", None),
    ],
}
CLIPS = {  # (case_id, start) -> 15s 切片
    "质心挑源 81c2d4fc@50s": ("81c2d4fc5a99", 50.0),
    "质心挑源 81c2d4fc@105s": ("81c2d4fc5a99", 105.0),
    "质心挑源 6cdd25bc@80s": ("6cdd25bc75f2", 80.0),
    "质心挑源 97420746@175s": ("9742074652c4", 175.0),
    "质心挑源 a79503d0@70s": ("a79503d0b0ca", 70.0),
    "质心挑源 ea3154b3@190s": ("ea3154b3040f", 190.0),
}


def main():
    tmpd = "/tmp/refq"
    os.makedirs(tmpd, exist_ok=True)
    for gname, cids in GROUPS.items():
        print("\n===== %s：参照 = 该组 %d 条案例原声 =====" % (gname, len(cids)))
        refs = {}
        for cid in cids:
            p = os.path.join(CASES, cid, "merged.wav")
            if os.path.exists(p):
                refs[cid] = emb(p)
        print("（已算 %d 条案例原声嵌入）" % len(refs))
        for name, path in CAND[gname]:
            if path is None:
                cid, st = CLIPS[name]
                p = clip(os.path.join(CASES, cid, "merged.wav"), st, 15.0,
                         os.path.join(tmpd, "%s.wav" % name.replace("/", "_")))
            elif not os.path.exists(path):
                print("  MISSING", name)
                continue
            else:
                p = path
            e = emb(p)
            sims = [cos(e, r) for r in refs.values()]
            print("  %-42s 均值 %.4f  中位 %.4f  最低 %.4f  最高 %.4f"
                  % (name, float(np.mean(sims)), float(np.median(sims)),
                     float(np.min(sims)), float(np.max(sims))))


if __name__ == "__main__":
    main()
