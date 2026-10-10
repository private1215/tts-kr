# -*- coding: utf-8 -*-
"""为 GPT-SoVITS 准备韩华案例(이민정)的训练数据。

🔴 关键：**必须留出测试区**，否则模型只是背出了对比用的那几句，
对比结果毫无意义（这正是 GPT 方案里"训练集与测试集分离"那条）。

做法：
  - 测试区 = 案例 merged.wav 的前 HOLD_SEC 秒
    （这段对应我们线上对比用的参照 hh_ref.mp3，以及 Qwen3-TTS 生成的 hh_el）
  - 训练集 = 测试区**之后**的全部音频
  这样 GPT-SoVITS 生成对比文案时是真正的"没见过"，和我们现有引擎公平对比。

输出：
  dataset/train/*.wav + dataset/train.list
  dataset/val/*.wav   + dataset/val.list
"""
import os, sys, subprocess, random
import numpy as np

CASE = "/mnt/tts/data/cases/c3117b7b0fb1"
MERGED = os.path.join(CASE, "merged.wav")
OUT = "/mnt/tts/projects/gpt-sovits/dataset"
SPK = "hh_minjung"
LANG = "ko"
HOLD_SEC = 60.0        # 前 60 秒留给测试（覆盖面比 41 秒参照更宽，保险）
SR_OUT = 32000         # v2Pro 系列原生 32k

# 切片参数
MIN_SEC, MAX_SEC = 2.0, 11.0
SIL_THRESH_DB = -38.0
SIL_MIN_SEC = 0.35
FRAME = 0.02

os.makedirs(os.path.join(OUT, "train"), exist_ok=True)
os.makedirs(os.path.join(OUT, "val"), exist_ok=True)


def load_wav(path, sr_out=SR_OUT):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != sr_out:
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
             "-ac", "1", "-ar", str(sr_out), "-f", "f32le", "-"],
            capture_output=True)
        x = np.frombuffer(r.stdout, dtype="<f4")
        sr = sr_out
    else:
        sr = sr_out
    return np.asarray(x, dtype=np.float32), sr


def slice_(x, sr):
    """按静音切成短段。返回 [(start_s, end_s), ...]"""
    n = int(FRAME * sr)
    m = (len(x) // n) * n
    fr = x[:m].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)
    voiced = db > SIL_THRESH_DB
    # 找出有声区间
    segs, s = [], None
    for i, v in enumerate(voiced):
        if v and s is None:
            s = i
        elif not v and s is not None:
            segs.append((s, i)); s = None
    if s is not None:
        segs.append((s, len(voiced)))
    # 合并间隔小于 SIL_MIN_SEC 的相邻有声段
    merged = []
    for a, b in segs:
        if merged and (a - merged[-1][1]) * FRAME < SIL_MIN_SEC:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    out = []
    for a, b in merged:
        st, en = a * FRAME, b * FRAME
        d = en - st
        if d < MIN_SEC:
            continue
        if d <= MAX_SEC:
            out.append((st, en))
        else:
            # 长段：在最长静音处递归二分（简单起见按等长 + 静音吸附）
            k = int(np.ceil(d / MAX_SEC))
            step = d / k
            for j in range(k):
                out.append((st + j * step, min(st + (j + 1) * step, en)))
    return out


def main():
    print("载入 %s" % MERGED, flush=True)
    x, sr = load_wav(MERGED)
    total = len(x) / sr
    print("总时长 %.1fs" % total, flush=True)

    # ---- 测试区：前 HOLD_SEC 秒（不参与训练）----------------------------
    hold_n = int(HOLD_SEC * sr)
    test_audio = x[:hold_n]
    train_audio = x[hold_n:]
    print("测试区 %.1fs / 训练区 %.1fs" % (HOLD_SEC, (len(x) - hold_n) / sr), flush=True)
    with open(os.path.join(OUT, "..", "hold_test.txt"), "w") as f:
        f.write("前 %.1f 秒留作测试，不进训练集\n" % HOLD_SEC)

    segs = slice_(train_audio, sr)
    print("切出 %d 段" % len(segs), flush=True)

    # ---- ASR 逐段转写 ---------------------------------------------------
    sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
    os.environ.setdefault("HF_HOME", "/mnt/tts/models/asr")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from app.asr_check import transcribe
    print("ASR 载入完成", flush=True)
    import tempfile
    import soundfile as _sf
    _tmp = os.path.join(OUT, "_tmp_seg.wav")

    rows = []
    for i, (st, en) in enumerate(segs):
        seg = train_audio[int(st * sr):int(en * sr)]
        _sf.write(_tmp, seg, SR_OUT, subtype="PCM_16")
        txt = transcribe(_tmp, "ko")          # 只收文件路径，返回 str
        txt = (txt or "").strip()
        if len(txt) < 4:
            continue
        rows.append({"i": i, "st": st, "en": en, "txt": txt})
        if i % 10 == 0:
            print("  已转写 %d/%d" % (i, len(segs)), flush=True)
    if os.path.exists(_tmp):
        os.remove(_tmp)
    print("有效样本 %d 条" % len(rows), flush=True)

    # ---- 训练 / 验证分离 -------------------------------------------------
    random.Random(42).shuffle(rows)
    n_val = max(3, int(len(rows) * 0.12))
    val, train = rows[:n_val], rows[n_val:]

    def dump(subset, split):
        lines = []
        for r in subset:
            seg = train_audio[int(r["st"] * sr):int(r["en"] * sr)]
            name = "%s_%05d.wav" % (SPK, r["i"])
            p = os.path.join(OUT, split, name)
            import soundfile as sf
            sf.write(p, seg, SR_OUT, subtype="PCM_16")
            lines.append("%s|%s|%s|%s" % (p, SPK, LANG, r["txt"]))
        with open(os.path.join(OUT, split + ".list"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return len(lines)

    nt = dump(train, "train")
    nv = dump(val, "val")
    print("训练集 %d 条，验证集 %d 条" % (nt, nv), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
