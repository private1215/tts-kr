#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用修复链（把 raw 合成结果加工到可交付）——新文案版。

与 female_fix_v3 / male_fix_v3 的差别：那两个脚本硬要求「同一篇文案的 EL 参考」，
拿来做逐窗速率对齐。但**新文案没有 EL 参考**（我们无法生成新的 ElevenLabs 音频），
所以这里支持两种定速率的方式：
  --el <wav>          有同文本 EL：逐窗对齐（原口径，最准）
  --target-rate <r>   无 EL：对齐到 EL 的**典型速率**（默认 女8.5 / 男8.9）
                      这两个数来自 52727ec43154(女) 与 8197d9ac8a35(男) 的实测均值。

流程（与获胜链路一致，顺序不能换）：
  ① 逐 60s 窗音节核速率对齐 → 在静音处切窗做 atempo（只放慢，不加快，钳 [1.0,1.30]）
  ② 停顿清洗：≥0.08s 停顿内容整体替换为合成 clean room tone（长度不变、5ms 淡化）
  ③ 男声压亮 EQ（highshelf 4000 -4dB, lowpass 14000）；**女声不加 EQ**
     —— v2 实测女声频谱已与 EL 对齐，加 EQ 反而更差
  ④ 响度对齐到目标 LUFS（女 -22.6 / 男 -25.0，取自 EL 实测）

用法:
  fix_chain.py <raw.wav> <out.wav> [--gender f|m] [--el EL.wav]
               [--target-rate 8.5] [--lufs -22.6] [--no-clean]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import shutil
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "..", "diag", "cross-20261013"),
          "/mnt/tts/diag/cross-20261013"):
    if p not in sys.path:
        sys.path.insert(0, p)

from pause_clean_v3 import clean_gaps, detect_gaps_wav, verify_pauses  # noqa: E402
import declick  # noqa: E402  v3.6 数字零填充（见 main 末尾 ZEROFILL 段）

# ---- v3.6 数字零填充（ZEROFILL）------------------------------------------
# 2026-10-21 员工反馈 202610210738（男）/ 202610210939（女）：逐秒报
#   「0 秒开头异声」「8 秒 0.1 秒没音」「14 秒 0.1 秒额音」「27/42/57…秒 嗯」
#   「电磁音全程很多」「碰碰声全程个别地方有」。
# 取证（tools/fback53d.py）：员工报的 26 个时间点里 **24 个**精确落在
#   |y| < 1e-9 的数字零段上（±1.5s），男 52 处 / 女 44 处 ≥20ms 全零段，
#   而 EL 标杆同口径 **0 处** —— 我们的停顿是模型输出的**数字真零**，不是安静。
# 🔴 与第 7 节「已证伪：抬停顿底噪」不是一回事，别混：
#    · 已证伪的是「把停顿抬到 -56/-50dB 做出房间感」→ z44 证明员工听不出；
#    · 这里填的是 **-80dB**，只把 < -80dB 的深坑顶到 -80dB，
#      目标是**消除硬切**（0 与 -24dB 之间 0.1 秒跳 130dB 的「咔/没音」），
#      不是制造房间感。填完静音段中位数仍是 -60（EL -59），分布主体不变。
# 开关：TTS_ZEROFILL=0 关闭（C 级变更的一键回退）。
ZEROFILL_ON = os.environ.get("TTS_ZEROFILL", "1").strip().lower() not in (
    "0", "false", "no", "off")
# 填充电平 = 门限，同取 -80dB：低于 -80dB 的样本顶到 -80dB 的粉噪。
# 网格实测（tools/zerofix53b.py，男女各 6 档）：
#   男 零段 52->0 / 陡降 169->42（EL 28）/ p5 -138->-95（EL -91）/ 语音样本改动 0
#   女 零段 44->0 / 陡降  45-> 5（EL  0）/ p5  -83->-81（EL -76）/ 语音样本改动 0
#   时长零变化，p25/med 不变（男 -76/-60 vs EL -70/-59；女 -62/-54 vs EL -61/-54）
# ⚠️ -80dB 比自身静音中位（-60/-54）低 20~26dB，不会盖住真实停顿起伏。
ZEROFILL_DB = float(os.environ.get("TTS_ZEROFILL_DB", "-80"))

WIN = 60.0
# EL 典型速率（音节核/秒，实测均值）：女 8.51 / 男 8.93
DEFAULT_RATE = {"f": 8.5, "m": 8.9}
# 🔴 EL 实测「字/秒」（chars per second）：女 7.45 / 男 8.37
#   这才是**跨文案可比**的锚。2026-10-14 踩坑：用音节核速率当固定目标时，
#   男声成品 9.17 字/秒 vs EL 8.37，快了 9.6%，员工直接判「不像真人」。
#   老脚本没出事，是因为它们拿**同一篇文案的 EL** 逐窗对齐，量纲差互相抵消；
#   一旦换成固定常数，音节核/秒 ≠ 字/秒 的差异就暴露了。
# 🔴 女声 2026-10-14 复核：4 条 EL 案例实测 7.45 / 7.622 / 7.507 / 7.612 → 均值 7.573。
#   之前锚在单条案例的 7.45，等于我们比 EL 快 1.6%，员工连续两轮点名「女声太快」。
DEFAULT_CPS = {"f": 7.57, "m": 8.37}
# EL 实测响度：女 -22.6 / 男 -25.0
DEFAULT_LUFS = {"f": -22.6, "m": -25.0}
# 第31轮：男声目标 F0 197.7 -> 188.0。197.7 需要 R=1.094 上搬，
#   asetrate 把共振峰一起搬走，音色相似度 0.684 掉到「同人」阈值 0.75 以下；
#   188 只需 R≈1.05，音色 0.797，Σz² 37.4 仍落进 EL 自身分布(17~240)。
DEFAULT_F0 = {"f": 276.0, "m": 188.0}
# 🔴 变调会把整个频谱搬上去：男声升 1.21 倍后 >4kHz 能量占比 0.0185，
#    而 EL 只有 0.0104（高 78%，听感发尖）。10*log10(0.0104/0.0185) = -2.5dB，
#    所以变调后高频要压到 -6.5dB。未变调时保持默认 -4dB。
#    ❌ rubberband 的 formant=preserved（只升基频不动声道）**不可用**：
#       频谱平坦度 0.050 → 0.015（EL 0.042），声音发闷发假。
# 🔴 男声 EQ 四段式（2026-10-14 第五轮网格搜出来的，**全篇/中段口径**）：
#    equalizer f=800 w=1.2 -2dB（削鼻音共振区）
#    + highshelf 1500 +2.5dB（补 1~2k 存在感）
#    + highshelf 2500 -1.5dB（压 2~4k）
#    + highshelf 4000 %g（压 4~8k）
#    + lowpass 14000
#    搜索口径 [60s,360s] 对 2 条 EL 男声案例均值，L1 0.1777 → **0.0976**。
#    ⚠️ 口径教训：这个数**不能**用前 60s 搜（分窗变调后第一窗比例最大，
#       前 60s 是最坏样本，会过度补偿）。见 deploy/eq_search5.py 与 band_full.py。
#    未变调（--no-pitch）时高频不过量，%g 用 -4；变调用 -9.5。
EQ_MALE_TMPL = ("highpass=f=110:poles=2,"
                "equalizer=f=90:t=q:w=1.1:g=7.5,"
                "equalizer=f=350:t=q:w=1.1:g=3,"
                "equalizer=f=1400:t=q:w=1.1:g=3,"
                "equalizer=f=250:t=q:w=1.1:g=0.45,"
                "highshelf=f=4000:g=%g,lowpass=f=14000")

# 自适应高频用的「只有 boost、没有高频档」的第一段（见 adaptive_hf_db 注释）
# 🔴 末尾的 250Hz -0.75dB（2026-10-15）：修 150~300Hz +3.96σ 的「感冒闷」。
#    它是**系统性**的（8197 与 6cdd25 两篇都是 +3.96σ），根源是上面
#    90Hz +7.5dB / 350Hz +3dB 两个低频 boost 把这段抬过头了。
#    必须放在最后：数值是在成品上实测的增量。
EQ_MALE_BOOSTS = ("highpass=f=110:poles=2,"   # 第31轮：修 60~150Hz
                  "equalizer=f=90:t=q:w=1.1:g=7.5,"
                  "equalizer=f=350:t=q:w=1.1:g=3,"
                  "equalizer=f=1400:t=q:w=1.1:g=3,lowpass=f=14000")
# 🔴 250Hz 衰减（2026-10-15）：修男声 150~300Hz +3.96σ 的「感冒闷」。
#    两篇交叉验证（8197 / 6cdd25 都是 +3.96σ，系统性）：-0.75dB 让
#    Σz² 44.1->22.9 / 46.9->21.3，150~300 归零且其他项不恶化。
#    ⚠️ 离线(在成品上叠加)归零用 -0.75dB，但**端到端会过冲到 -2~-3.4σ**
#       （链路里的自适应/响度闭环与离线不等价），故按端到端反馈收到 -0.45dB。
#    -1.5dB 过冲（Σz² 反而升到 44.9 / 33.7）。
#    ⚠️ 必须放在**自适应高频之后**：自适应是按 boost 后的频谱反算
#       highshelf 的，衰减插在前面会把那个计算带偏（实测 Σz² -> 78.4）。
MALE_250 = "equalizer=f=250:t=q:w=1.1:g=0.75,"   # 第31轮：F0 目标下调后低频分布变了
HF_FREQ = 4000
HF_TARGET = 0.0129      # EL 男声 b4000_8000 均值（el_base_clean_male.json）
# 🔴 上界从 0.0 放开到 +6dB（2026-10-15）：原来「只衰减不提升」，于是
#    t5 的高频一旦**低于**目标就补不回来——实测 8197 篇 t5 的 b4000_8000
#    只有 0.0091（目标 0.0129），成品 -5.96σ，听感就是「闷」。
#    下界 -9.5 不动（那是当年把 flat 打到 -6σ 发假的教训）。
HF_DB_MIN, HF_DB_MAX = -9.5, 6.0
# 🔴 第31轮：150~300Hz 也改成自适应。写死的 -0.45dB 是按当年端到端
#    手调出来的，而总能量会随高频架子变化 → 各频段占比跟着浮动，
#    固定值不可能每次都刚好（历史：-1.5dB 过冲，Σz² 反而升到 44.9）。
T250_FREQ = 250
T250_TARGET = 0.254121      # EL 男声 b150_300 均值
T250_DB_MIN, T250_DB_MAX = -2.0, 1.0   # -1.5dB 已过冲，上限定在这里
MALE_250_DB = -0.45         # 迭代初值（沿用历史手调值）


def _band_ratio(path, lo, hi):
    """24k / n_fft 2048 / hop 512 的频段能量占比，与 el_baseline2 同口径。"""
    import librosa
    y, _ = librosa.load(path, sr=24000, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=24000, n_fft=2048)
    return float(S[(f >= lo) & (f < hi)].sum() / (S.sum() + 1e-12))


def adaptive_hf_ratio(path):
    return _band_ratio(path, 4000, 8000)


def adaptive_hf_db(path):
    """按实测 b4000_8000 反算 highshelf 增益。

    为什么必须**先上 boost 再量**：三个 boost（+7.5/+3/+3 dB）会把总能量
    分母撑大，从而间接压低所有频段的占比。拿裸片直接算会低估实际过量程度，
    补偿就偏浅。钳位 [-9.5, 0]：超过 -9.5dB 说明素材本身高频严重过量，
    再深就会像当年 -9.5 那版一样把 flat 打到 -6σ（发假）。
    """
    try:
        r = adaptive_hf_ratio(path)
    except Exception as e:
        sys.stderr.write("[warn] 自适应高频测量失败，回退 -1dB: %r\n" % e)
        return -1.0
    if r <= 0:
        return -1.0
    db = 10.0 * np.log10(HF_TARGET / r)
    return float(np.clip(db, HF_DB_MIN, HF_DB_MAX))

# 🔴 男声 EQ 改成**外部配置驱动**（2026-10-14）：
#    每次 eq_search6.py 搜出新参数都要改代码 + 重启，容易漏、也不好回滚。
#    现在只要把 {"eq": "..."} 写进 /mnt/tts/data/eq_male.json 即可生效，
#    文件不存在 / 解析失败就回退到上面的硬编码模板（行为不变）。
#    字符串里带 %g 的仍会按变调幅度填 highshelf 增益。
EQ_MALE_JSON = "/mnt/tts/data/eq_male.json"
# 女声默认不加 EQ（与 2026-10-14 之前的行为一致）；写了这个文件才生效。
EQ_FEMALE_JSON = "/mnt/tts/data/eq_female.json"


def _eq_json(path):
    """读 {"eq": "..."} 配置；不存在/解析失败都返回空串（= 行为不变）。"""
    try:
        if os.path.exists(path):
            d = json.load(open(path, encoding="utf-8"))
            if d.get("eq"):
                return str(d["eq"])
    except Exception as e:
        sys.stderr.write("[warn] 读 %s 失败，不加 EQ: %r\n" % (path, e))
    return ""


def eq_male_tmpl():
    s = _eq_json(EQ_MALE_JSON)
    return s or EQ_MALE_TMPL


def eq_female_tmpl():
    return _eq_json(EQ_FEMALE_JSON)
# EL 实测 F0 中位（librosa.pyin / 16k 口径，与 tools 里的诊断同口径，不可换算法）：
#   男（13 条 EL 案例）190.5~205.3 → 均值 197.7
#   女（5 条 SOLA 案例）271.0~282.1 → 均值 276.0
# 第31轮：男声目标 F0 197.7 -> 188.0。197.7 需要 R=1.094 上搬，
#   asetrate 把共振峰一起搬走，音色相似度 0.684 掉到「同人」阈值 0.75 以下；
#   188 只需 R≈1.05，音色 0.797，Σz² 37.4 仍落进 EL 自身分布(17~240)。
DEFAULT_F0 = {"f": 276.0, "m": 188.0}
EQ_MALE = "highshelf=f=4000:g=-4,lowpass=f=14000"


def f0_median(y, sr, win=60.0):
    """F0 中位。必须与 EL 基准值同口径：pyin / 16k / fmin60 fmax400 / **前 60s**。
    🔴 2026-10-14 踩坑：这里取整篇、而 DEFAULT_F0 是前 60s 口径，
       全篇 F0 比开头高约 13Hz（185.1 vs 171.7），导致变调只做了 1.14 半音
       （应该 2.5 半音），音高根本没补够。跨口径数字害死人。"""
    n = int(win * sr)
    if len(y) > n:
        y = y[:n]
    try:
        import librosa
        f0, vo, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr,
                                 frame_length=1024, hop_length=256)
        m = (vo if vo is not None else np.zeros(len(f0), dtype=bool))
        m = m & np.isfinite(f0)
        v = f0[m]
        return float(np.median(v)) if len(v) > 50 else None
    except Exception as e:
        print("F0 测量失败（缺 librosa？）: %r" % (e,), flush=True)
        return None


def load16(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", t], check=True)
    y, sr = sf.read(t, dtype="float32")
    os.unlink(t)
    return y, sr


def syl_rates(y, sr):
    hop = int(0.01 * sr)
    n = len(y) // hop
    env = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean()
                            for i in range(n)]))
    env = np.convolve(env, np.ones(5) / 5, mode="same")
    thr = np.percentile(env, 30) * 2.2
    w = int(WIN * 100)
    out, peaks, voiced = [], 0, 0
    for i in range(n):
        if env[i] > thr:
            voiced += 1
        if (env[i] > thr and 0 < i < n - 1 and env[i] >= env[i - 1]
                and env[i] > env[i + 1]
                and env[i] - max(env[max(0, i - 8):i].min(),
                                 env[i + 1:i + 9].min() if i + 9 < n else 0)
                > thr * 0.35):
            peaks += 1
        if (i + 1) % w == 0:
            if voiced > 0:
                out.append(peaks / (voiced / 100.0))
            peaks, voiced = 0, 0
    # 🔴 短文本（< 60s）一个整窗都凑不出来，out 会是空的，
    #    下面 np.concatenate([]) 直接崩（P1 冒烟 172 字/20s 实测触发）。
    #    只在**一个都没有**时补这个不足一窗的尾巴 —— 长篇仍按整窗切，
    #    否则会给 470s 的成品多加一个 50s 的碎窗，改变已验证的节奏分布。
    if not out and voiced > 0:
        out.append(peaks / (voiced / 100.0))
    return out


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    ms = re.findall(r"I:\s+(-?[\d.]+)\s+LUFS", r.stderr)
    return float(ms[-1]) if ms else -24.0



def _headfix_cut(y, sr):
    """v3.5 文件头孤立段切除（deploy/patch_headfix53.py + patch_headfix53b.py）。

    缺陷形态：起音即碎片（<200ms 的 >-30dB 段紧贴文件头）+ >=250ms 间隙
    🔴 v3.5：间隙阈值 150->250ms（v3.4 会把「한 퍼센트」的实词「한」
      170ms+150ms 当碎片切掉，实测吃掉首词；13 样本上 250ms 是干净分界）
    + 持续语音。ASR 对照证明碎片是幻觉级垃圾，切掉首词完整、音色 0.9994、
    click 不变。词内断裂型由「碎片组恰好 1 组」排除。

    🔴 任何异常/判据不满足一律原样返回，绝不让整单失败。
    🔴 必须在 load16 之后、任何增益之前调用（阈值是原始 dBFS 口径）。
    """
    try:
        hop = max(1, int(0.01 * sr))
        n = len(y) // hop
        if n < 100:
            return y, {}
        v = 20 * np.log10(np.sqrt(
            (y[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-12)
        # 1) 持续语音起点：首个 >-25dB 且其后 300ms 均值 >-28dB
        si = None
        for i in range(len(v) - 30):
            if v[i] > -25 and v[i:i + 30].mean() > -28:
                si = i
                break
        if si is None:
            return y, {}
        # 2) 从 S 往前跳过 lead-in，找第一个 >=150ms 间隙（切点 = 间隙终点）
        i = si - 1
        while i >= 0:
            if v[i] > -30:
                j = i
                while j >= 0 and v[j] > -30:
                    j -= 1
                i = j
            else:
                j = i
                while j >= 0 and v[j] <= -30:
                    j -= 1
                if (i - j) * 0.01 >= 0.25:
                    break
                i = j
        if i < 0:
            return y, {}
        cut = (i + 1) * 0.01
        # 3) 头部区域 [0, 断口) 内 >-30dB 碎片组：恰好 1 组、<=350ms、起点 <0.30s
        groups = []
        k = 0
        while k <= i:
            if v[k] > -30:
                m = k
                while m <= i and v[m] > -30:
                    m += 1
                groups.append((k * 0.01, m * 0.01))
                k = m
            else:
                m = k
                while m <= i and v[m] <= -30:
                    m += 1
                k = m
        if len(groups) != 1:
            return y, {}
        if groups[0][1] - groups[0][0] > 0.20 or groups[0][0] > 0.30:
            return y, {}
        # 4) 安全闸：切点后 400ms 内恢复持续正常语音
        k_end = min(len(v), i + 1 + 40)
        if not (i + 1 < len(v) and v[i + 1:k_end].max() > -25):
            return y, {}
        ncut = int(round(cut * sr))
        if ncut <= 0 or ncut >= len(y) - int(0.5 * sr):
            return y, {}
        y2 = y[ncut:].copy()
        fade = min(len(y2), max(1, int(0.005 * sr)))
        y2[:fade] *= np.linspace(0.0, 1.0, fade)
        return y2, {"cut": round(cut, 2), "frag": groups[0],
                    "S": round(si * 0.01, 2)}
    except Exception:                                          # noqa: BLE001
        return y, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--gender", default="f", choices=["f", "m"])
    ap.add_argument("--el", default="")
    ap.add_argument("--target-rate", type=float, default=None)
    ap.add_argument("--target-cps", type=float, default=None,
                    help="按「字/秒」锚定（跨文案可比，推荐）。需配合 --chars")
    ap.add_argument("--chars", type=int, default=0,
                    help="raw 对应的正文字数，--target-cps 模式下必填")
    ap.add_argument("--lufs", type=float, default=None)
    ap.add_argument("--pitch", type=float, default=0.0,
                    help="手动频率比（不是半音）：>1 升调。留空则按 --target-f0 自动算")
    ap.add_argument("--target-f0", type=float, default=None,
                    help="目标 F0 中位(Hz)。默认男 197.7 / 女 276.0（EL 案例实测均值）")
    ap.add_argument("--pitch-win", type=float, default=15.0,
                    help="变调分窗长度(秒)。音高漂移发生在窗内，60s 太粗"
                         "（实测跨窗 F0 极差 42.9Hz vs EL 29.8Hz）。"
                         "速度窗仍用 WIN=60s，不受此参数影响。")
    ap.add_argument("--no-pitch", action="store_true",
                    help="跳过 F0 对齐（对照组用）")
    ap.add_argument("--eq-db", type=float, default=None,
                    help="覆盖男声 highshelf 增益(dB)。默认：变调时 -6.5，否则 -4")
    ap.add_argument("--eq", default="",
                    help="完整 EQ 滤镜串，覆盖默认模板（A/B 实验用）。"
                         "传 'anull' 表示不做 EQ，用来产出「裸片」供网格搜索")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--no-adaptive-hf", dest="adaptive_hf",
                    action="store_false", default=True,
                    help="关掉男声自适应高频档，回到固定 -1dB（对照/回退用）")
    a = ap.parse_args()

    g = a.gender
    target = a.target_rate if a.target_rate else DEFAULT_RATE[g]
    lufs_t = a.lufs if a.lufs is not None else DEFAULT_LUFS[g]

    y, sr = load16(a.raw)
    # 🔴 v3.4 头部孤立段切除（deploy/patch_headfix53.py）：必须在任何增益之前
    #    （阈值是原始 dBFS 口径）。切完把 a.raw 重指到切过的文件——否则下面
    #    detect_gaps_wav(a.raw) 的时间轴会整体晚一个切长，60s 窗边界全偏。
    if os.environ.get("TTS_HEADFIX", "1") != "0":
        y, _hx = _headfix_cut(y, sr)
        if _hx.get("cut"):
            _hd = tempfile.mkdtemp(prefix="headfix_")
            _hp = os.path.join(_hd, "cut.wav")
            sf.write(_hp, y, sr)
            a.raw = _hp
            print("头部孤立段切除 %.2fs（碎片 %.2f-%.2f，持续语音 %.2fs 起）"
                  % (_hx["cut"], _hx["frag"][0], _hx["frag"][1],
                     _hx.get("S") or -1), flush=True)
    dur = len(y) / sr
    rc = syl_rates(y, sr)

    if a.el and os.path.exists(a.el):
        ely, _ = load16(a.el)
        dur_el = len(ely) / sr
        ec = syl_rates(ely, sr)
        centers = [(j + 0.5) * WIN / dur_el for j in range(len(ec))]
        fs = []
        for i, r in enumerate(rc):
            frac = (i + 0.5) * WIN / dur
            er = float(np.interp(frac, centers, ec)) if ec else r
            fs.append(min(max(r / er if er > 0 else 1.0, 1.0), 1.30))
        mode = "EL逐窗对齐"
    elif a.target_cps and a.chars > 0:
        # 字/秒锚：先算全局该慢多少，再按各窗相对快慢分配，保留自然起伏
        our_cps = a.chars / dur
        f = our_cps / a.target_cps
        mean_r = sum(rc) / len(rc) if rc else 1.0
        fs = [min(max(f * (r / mean_r) if r > 0 else f, 1.0), 1.35) for r in rc]
        mode = (f"字/秒锚 {a.target_cps}（当前 {our_cps:.2f} 字/秒，"
                f"全局需慢 {f:.3f}×）")
    else:
        fs = [min(max(r / target if target > 0 else 1.0, 1.0), 1.30) for r in rc]
        mode = f"目标速率 {target}"

    fs = [(fs[i - 1] + 2 * fs[i] + fs[i + 1]) / 4 if 0 < i < len(fs) - 1 else fs[i]
          for i in range(len(fs))]
    # 🔴 第31轮：空表兜底。`syl_rates` 只在 `voiced > 0` 时补尾巴，
    #    全静音时 `out` 仍为空 → `fs=[]` → 下面 `segs` 也为空 →
    #    `np.concatenate([])` 崩溃，整单退回裸音频。
    if not fs:
        fs = [1.0]
        print("[warn] 速率窗口为空（素材近乎全静音？），按 1.0 处理", flush=True)
    print(f"模式: {mode}")
    print("raw速率:", [round(x, 2) for x in rc])
    print("窗口系数:", [round(x, 3) for x in fs], flush=True)

    graw, _ = detect_gaps_wav(a.raw)
    bounds = [0.0]
    for i in range(1, len(fs)):
        want = i * WIN
        cand = [gg for gg in graw if abs(gg[0] - want) <= 10.0 and gg[1] >= 0.08]
        b = min(cand, key=lambda gg: abs(gg[0] - want))[0] if cand else want
        bounds.append(max(b, bounds[-1] + 1.0))
    bounds.append(dur)

    tmpd = tempfile.mkdtemp()
    segs = []
    for i in range(len(fs)):
        s0, s1 = bounds[i], bounds[i + 1]
        t1 = os.path.join(tmpd, "in.wav")
        t2 = os.path.join(tmpd, "out.wav")
        sf.write(t1, y[int(s0 * sr):int(s1 * sr)], sr)
        # 🔴 第30轮：变速**不再单独走一遍 atempo**。
        #    变调 R<1 本身就把时长拉长 1/R（R=0.92 → 1.087×），
        #    而语速锚 fs 也是要放慢 —— 两步 WSOLA 方向相反、几乎互相抵消，
        #    等于把语音白揉两遍（现役平坦度保持率 83.9%，融合后 88.4%）。
        #    融合后单次 atempo = 1/(R·fs) ≈ 0.988，几乎不动（实测 98.6% 保持）。
        #    这里保留原始片段，fs 折进变调步的那一次 atempo（见下方 FUSE）。
        # （ratios 在下面才算，这里不能引用；反正两步都不需要拉伸）
        shutil.copyfile(t1, t2)
        s2, _ = sf.read(t2, dtype="float32")
        segs.append(s2)
    buf = np.concatenate(segs)
    print("分段切窗后 %.1fs（变速已折进变调步，此处不再单独拉伸）"
          % (len(buf) / sr), flush=True)

    # ---- ② 变调：**分窗**校正，不再用全局单一比例 ------------------------
    # 🔴 2026-10-14 实测（deploy/f0_drift.py）推翻了"全局变调"的写法：
    #     VoxCPM2 的 F0 在长篇里会**往上漂**，EL 却是稳的：
    #       Vox raw   169.7 → 174.7 → 176.7 → 190.5 → 190.5 → 193.8 → 184.0 → 188.3（+11%）
    #       EL 8197   198.3 → 203.0 → 193.8 → 188.3 → 207.7 → 199.5 → 203.0 → 192.7（平）
    #     而我们按「前 60s」定一个全局比例 1.213，于是：
    #       base_p    196.1 → 205.3 → **222.6 → 212.6 → 233.2 → 225.2 → 229.2 → 226.5**
    #     开头正好 197.7，后半段却比 EL 高 15%（极差 37Hz vs EL 19Hz）——
    #     听感就是"发尖、发假、像捏着鼻子"，比 60-150Hz 那点差距重要得多。
    #     这也顺带解释了"感冒音"：变调把共振峰一起搬高，后半段搬得最多。
    #   ✅ 正确做法：每个窗按**自己的** F0 算比例，把整条 F0 曲线压平到 EL 水平。
    tf = a.target_f0 if a.target_f0 else DEFAULT_F0[g]

    # 🔴 第31轮：变调窗**单独细化**（--pitch-win，默认 15s）。
    #    融合模式下 buf 就是原始 y（变速已折进下面的 atempo，此处未拉伸），
    #    所以可以直接用 graw（原始停顿位置）来切子窗。
    PW = max(5.0, float(getattr(a, "pitch_win", 0) or WIN))
    _pw = [0.0]
    for i in range(1, int(dur // PW) + 1):
        want = i * PW
        cand = [gg for gg in graw if abs(gg[0] - want) <= 8.0 and gg[1] >= 0.08]
        b = min(cand, key=lambda gg: abs(gg[0] - want))[0] if cand else want
        _pw.append(max(b, _pw[-1] + 1.0))
    if _pw[-1] < dur - 0.05:
        _pw.append(dur)
    if len(_pw) < 2:
        _pw = [0.0, dur]
    sub_segs, fs_sub = [], []
    for i in range(len(_pw) - 1):
        s0, s1 = _pw[i], _pw[i + 1]
        if s1 - s0 < 0.5:
            continue
        sub_segs.append(buf[int(s0 * sr):int(s1 * sr)])
        # 语速系数按**原始时间**的中心点映射到 60s 的 fs 窗
        j = min(int(((s0 + s1) / 2.0) // WIN), len(fs) - 1)
        fs_sub.append(fs[max(0, j)])
    print("变调子窗 %d 个（每窗 %.0fs，速度窗仍 %.0fs）"
          % (len(sub_segs), PW, WIN), flush=True)

    if a.no_pitch:
        ratios = [1.0] * len(sub_segs)
    elif a.pitch:
        ratios = [a.pitch] * len(sub_segs)
    else:
        ratios = []
        for s2 in sub_segs:
            f0w = f0_median(s2, sr, win=1e9)   # 该窗**全段**，不是前 60s
            ratios.append(min(max(tf / f0w, 0.90), 1.30)
                          if f0w and f0w > 0 else 1.0)
        # 相邻窗之间 1:2:1 平滑，避免比例跳变在拼接点听出来
        if len(ratios) > 2:
            ratios = ([ratios[0]]
                      + [(ratios[i - 1] + 2 * ratios[i] + ratios[i + 1]) / 4
                         for i in range(1, len(ratios) - 1)]
                      + [ratios[-1]])
        print("分窗 F0: " + ", ".join(
            "%.0f" % x for x in
            [(f0_median(s2, sr, win=1e9) or 0) for s2 in sub_segs]), flush=True)
        print("分窗变调: " + ", ".join("%.3f" % r for r in ratios)
              + " （目标 %.1f Hz）" % tf, flush=True)
    # 🔴 第30轮 FUSE：不管变调比例是不是 1，都要走这一步 ——
    #    因为语速 fs 也折进了这里的 atempo。R==1 且 fs==1 才真正跳过。
    if True:
        new_segs = []
        for i, s2 in enumerate(sub_segs):
            R = ratios[i]
            if abs(R - 1.0) < 1e-6 and abs(fs_sub[i] - 1.0) < 1e-6:
                new_segs.append(s2)       # 既不改速也不变调，原样保留
                continue
            t1 = os.path.join(tmpd, "p%d_in.wav" % i)
            t2 = os.path.join(tmpd, "p%d_out.wav" % i)
            sf.write(t1, s2, sr)
            # 🔴 变调用 asetrate 三步，**不用 rubberband**：
            #    rubberband 会把频谱平坦度压掉（0.050 → 0.035，而 EL 是 0.042），
            #    声音变"过净"、发假。asetrate 只是重采样，波形结构保得住
            #    （flat 0.049、发声段数 110 = EL 110、段长 0.291 vs EL 0.277）。
            #    🔴 三步缺一不可：asetrate=sr*R（音高×R、时长/R）
            #      → aresample=sr（点数 N→N/R，时长仍 /R）→ atempo=1/R 补回时长。
            #      漏了 atempo → 成品 440.9s 而应为 534.8s（短 17.6%）。
            # 🔴 FUSE：把语速系数 fs[i] 折进这一次 atempo。
            #    asetrate(R) → 时长 orig/R；atempo(1/(R·fs)) → orig/R × R·fs
            #    = orig·fs ✓   音高 ×R ✓   与现役完全等价，但只揉一遍。
            _fuse_t = (1.0 / (R * fs_sub[i]) if fs_sub[i] > 0
                       else 1.0 / R)
            _fuse_t = min(max(_fuse_t, 0.5), 2.0)   # atempo 合法区间
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", t1, "-af",
                            "asetrate=%d,aresample=%d,atempo=%.6f"
                            % (int(round(sr * R)), sr, _fuse_t),
                            t2], check=True)
            s3, _ = sf.read(t2, dtype="float32")
            new_segs.append(s3)
        buf = np.concatenate(new_segs)
    t3 = os.path.join(tmpd, "st.wav")
    sf.write(t3, buf, sr)
    print("变调后 %.1fs" % (len(buf) / sr), flush=True)

    if a.no_clean:
        cur = t3
    else:
        gs, _ = detect_gaps_wav(t3)
        buf, n_clean, floors = clean_gaps(buf, sr, gs, min_dur=0.08, w_ms=5)
        fl = 20 * np.log10(np.median(floors) + 1e-9) if floors else float("nan")
        print("停顿清洗 %d 处（合成tone，底噪中位 %.1f dBFS，长度不变）"
              % (n_clean, fl), flush=True)
        t4 = os.path.join(tmpd, "clean.wav")
        sf.write(t4, buf, sr)
        verify_pauses(t4, "清洗后")
        cur = t4

#    🔴 变调后高频需要额外回压（默认 -2.5dB，见 EQ_MALE_DB 注释）。
    # 变调后频谱被整体搬高，高频档要加深。现在变调是**分窗**的，
    # 所以按**最大**比例判断（只要有一段被明显抬过，高频就得压）。
    _pmax = max(ratios) if ratios else 1.0
    eq_db = a.eq_db if a.eq_db is not None else -1.0
    _t = eq_male_tmpl()
    eq_m = a.eq if a.eq else (_t % eq_db if "%g" in _t else _t)
    # 🔴 女声此前**完全不加 EQ**（这里原来写死 `if g == "m" else ""`）。
    #    2026-10-14 实测女声偏离比男声更大（Σz²=86.2 vs 男声 19.2），
    #    而且方向相反：男声太干净(flat -3.51σ)，女声**太噪(flat +5.26σ)**。
    #    现在也走配置驱动：写了 /mnt/tts/data/eq_female.json 才生效，
    #    没写就是空串 —— 与改之前的行为完全一致，可安全上线。
    # 🔴 男声高频档改成**自适应**（2026-10-14）：
    #    固定 -1dB 在两篇文案上分别是 b4000_8000 = -1.67σ(8197) 和 +3.73σ(cf9)，
    #    差 5.4σ —— 一篇压过头、一篇没压够，固定值不可能同时满足。
    #    做法：先施加三个 boost（它们会把总能量分母撑大，间接压低该频段比值），
    #    **然后**实测 b4000_8000，按 10*log10(EL目标/实测) 反算 highshelf 增益。
    #    ⚠️ 顺序不能反：拿裸片直接算会因为分母没被撑大而得出过浅的补偿。
    eq_f = "" if g == "m" else eq_female_tmpl()
    if g == "m":
        # 🔴 只有「内置模板里带 %g 高频档」时才走自适应：
        #    传了 --eq、或配置里是不带 %g 的定串时，一律保持原行为，
        #    不能拿 BOOSTS 把用户指定的 EQ 顶掉。
        # 🔴 另：非自适应分支**必须**做 %g 代入 —— 我第一版重构时漏了这步，
        #    结果滤镜串里带着字面量 `g=%g` 直接喂给 ffmpeg，退出码 234。
        _t = eq_male_tmpl()
        _plain = a.eq if a.eq else (_t % eq_db if "%g" in _t else _t)
        _adaptive = (not a.eq) and a.adaptive_hf and ("%g" in _t)
        _eq = EQ_MALE_BOOSTS if _adaptive else _plain
    else:
        _adaptive = False
        _eq = eq_f
    gain = round(lufs_t - lufs(cur), 1)
    # 🔴 二次校正：EQ 会改变响度，按 EQ 前的响度算出来的增益是错的
    #   （实测男声目标 -25.0 结果落到 -25.4）。所以先粗调、再实测补一次。
    if _adaptive:
        # 第一段：只上 boost -> 实测 b4000_8000 -> 反算 highshelf 增益
        t5 = os.path.join(tmpd, "boosted.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", cur,
                        "-af", _eq, t5], check=True)
        hf_db = adaptive_hf_db(t5)
        print("自适应高频：boost 后 b4000_8000=%.5f (目标 %.5f) -> highshelf %+.2f dB"
              % (adaptive_hf_ratio(t5), HF_TARGET, hf_db), flush=True)
        # 🔴 250Hz 衰减放在**自适应之后**：不影响 adaptive_hf_db(t5) 的输入
        # 🔴 第31轮：改成**迭代收敛**。highshelf 不是理想砖墙（4000Hz 处只有
        #    一半增益、往上才渐近到 g），所以一次反算到不了位 ——
        #    实测残差 +0.31~+0.35 dB，而这一项占了 Σz² 的 36%。
        #    做法：量 → 补 → 再量，最多 3 轮，单轮残差 <0.25dB 即收工。
        # 🔴 第31轮：250Hz **不参与**迭代，恢复写死的 MALE_250（−0.45dB）。
        #    试过把它也做成自适应（deploy/apply_dual31.py），同配置连跑两次
        #    （r33/r34）证明平坦度是配置导致的：0.0470 → 0.031（−33%）。
        #    机理：`equalizer=f=250:t=q:w=1.1` 是窄带，窄带衰减对
        #    「平坦度 = 几何均值/算术均值」是致命的 —— 几何均值对接近零的
        #    频点极敏感，陷波一深分子就塌，而分母几乎不动。
        #    历史同源教训：highshelf -9.5dB 把 flat 打到 -6σ（发假）；
        #    MALE_250 -1.5dB 过冲，Σz² 反而升到 44.9 / 33.7。
        _last = None
        for _it in range(3):
            af = (("highshelf=f=%d:g=%g," % (HF_FREQ, hf_db)) + MALE_250
                  + "volume=%gdB" % gain)
            _t = os.path.join(tmpd, "ad%d.wav" % _it)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", t5,
                            "-af", af, _t], check=True)
            _r = adaptive_hf_ratio(_t)
            _d = 0.0
            if _r > 0:
                _d = float(np.clip(10.0 * np.log10(HF_TARGET / _r),
                                   HF_DB_MIN - hf_db, HF_DB_MAX - hf_db))
            print("   迭代%d: b4000_8000=%.5f -> 补 %+.2f dB（累计 %+.2f）"
                  % (_it, _r, _d, hf_db + _d), flush=True)
            hf_db = float(np.clip(hf_db + _d, HF_DB_MIN, HF_DB_MAX))
            _last = _t
            if abs(_d) < 0.25:
                break
        shutil.move(_last, a.out)
    else:
        af = (_eq + "," if _eq else "") + "volume=%gdB" % gain
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", cur, "-af", af,
                        a.out], check=True)
    delta = round(lufs_t - lufs(a.out), 2)
    if abs(delta) > 0.05:
        t5 = os.path.join(tmpd, "loud2.wav")
        # os.replace 跨设备会炸（/tmp → /mnt），必须 shutil.move
        shutil.move(a.out, t5)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", t5, "-af",
                        "volume=%gdB" % delta, a.out], check=True)
        print("响度二次校正 %+.2fdB（EQ 改变了响度）" % delta, flush=True)

    # ---- ⑤ 数字零填充（v3.6，见文件头 ZEROFILL 段）----------------------
    # 🔴 位置两个都不能错：
    #   · 在响度校正**之后**：填充不改响度，但先填会让上面的二次校正白算一遍；
    #   · 在 libmp3lame **之前**（陷阱 #18：链尾 mp3 会重新定电平，真零段在
    #     mp3 里比特分配极少，解码后就是员工说的「滋滋电磁音」）。
    # 🔴 必须用 FLOAT 写回：ffmpeg 默认输出 pcm_s16le，16bit 下 -80dB 只剩约
    #    1.6 LSB，会被量化成方波状杂散，等于白填。下游 server.py 用
    #    sf.read(dtype="float32") 读，尾段 tmp_wav 也是 float，全链路无损。
    if ZEROFILL_ON:
        _zy, _zsr = load16(a.out)
        _z0 = int((np.abs(_zy) < 1e-9).sum())
        _zy2 = declick.apply(_zy, _zsr, fill_db=ZEROFILL_DB, enabled=True,
                             mode="floor", quiet_db=ZEROFILL_DB)
        sf.write(a.out, _zy2, _zsr, subtype="FLOAT")
        _z1 = int((np.abs(_zy2) < 1e-9).sum())
        print("数字零填充 %.0fdB（真零样本 %d -> %d，时长 %.3fs 不变）"
              % (ZEROFILL_DB, _z0, _z1, len(_zy2) / _zsr), flush=True)

    yv, srv = load16(a.out)
    fc = syl_rates(yv, srv)
    gs2, _ = detect_gaps_wav(a.out)
    mid = [d for _, d in gs2 if 0.3 <= d < 0.8]
    print("修复后速率:", [round(x, 2) for x in fc], flush=True)
    print("DONE %s %.1fs 停顿%d处(0.3-0.8s:%d处/%.2fs) LUFS=%.1f(目标 %.1f)"
          % (a.out, len(yv) / srv, len(gs2), len(mid), sum(mid), lufs(a.out), lufs_t),
          flush=True)


if __name__ == "__main__":
    main()
