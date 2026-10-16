# -*- coding: utf-8 -*-
"""patch_lat48.py —— CER 新增「英文缩写 → 韩语字母读法」归一化（v1.3）。

## 实测动机（不是拍脑袋）
`tools/cerdiff48.py`（女声 3415 字全篇）dump 出的 latin 类假错：

    ref=«엠엘씨씨»  hyp=« MLCC»
    ref=«알앤디»    hyp=«R&D »
    ref=«에이아이»  hyp=«AI»

`tools/cernum48.py`（数字最密集且健康的真实 job `fa1412a941b4`，
15.49% 是数字）新口径下残余最大差异块：

    ref=«브이엠»  ASR=«vm»   ← 出现 5 次

→ 假错方向是**固定的**：参考侧已被写成韩语字母读法，whisper 的假设侧写回拉丁。
三个案例的韩语读法**正好等于标准字母名**，所以映射可由数据自证，不用猜：
    MLCC → 엠엘씨씨 / R&D → 알앤디 / AI → 에이아이 / vm → 브이엠

## 为什么这次敢做（v1.2 时判断「双向映射风险」是错误的保守）
v1.2 的数字归一化只敢动**带单位**的阿拉伯数字，是因为裸数字读法不定
（股票代码、年份代号各有读法）。但拉丁字母不同：
- 韩语语音里**不存在**"该念成别的"的歧义 —— A 就是 에이、M 就是 엠；
- 我们只替换 `[A-Za-z]+` 这种纯拉丁 token，真正的韩语语音里除了缩写
  根本不会出现拉丁串 → **误伤面几乎为零**；
- 而且 `_KEEP = [가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]` 会保留拉丁，
  不转就是实打实的错配（ref 브이엠 vs hyp vm，3 字符对 2 字符，全错）。

## 做法
在 `normalize()` 里、数字归一化之后加一道 `_lat_to_ko()`，
对**参考侧和假设侧同时**生效（`cer()` 两边都走 `normalize`）。
开关 `TTS_ASR_LAT_NORM=0` 可关。

## 已量化
- 女声全篇：latin 假错占原文 0.0064（CER 口径约 0.003）
- `fa1412a941b4`：브이엠/vm 出现 5 次，是残余 diff 里最大的非数字项
🔴 量级**远不足以跨过 0.35 重掷阈值** —— 这个补丁的目的是
   **让 CER 这把尺子更准**，不是提速。别指望它减少重掷。
"""
import io
import os
import sys

PATH = "/mnt/tts/projects/qwen3-tts/app/asr_check.py"
BAK = PATH + ".bak-before-lat48"

ANCHOR = "def normalize(text: str) -> str:"

FN = '''# ---- v1.3：英文缩写 → 韩语字母读法 --------------------------------------
# 详见 deploy/patch_lat48.py 文件头。实测假错方向固定为
# 「参考侧是韩语读法（엠엘씨씨 / 알앤디 / 에이아이 / 브이엠），
#   whisper 假设侧写回拉丁（MLCC / R&D / AI / vm）」。
# 标准字母名正好等于这三个案例的韩语读法 → 映射由数据自证，不是猜的。
_LATIN2KO = {
    "a": "에이", "b": "비", "c": "씨", "d": "디", "e": "이",
    "f": "에프", "g": "지", "h": "에이치", "i": "아이", "j": "제이",
    "k": "케이", "l": "엘", "m": "엠", "n": "엔", "o": "오",
    "p": "피", "q": "큐", "r": "알", "s": "에스", "t": "티",
    "u": "유", "v": "브이", "w": "더블유", "x": "엑스",
    "y": "와이", "z": "지",
}
_LATIN_TOKEN = re.compile(r"[A-Za-z]+(?:\\s*&\\s*[A-Za-z]+)*")
_LAT_NORM = os.environ.get("TTS_ASR_LAT_NORM", "1").strip().lower() not in (
    "0", "false", "no")


def _lat_to_ko(text: str) -> str:
    """把纯拉丁缩写 token 逐字母展开成韩语读法（MLCC -> 엠엘씨씨）。

    🔴 只动 `[A-Za-z]+`（可含 &），不碰韩语、不碰数字 ——
       真正的韩语语音里除缩写外不会出现拉丁串，误伤面几乎为零。
       `_KEEP` 会保留拉丁，不转就是实打实的错配。
    """
    if not _LAT_NORM or not text:
        return text

    def _one(m):
        out = []
        for ch in m.group(0).lower():
            if ch == "&":
                out.append("앤")
            elif ch in _LATIN2KO:
                out.append(_LATIN2KO[ch])
        return "".join(out) or m.group(0)

    try:
        return _LATIN_TOKEN.sub(_one, text)
    except Exception as e:  # 归一化失败绝不能拖垮回检
        log.warning("拉丁归一化失败，回退到原文本: %s", e)
        return text


'''

OLD_RET = '''    return "".join(_KEEP.findall(_num_to_ko(text or "")))'''
NEW_RET = '''    return "".join(_KEEP.findall(_lat_to_ko(_num_to_ko(text or ""))))'''


def main():
    src = io.open(PATH, encoding="utf-8").read()
    if "_lat_to_ko" in src:
        print("已打过补丁，跳过")
        return 0
    if ANCHOR not in src:
        print("❌ 找不到 normalize 锚点")
        return 1
    if OLD_RET not in src:
        print("❌ 找不到 normalize 的 return（可能已改过）")
        return 1
    if not os.path.exists(BAK):
        io.open(BAK, "w", encoding="utf-8").write(src)
        print("备份 ->", BAK)
    out = src.replace(ANCHOR, FN + ANCHOR, 1)
    out = out.replace(OLD_RET, NEW_RET, 1)
    io.open(PATH, "w", encoding="utf-8").write(out)
    print("✅ 已写入", PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
