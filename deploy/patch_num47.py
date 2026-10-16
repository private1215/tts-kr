# -*- coding: utf-8 -*-
"""patch_num47.py —— ASR 回检的**数字归一化**（C 级改动：不改音频、不重掷）。

问题
----
whisper-small 会把听到的韩语数字读法**规范化成阿拉伯数字**再输出，
于是「念对了」的段落 CER 也虚高：

    原文（送进 TTS 的读法）    ASR 输出            现状判定
    칠만오천팔백원              75,800원            ❌ 算成错字
    삼쩜이칠퍼센트              3.27%               ❌ 算成错字
    시월 일일 열두 시 이십이분   10월 1일 12시 22분   ❌ 算成错字

财经文本数字密集 → CER 被这些「假错」顶到 0.35 以上 → 触发换种子重掷
（`ASR_MAX_RETRY=2`，最多跑 3 遍）→ **同一条成品慢 3 倍**（单条 1500 字
从 147 秒变成 20+ 分钟）。

原理
----
把**两边**（原文 / ASR 假设）的阿拉伯数字都转成韩语读法，统一口径。
因为 TTS 前端 `konum`（`TTS_KO_NUM=1`）送进模型的就是韩语读法，
所以转到韩语读法 = 与模型实际朗读的内容对齐。

    normalize("칠만오천팔백원")            -> 칠만오천팔백원   （无阿拉伯数字，不变）
    normalize("75,800원")                -> 칠만오천팔백원   （转后一致 ✅）

🔴 边界：只转**带单位**的数字。裸数字（股票代码 005930、年份代号）各有读法，
规则化反而会制造新的错配，保持原样。

🔴 不动 `konum.RULES`：那是 TTS 前端，改了会影响真实发音。
   这里是 asr_check 内部独立的、更全的规则集（多了 시/분/초/년/억/조）。

开关：`TTS_ASR_NUM_NORM=0` 可一键关掉，回到旧口径做对照。
"""
import io
import os
import re
import sys

PATH = "/mnt/tts/projects/qwen3-tts/app/asr_check.py"
BAK = PATH + ".bak-before-num47"

OLD_NORMALIZE = '''def normalize(text: str) -> str:
    """归一化：去空白与标点，只留实质字符。"""
    return "".join(_KEEP.findall(text or ""))
'''

NEW_NORMALIZE = '''def normalize(text: str) -> str:
    """归一化：去空白与标点，只留实质字符。

    🔴 2026-10-16 v1.2：归一化**之前**先把带单位的阿拉伯数字转成韩语读法。
    理由：whisper-small 会把听到的韩语数字规范化成阿拉伯数字输出
    （칠만오천팔백원 -> 75,800원），不处理就全是「念对了却算错字」的假错，
    把财经文本的 CER 顶过重掷阈值，导致同一条成品多跑 2 遍（慢 3 倍）。
    详见 deploy/patch_num47.py 文件头。
    """
    return "".join(_KEEP.findall(_num_to_ko(text or "")))
'''


# ---- 下面这一整段是要插进 asr_check.py 的数字转换实现 --------------------
NUM_BLOCK = '''
# ---------------------------------------------------------------------------
# 数字归一化（只服务于 ASR 回检比对，不影响 TTS 前端发音）
# ---------------------------------------------------------------------------
try:
    # 🔴 生产现场是**扁平导入**：systemd 的 WorkingDirectory=/mnt/tts/projects/qwen3-tts/app，
    #    server.py 写的是 `import asr_check`，此时 `app` 不是一个可导入的包。
    #    写成 `from app import konum` 会让服务启动即 ModuleNotFoundError 退出
    #    （2026-10-16 实测：连崩 7 次，restart counter 打满）。这里必须扁平导入。
    import konum as _konum
except ImportError:  # tools/ 下的脚本走包导入，两条路都留着
    from app import konum as _konum

_NUM_NORM = os.environ.get("TTS_ASR_NUM_NORM", "1").strip().lower() not in (
    "0", "false", "no")


# 🔴 比 konum.RULES 多出来的单位：년 / 억 / 조（TTS 前端没覆盖，但 ASR 侧常见）。
# 🔴🔴 **刻意不转 시 / 분 / 초（时间）**：韩语钟点读法不确定 —— 12시 既可读
#    열두시（固有数词）也可读 십이시（汉字数词），而 ko_number() 只会给后者。
#    实测单测：`시월 일일 열두 시 이십이분` vs ASR `10월 1일 12시 22분`
#    转后是 십이시 vs 열두시 —— **把原本 0 处错配变成 2 处**。
#    转了可能制造新错，不转至少维持现状；财经文本的假错主力是 원/퍼센트/만/주，
#    时间占比很小。等有真实 ASR 证据证明"总是读汉字数词"再加。
#    顺序重要：先复合（월..일），再百分号，再 만，最后通用单位。
_ASR_NUM_RULES = [
    (re.compile(r"(\\d{1,2})\\s*월\\s*(\\d{1,2})\\s*일"), _konum._date),
    (re.compile(r"(\\d[\\d,]*(?:\\.\\d+)?)\\s*%"), _konum._pct),
    (re.compile(r"(\\d[\\d,]*(?:\\.\\d+)?)\\s*(퍼센트|퍼센|프로)"), _konum._pct),
    (re.compile(r"(\\d[\\d,]*)\\s*만(?=\\s|주|원|개|명|$)"), _konum._man),
    # 🔴🔴 **刻意不含 개 / 명 / 배 / 번 / 살**：这些量词韩语用**固有数词**
    #   （3개 = 세 개，不是 삼 개；3명 = 세 명），而 ko_number() 只会给汉字数词
    #   （삼）。把它们写进来 = 把 0 处错配变成 1 处，**净制造新错**。
    #   保留的都是确定读汉字数词的：원 / 주 / 달러 / 포인트 / 년 / 월 / 일 / 억 / 조
    (re.compile(r"(\\d[\\d,]*(?:\\.\\d+)?)\\s*"
                r"(원|주|달러|포인트|년|월|일|억|조)"),
     _konum._money),
]


def _num_to_ko(text: str) -> str:
    """把带单位的阿拉伯数字转成韩语读法（与 TTS 前端送进模型的读法对齐）。

    裸数字（股票代码、年份代号）不动 —— 它们各有读法，规则化会制造新错配。
    """
    if not _NUM_NORM or not text:
        return text
    try:
        t = text
        for pat, fn in _ASR_NUM_RULES:
            t = pat.sub(fn, t)
        return t
    except Exception as e:  # 归一化失败绝不能拖垮回检
        log.warning("数字归一化失败，回退到原文本: %s", e)
        return text

'''


def main():
    src = io.open(PATH, encoding="utf-8").read()

    if "_num_to_ko" in src:
        print("已打过补丁，跳过")
        return 0
    if OLD_NORMALIZE not in src:
        print("❌ 没找到目标 normalize 定义，文件可能与预期不同")
        return 1

    if not os.path.exists(BAK):
        io.open(BAK, "w", encoding="utf-8").write(src)
        print("备份 ->", BAK)

    # 1) 换 normalize 实现
    out = src.replace(OLD_NORMALIZE, NEW_NORMALIZE, 1)

    # 2) 插入数字转换实现：放在 normalize 定义之前
    anchor = "def normalize(text: str) -> str:"
    i = out.index(anchor)
    out = out[:i] + NUM_BLOCK.lstrip("\n") + out[i:]

    io.open(PATH, "w", encoding="utf-8").write(out)
    print("✅ 已写入", PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
