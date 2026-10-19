# -*- coding: utf-8 -*-
"""pipeline53.py —— 男声完整优化管线（一键串起来）。

## 🔴 顺序是实测改过的（2026-10-18）

原顺序 53b → 53c → EQ，在 LTAS 那步暴露了问题：
**firequalizer 是 FIR（线性相位），有振铃 / 时域涂抹** —— 它会在原本干净的
静音间隙里「抹」出 −44dB、5ms 的低能量碎片。实测 LTAS 后句末计数 2 → 8。

→ **频谱类处理必须在清理之前**，清理放最后才能把振铃碎片一起清掉；
  clickfix53 是样本级改动，放最后不干扰前面任何一步：
    ① ltasmatch53（LTAS 全频段对齐 EL，含中频 EQ 的效果）
  → ② tailclean53b（句末粘连尾巴）
  → ③ tailclean53c（句首紧贴碎片）
  → ④ clickfix53（精准样本级 click 修复，th=0.26）

实测（prod_male，564s，EL = d172b8c4efb3 压 mp3；**括号内是再压一次 192k mp3 后，
因为生产交付就是 mp3，编码会重新引入毛刺**）：
    原产         句末44 / 碎片21 / click **1.39×EL**
    旧顺序(清→EQ) 句末2  / 碎片7  / click 1.37
    +新顺序(LTAS→清) 句末0 / 碎片8 / click 1.30
    +clickfix(0.26)   句末0 / 碎片8 / click **0.09×EL**（压mp3后仍是 0.09）

⚠️ 女声**只跑第 ④ 步**：女声碎片已比 EL 少（10 vs 11）、中频已 1.01，
   且 LTAS 会让女声 click 恶化 19%。clickfix 对女声同样有效且零副作用
   （click 1.59×EL → 0.04×EL）。

用法：
    python3 pipeline53.py <in.wav> <el.wav> <out.wav>
"""
import os
import subprocess
import sys
import tempfile

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))


def run(args):
    print("$ %s" % " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main():
    if len(sys.argv) < 4:
        print("用法：pipeline53.py <in.wav> <el.wav> <out.wav>")
        return 1
    src, el, out = sys.argv[1:4]
    t1 = tempfile.mktemp(suffix=".wav")
    t2 = tempfile.mktemp(suffix=".wav")
    t3 = tempfile.mktemp(suffix=".wav")

    print("== 1/3 ltasmatch53：LTAS 全频段对齐 EL（必须在清理前）==",
          flush=True)
    run([PY, os.path.join(HERE, "ltasmatch53.py"), src, el, t1])

    print("== 2/3 tailclean53b：句末粘连尾巴（含振铃碎片）==", flush=True)
    run([PY, os.path.join(HERE, "tailclean53b.py"), t1, t2])

    print("== 3/4 tailclean53c：句首紧贴碎片 ==", flush=True)
    run([PY, os.path.join(HERE, "tailclean53c.py"), t2, t3])

    print("== 4/4 clickfix53：精准样本级 click 修复 (th=0.26) ==",
          flush=True)
    run([PY, os.path.join(HERE, "clickfix53.py"), t3, out,
         "--th", "0.26", "--rounds", "3"])

    print("\n成品 -> %s" % out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
