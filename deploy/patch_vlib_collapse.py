# -*- coding: utf-8 -*-
"""patch_vlib_collapse.py — 「已录入音色」模块默认折叠（2026-10-20）。

## 为什么
`#voiceLibraryCard`（已录入音色）虽然已经 `admin-only`（员工看不到），
但管理员页面上它整块摊开，把「标杆案例 → 生成」主流程挤到下面，排版难看。

## 做法
`<div class="card admin-only">` → `<details class="card admin-only">`，
标题行改成 `<summary>`，**默认关闭**（`details` 不带 `open` 就是折叠态）。

- 计数 `#vcount` 留在 summary 里：折叠时也能看到「共 N 条」，不用展开。
- 内容（`#vlist` / 提示语）原样保留在 details 内，JS 不用改（只在 DOM 里查询 id）。
- 补三条 CSS：去掉默认三角箭头、改成 ▸/▾、summary 不吃 `.card` 的外框样式。

## 回滚
`cp app/ui/index.html.bak-vlib-<时间戳> app/ui/index.html`（脚本会自动备份）。

用法：`venv/bin/python deploy/patch_vlib_collapse.py [--dry]`
"""
import io
import shutil
import time

T = "/mnt/tts/projects/qwen3-tts/app/ui/index.html"

OLD = """      <div class="card admin-only" id="voiceLibraryCard">
        <div class="row" style="margin-bottom:8px">
          <h2 style="margin:0" data-i18n="savedVoices">已录入音色</h2>
          <span class="hint" style="margin-left:auto" id="vcount"></span>
        </div>"""

NEW = """      <!-- 2026-10-20：默认折叠。整块摊开会把主流程挤下去，展开才看列表；
           计数留在 summary 里，折叠态也能看到「共 N 条」。 -->
      <details class="card admin-only" id="voiceLibraryCard">
        <summary id="vlibSum">
          <h2 style="margin:0" data-i18n="savedVoices">已录入音色</h2>
          <span class="hint" style="margin-left:auto" id="vcount"></span>
        </summary>"""

# 原 div 的收尾 </div> 要变成 </details>。用「紧跟其后的内容」定位，避免误伤别的 div。
OLD_END = """        <div class="hint" data-i18n="voiceListHint">音色由案例库自动生成（每条案例取最佳 15 秒并对齐文字稿），新增说话人请在顶部「标杆案例 → + 上传案例」上传。可试听、可查看文字稿。</div>
        <div id="vlist" style="margin-top:10px"></div>
      </div>"""

NEW_END = """        <div class="hint" data-i18n="voiceListHint">音色由案例库自动生成（每条案例取最佳 15 秒并对齐文字稿），新增说话人请在顶部「标杆案例 → + 上传案例」上传。可试听、可查看文字稿。</div>
        <div id="vlist" style="margin-top:10px"></div>
      </details>"""

CSS = """
/* ---- 已录入音色：可折叠卡片（2026-10-20）---- */
#voiceLibraryCard>summary{display:flex;align-items:center;gap:10px;cursor:pointer;
  list-style:none;margin:0 0 8px;padding:0;background:none;border:none;border-radius:0}
#voiceLibraryCard>summary::-webkit-details-marker{display:none}
#voiceLibraryCard>summary::after{content:'\\25B8';margin-left:8px;color:var(--dim);font-size:12px}
#voiceLibraryCard[open]>summary::after{content:'\\25BE'}
"""


def main(dry=False):
    src = io.open(T, encoding="utf-8").read()
    n = 0
    for old, new, tag in ((OLD, NEW, "开标签"), (OLD_END, NEW_END, "收尾标签")):
        assert old in src, "锚点未命中：%s" % tag
        src = src.replace(old, new, 1)
        n += 1
    assert "</style>" in src, "找不到 </style>"
    i = src.rfind("</style>")
    src = src[:i] + CSS + src[i:]
    n += 1
    if dry:
        print("dry-run：%d 处替换，未写入" % n)
        return 0
    bak = T + ".bak-vlib-" + time.strftime("%m%d%H%M")
    shutil.copy2(T, bak)
    io.open(T, "w", encoding="utf-8").write(src)
    print("✅ 已写入 %s（%d 处），备份 %s" % (T, n, bak))
    return 0


if __name__ == "__main__":
    import sys
    main(dry="--dry" in sys.argv)
