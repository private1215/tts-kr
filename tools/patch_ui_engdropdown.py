# -*- coding: utf-8 -*-
"""patch_ui_engdropdown.py — 修音色库引擎下拉（10-18 事故防护）。

背景：10-18 员工在音色库删模版时，9 个音色的 engine 被误改成 cosyvoice3。
肇因：引擎下拉只有 qwen/cosyvoice3 两个选项（omnivoice/gptsovits 时代的音色
显示为"未选中"），紧挨删除键，onchange 立即保存无确认。
修法：补全 5 个引擎选项 + 正确选中 + 真改前确认（说清新旧值）。
"""
import io
import shutil
import time

T = "/mnt/tts/projects/qwen3-tts/app/ui/index.html"
src = io.open(T, encoding="utf-8").read()
n = 0

OLD_SEL = """        <select class="veng" style="flex:1" title="${t('engineHint')}">
          <option value="qwen"${v.engine === 'cosyvoice3' ? '' : ' selected'}>${t('engQwen')}</option>
          <option value="cosyvoice3"${v.engine === 'cosyvoice3' ? ' selected' : ''}>${t('engCosy')}</option>
        </select>"""
NEW_SEL = """        <select class="veng" style="flex:1" title="${t('engineHint')}" data-cur="${esc(v.engine || '')}">
          <option value="qwen"${v.engine === 'qwen' ? ' selected' : ''}>${t('engQwen')}</option>
          <option value="cosyvoice3"${v.engine === 'cosyvoice3' ? ' selected' : ''}>${t('engCosy')}</option>
          <option value="omnivoice"${v.engine === 'omnivoice' ? ' selected' : ''}>${t('engOmni')}</option>
          <option value="gptsovits"${v.engine === 'gptsovits' ? ' selected' : ''}>${t('engGs')}</option>
          <option value="voxcpm"${v.engine === 'voxcpm' ? ' selected' : ''}>${t('engVox')}</option>
        </select>"""
assert OLD_SEL in src, "select block not found"
src = src.replace(OLD_SEL, NEW_SEL, 1)
n += 1

OLD_CH = """    const es = item.querySelector('.veng');
    if(es) es.onchange = async () => {
      const val = es.value;
      try{"""
NEW_CH = """    const es = item.querySelector('.veng');
    if(es) es.onchange = async () => {
      const val = es.value;
      const cur = es.dataset.cur || '';
      // 🔴 2026-10-19：引擎下拉紧挨删除键，误触一下就悄悄改写生产链路
      //   （10-18 事故：9 个音色被误改成 cosyvoice3，成品整批跑错引擎）。
      //   真改之前必须确认，并说清新旧值。
      if(val === cur) return;
      if(!confirm(t('confirmEngChange')(n, cur || '未设置', val))){
        es.value = cur;
        if(!cur) loadMeta();
        return;
      }
      try{"""
assert OLD_CH in src, "onchange block not found"
src = src.replace(OLD_CH, NEW_CH, 1)
n += 1

I18N = [
    ("    vEngine:'引擎', engQwen:'千问 Qwen（无情感）', engCosy:'CosyVoice3（有情感）', engSaved:'引擎已保存', engFail:'引擎保存失败',",
     "    vEngine:'引擎', engQwen:'千问 Qwen（无情感）', engCosy:'CosyVoice3（有情感）', engOmni:'OmniVoice（现役男女声）', engGs:'GPT-SoVITS（GS 微调）', engVox:'VoxCPM（已退役）', engSaved:'引擎已保存', engFail:'引擎保存失败', confirmEngChange:(n,a,b)=>`把音色「${n}」的引擎从「${a}」改成「${b}」？该音色的生产链路会随之改变。`,"),
    ("    vEngine:'Engine', engQwen:'Qwen (most accurate)', engCosy:'CosyVoice3 (expressive)', engSaved:'Engine saved', engFail:'Failed to save engine',",
     "    vEngine:'Engine', engQwen:'Qwen (most accurate)', engCosy:'CosyVoice3 (expressive)', engOmni:'OmniVoice (current M+F)', engGs:'GPT-SoVITS (fine-tuned)', engVox:'VoxCPM (retired)', engSaved:'Engine saved', engFail:'Failed to save engine', confirmEngChange:(n,a,b)=>`Change voice \"${n}\" engine from \"${a}\" to \"${b}\"? Its production pipeline will change.`,"),
    ("    vEngine:'엔진', engQwen:'Qwen (가장 정확)', engCosy:'CosyVoice3 (감정 표현)', engSaved:'엔진이 저장되었습니다', engFail:'엔진 저장 실패',",
     "    vEngine:'엔진', engQwen:'Qwen (가장 정확)', engCosy:'CosyVoice3 (감정 표현)', engOmni:'OmniVoice (현역 남녀 음성)', engGs:'GPT-SoVITS (파인튜닝)', engVox:'VoxCPM (퇴역)', engSaved:'엔진이 저장되었습니다', engFail:'엔진 저장 실패', confirmEngChange:(n,a,b)=>`음성 \"${n}\"의 엔진을 \"${a}\"에서 \"${b}\"(으)로 바꿀까요? 프로덕션 경로가 바뀝니다.`,"),
    ("    vEngine:'エンジン', engQwen:'Qwen（最も正確）', engCosy:'CosyVoice3（感情表現）', engSaved:'エンジンを保存しました', engFail:'エンジンの保存に失敗',",
     "    vEngine:'エンジン', engQwen:'Qwen（最も正確）', engCosy:'CosyVoice3（感情表現）', engOmni:'OmniVoice（現役の男女音声）', engGs:'GPT-SoVITS（ファインチューニング）', engVox:'VoxCPM（退役）', engSaved:'エンジンを保存しました', engFail:'エンジンの保存に失敗', confirmEngChange:(n,a,b)=>`音色「${n}」のエンジンを「${a}」から「${b}」に変更しますか？プロダクション経路が変わります。`,"),
]
for o, nw in I18N:
    assert o in src, "i18n line not found: " + o[:40]
    src = src.replace(o, nw, 1)
    n += 1

bak = T + ".bak-before-engdropdown-" + time.strftime("%Y%m%d_%H%M%S")
shutil.copy2(T, bak)
io.open(T, "w", encoding="utf-8").write(src)
print("OK %d patches, backup %s" % (n, bak))
