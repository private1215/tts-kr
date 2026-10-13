# -*- coding: utf-8 -*-
"""撤掉播放强制门控（20261013g 修订）：员工不是专业人士，只保留被动播放记录。
- 点击选项不再拦截；
- 进度显示改为中性「已听 X 秒」（不显示要求）；
- play 字段仍随票上报（离线参考，不强制）。
"""
FB = "/mnt/tts/projects/qwen3-tts/app/ui/feedback.html"

fb = open(FB, encoding="utf-8").read()

# 1) 撤销点击拦截
gate = """        var au = o.querySelector('audio');
        if (au){
          var need = needOf(au);
          if (playedOf(au.getAttribute('src')) < need){
            var m0 = document.getElementById('msg');
            m0.className = 'msg err';
            m0.textContent = '这段还没听到 ' + need + ' 秒，请先听完再选。';
            au.scrollIntoView({behavior:'smooth', block:'center'});
            try { au.play(); } catch (e2) {}
            return;
          }
        }
"""
assert gate in fb, "gate block missing"
fb = fb.replace(gate, "", 1)

# 2) 进度显示改中性
old_disp = """      sp.textContent = (done >= needOf(a)) ? '\\u2713 已听 ' + done + ' 秒'
                                          : '已听 ' + done + ' 秒 / 需 ' + needOf(a) + ' 秒';"""
assert old_disp in fb, "display missing"
new_disp = """      sp.textContent = '已听 ' + done + ' 秒';"""
fb = fb.replace(old_disp, new_disp, 1)

# 3) 初始文案
fb = fb.replace("还没播放</div>'", "已听 0 秒</div>'")

open(FB, "w", encoding="utf-8").write(fb)
print("gating removed, passive tracking kept")
