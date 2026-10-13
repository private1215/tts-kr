# -*- coding: utf-8 -*-
"""feedback.html + server.py 播放门控补丁（20261013g）。
- 页面：跟踪每段音频实际播放秒数；未听够（片段60%，10~25s）的选项点选时拦截；
  播放秒数随票 POST play 字段上报。
- 服务端：fb_vote 落盘 play 字段（离线可判断作答认真度）。
"""
import re, sys

FB = "/mnt/tts/projects/qwen3-tts/app/ui/feedback.html"
SRV = "/mnt/tts/projects/qwen3-tts/app/server.py"

fb = open(FB, encoding="utf-8").read()

# ---- 1. 播放跟踪工具（插在 PICK 恢复之后） ----
anchor1 = "try { PICK = JSON.parse(localStorage.getItem(LS) || '{}'); } catch (e) { PICK = {}; }\n"
assert anchor1 in fb, "anchor1 missing"
add1 = anchor1 + """
// ---- 播放时长跟踪（20261013g）：没听够的选项不能选；播放秒数随票上报 ----
var PLAY = {};
function playedOf(src){ return PLAY[src] || 0; }
function needOf(a){
  var d = (a && isFinite(a.duration)) ? a.duration : 35;
  return Math.round(Math.min(25, Math.max(10, d * 0.6)));
}
function trackAudio(a){
  if (a.dataset.tracked) return; a.dataset.tracked = '1';
  var src = a.getAttribute('src'); var last = null;
  a.addEventListener('timeupdate', function(){
    if (!a.paused && last != null){
      var d = a.currentTime - last;
      if (d > 0 && d < 1.5) PLAY[src] = playedOf(src) + d;
    }
    last = a.currentTime;
    var sp = a.parentNode.querySelector('.listened');
    if (sp){
      var done = Math.round(playedOf(src));
      sp.textContent = (done >= needOf(a)) ? '\\u2713 已听 ' + done + ' 秒'
                                          : '已听 ' + done + ' 秒 / 需 ' + needOf(a) + ' 秒';
    }
  });
}
"""
fb = fb.replace(anchor1, add1, 1)

# ---- 2. 选项音频后加进度显示 ----
anchor2 = """           +  (o.file ? '<audio controls preload="none" src="/fb-audio/' + esc(o.file) + '"></audio>'
                      : '<div class="noaudio">' + esc(o.text || '（无音频）') + '</div>')
           +  '</div>';"""
assert anchor2 in fb, "anchor2 missing"
add2 = """           +  (o.file ? '<audio controls preload="none" src="/fb-audio/' + esc(o.file) + '"></audio>'
                      + '<div class="listened" style="font-size:11.5px;color:var(--dim);margin-top:2px">还没播放</div>'
                      : '<div class="noaudio">' + esc(o.text || '（无音频）') + '</div>')
           +  '</div>';"""
fb = fb.replace(anchor2, add2, 1)

# ---- 3. ref 音频也显示进度 ----
anchor3 = """           +  '</div><audio controls preload="none" src="/fb-audio/'
           +  esc(it.ref.file) + '"></audio>'"""
assert anchor3 in fb, "anchor3 missing"
add3 = """           +  '</div><audio controls preload="none" src="/fb-audio/'
           +  esc(it.ref.file) + '"></audio>'
           +  '<div class="listened" style="font-size:11.5px;color:var(--dim);margin-top:2px">还没播放</div>'"""
fb = fb.replace(anchor3, add3, 1)

# ---- 4. render() 末尾：给所有 audio 挂跟踪 ----
anchor4 = """      });
    });
  });
  upd();
}"""
assert anchor4 in fb, "anchor4 missing"
add4 = """      });
    });
  });
  Array.prototype.forEach.call(box.querySelectorAll('audio'), function(a){ trackAudio(a); });
  upd();
}"""
fb = fb.replace(anchor4, add4, 1)

# ---- 5. 选项点击门控 ----
anchor5 = """      o.addEventListener('click', function(e){
        if (e.target.tagName === 'AUDIO') return;
        Array.prototype.forEach.call"""
assert anchor5 in fb, "anchor5 missing"
add5 = """      o.addEventListener('click', function(e){
        if (e.target.tagName === 'AUDIO') return;
        var au = o.querySelector('audio');
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
        Array.prototype.forEach.call"""
fb = fb.replace(anchor5, add5, 1)

# ---- 6. 提交带上 play ----
anchor6 = "body: JSON.stringify({picks: picks, free: document.getElementById('free').value.trim()})});"
assert anchor6 in fb, "anchor6 missing"
add6 = "body: JSON.stringify({picks: picks, free: document.getElementById('free').value.trim(), play: PLAY})});"
fb = fb.replace(anchor6, add6, 1)

open(FB, "w", encoding="utf-8").write(fb)
print("feedback.html patched")

# ================= server.py =================
srv = open(SRV, encoding="utf-8").read()
anchor7 = """    picks = body.get("picks") or {}
    free = (body.get("free") or "").strip()"""
assert anchor7 in srv, "anchor7 missing"
add7 = """    picks = body.get("picks") or {}
    free = (body.get("free") or "").strip()
    play = body.get("play") if isinstance(body.get("play"), dict) else {}"""
srv = srv.replace(anchor7, add7, 1)

anchor8 = """            "picks": {str(k): str(v) for k, v in picks.items()},
            "free": free[:4000],"""
assert anchor8 in srv, "anchor8 missing"
add8 = """            "picks": {str(k): str(v) for k, v in picks.items()},
            "free": free[:4000],
            "play": {str(k)[:150]: round(float(v), 1)
                     for k, v in list(play.items())[:200]
                     if isinstance(v, (int, float))},"""
srv = srv.replace(anchor8, add8, 1)
open(SRV, "w", encoding="utf-8").write(srv)
print("server.py patched")
