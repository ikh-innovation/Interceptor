#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = open(os.path.join(HERE, "mission_ui.html"), "rb").read()
FAILED = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


sys.path.insert(0, HERE)
import re
import mission as _M
_ui = {}
for mm in re.finditer(r'\{k: "(\w+)".*?c: \[([^\]]*)\], d: ("?\w+(?:\.\d+)?"?)', PAGE.decode()):
    ch = [x.strip().strip('"') for x in mm.group(2).split(",")]
    _ui[mm.group(1)] = (ch, mm.group(3).strip('"'))
_ms = {k: ([str(c) for c in ch], str(getattr(_M.Mission, a[0]))) for k, a, ch in _M.Mission.AUTO_SETTINGS}
_norm = lambda xs: [x if not x.replace(".", "").isdigit() else str(float(x)) for x in xs]
print("=== the page's AUTO settings vs mission.Mission.AUTO_SETTINGS ===")
check("same settings on the page as in the mission", sorted(_ui) == sorted(_ms), f"{sorted(_ui)} vs {sorted(_ms)}")
check("same choices and the same default for every one",
      all(_norm(_ui[k][0]) == _norm(_ms[k][0]) and _norm([_ui[k][1]]) == _norm([_ms[k][1]]) for k in _ms if k in _ui),
      str({k: (_ui.get(k), _ms[k]) for k in _ms if k in _ui and (_norm(_ui[k][0]) != _norm(_ms[k][0]) or _norm([_ui[k][1]]) != _norm([_ms[k][1]]))}))

if not shutil.which("firefox"):
    print("  [SKIP] firefox not installed -- browser test not run")
    sys.exit(0)

ST = dict(linked=True, armed=True, fc_mode="ALT_HOLD", target="DRONE", tilt_deg=30.0,
          detector="TensorRT", fps=28.0, follower=True,
          mission=dict(app="mission", phase="HOVER", engaged=False, ai_steering=False,
                       levels=dict(vert="SLOW", yaw="MEDIUM", push="SLOW", ai="MEDIUM", search="SLOW"),
                       speeds=dict(climb=1.0, descent=0.5, yaw=20, step=5, push=20, search=4), min_conf=0.2,
                       height_m=10.2, climb_mps=0.0, heading_deg=123,
                       vibe=dict(x=9, y=8, z=14, clip=0), det_conf=0.62, det_age_s=0.05,
                       target_fresh=True, target_az_deg=1.2, target_el_deg=-18.5),
          console_seq=2)
ST["mission"]["auto_params"] = {k: getattr(_M.Mission, a[0]) for k, a, _c in _M.Mission.AUTO_SETTINGS}
ST["mission"]["auto_defaults"] = dict(ST["mission"]["auto_params"])
ST["mission"]["level_defaults"] = dict(ST["mission"]["levels"])
ST["follower_default"] = True
CONSOLE = [[1, "12:00:00", "[MISSION] PILOT -> HOVER  (armed in ALT_HOLD)"],
           [2, "12:00:01", "[FC] override CONFIRMED"]]
CMDS, REP, SIMPOLLS = [], [], []
SLOW, DEAD = {"on": False}, {"on": False}

STEPS = [
    ("rep({step:'boot', phase:d.body.getAttribute('data-phase'), alt:q('#alt').textContent, card:!!q('#hdr')||!!q('#phase')||!!q('#hint'),"
     " hbL:q('#hbox').getBoundingClientRect().left, hbT:q('#hbox').getBoundingClientRect().top, hbBg:W.getComputedStyle(q('#hbox')).backgroundColor,"
     " dialBg:W.getComputedStyle(q('#dial')).backgroundColor, dialT:q('#dial').getBoundingClientRect().top,"
     " dialR:q('#dial').getBoundingClientRect().right, dialW:q('#dial').getBoundingClientRect().width,"
     " vbB:q('#vbox').getBoundingClientRect().bottom, vbL:q('#vbox').getBoundingClientRect().left,"
     " lockL:q('#b_lock').getBoundingClientRect().left, lockT:q('#b_lock').getBoundingClientRect().top,"
     " engL:q('#b_eng').getBoundingClientRect().left, engT:q('#b_eng').getBoundingClientRect().top,"
     " engR:q('#b_eng').getBoundingClientRect().right, padL:q('.pad').getBoundingClientRect().left,"
     " abort:!!q('#b_abort'), eng:q('#b_eng').className, engtxt:q('#engtxt').textContent,"
     " aimTop:q('#aim').style.top, badge:q('#badge').textContent, ai:q('#c_ai_k').textContent+' '+q('#c_ai').textContent,"
     " aicls:q('#c_ai').className, tgtbtn:!!q('#t_tgt'), engInVid:q('#vbox').contains(q('#b_eng')),"
     " engW:q('#b_eng').getBoundingClientRect().width,"
     " engRightOfPad:q('#b_eng').getBoundingClientRect().left>=q('.pad').getBoundingClientRect().right,"
     " engVsPad:Math.abs((q('#b_eng').getBoundingClientRect().top+q('#b_lock').getBoundingClientRect().bottom)/2"
     "-(q('.pad').getBoundingClientRect().top+q('.pad').getBoundingClientRect().bottom)/2),"
     " ticker:!!q('#ticker'), tools:!!q('.tools'), menu:!!q('#t_menu'),"
     " menuTL:q('#t_menu').getBoundingClientRect().left<40 && q('#t_menu').getBoundingClientRect().top<60,"
     " bodyBg:W.getComputedStyle(d.body).backgroundColor, btnBg:W.getComputedStyle(q('#b_up')).backgroundColor,"
     " grpC:Math.abs((q('.pad').getBoundingClientRect().left+q('#b_eng').getBoundingClientRect().right)/2-W.innerWidth/2),"
     " lockAbove:q('#b_lock').getBoundingClientRect().bottom<=q('#b_eng').getBoundingClientRect().top,"
     " lockW:q('#b_lock').getBoundingClientRect().width, lockBg:W.getComputedStyle(q('#b_lock')).backgroundColor,"
     " lockTxt:q('#b_lock').textContent, pills:d.querySelectorAll('.top .pill, #c_lock, #c_ctl').length,"
     " lockB:q('#b_lock').getBoundingClientRect().bottom, lockR:q('#b_lock').getBoundingClientRect().right,"
     " engR2:q('#b_eng').getBoundingClientRect().right,"
     " dialL:q('#dial').getBoundingClientRect().left, vbR:q('#vbox').getBoundingClientRect().right, dialH:q('#dial').getBoundingClientRect().height,"
     " dialOff:q('#dial').classList.contains('off'), vbL:q('#vbox').getBoundingClientRect().left,"
     " vbH:q('#vbox').getBoundingClientRect().height, vbW:q('#vbox').getBoundingClientRect().width,"
     " padC:Math.abs((q('.pad').getBoundingClientRect().left+q('.pad').getBoundingClientRect().right)/2-W.innerWidth/2)}); fetch('/reset');", 400),
    ("var cv=q('#tape'), vb=q('#vbox').getBoundingClientRect(), r=cv.getBoundingClientRect();"
     " window._ink=function(x0,x1){var c=cv.getContext('2d'), d=c.getImageData(Math.round(cv.width*x0),0,"
     "Math.max(1,Math.round(cv.width*(x1-x0))),cv.height).data, n=0; for(var i=3;i<d.length;i+=4) if(d[i]>100) n++; return n;};"
     " rep({step:'tape', top:r.top-vb.top, wdiff:Math.abs(r.width-vb.width), h:r.height, left:_ink(0,0.35), right:_ink(0.65,1)});"
     " fetch('/set?phase=HOVER&heading=352');", 900),
    ("rep({step:'tape352', left:_ink(0,0.35), right:_ink(0.65,1)}); fetch('/set?phase=HOVER&heading=123'); fetch('/reset');", 500),
    ("until(function(){return !q('#b_lock').disabled},3000).then(function(){ return fetch('/reset'); })"
     ".then(function(){ pe(q('#b_lock'),'pointerdown'); });", 3400),
    ("cmds().then(function(c){rep({step:'lock_on', cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})});"
     " fetch('/set?phase=HOVER&locked=1&lock_hdg=352');});", 800),
    ("until(function(){return q('#b_lock').classList.contains('on')},3000).then(function(){"
     " rep({step:'locked', cls:q('#b_lock').className, txt:q('#locktxt').textContent, phase:d.body.getAttribute('data-phase'),"
     "});"
     " fetch('/reset').then(function(){ pe(q('#b_lock'),'pointerdown'); }); });", 3600),
    ("cmds().then(function(c){rep({step:'lock_off', cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})});"
     " fetch('/set?phase=HOVER&locked=0').then(function(){ fetch('/reset'); });});", 400),
    ("until(function(){return !q('#b_lock').classList.contains('on') && !q('#b_yl').disabled},3000);", 3100),
    ("rep({step:'unlocked', cls:q('#b_lock').className, txt:q('#locktxt').textContent}); fetch('/reset');", 300),
    ("var dl=q('#dial'); pey(dl,'pointerdown',300);"
     " [1,2,3,4,5,6].forEach(function(k){ setTimeout(function(){ pey(dl,'pointermove',300-10*k); }, 80*k); });"
     " setTimeout(function(){ rep({step:'dragging', ask:q('#ask').classList.contains('on'), drag:dl.classList.contains('drag')});"
     " pey(dl,'pointerup',240); }, 600);", 900),
    ("cmds().then(function(c){ rep({step:'asked', ask:q('#ask').classList.contains('on'), deg:q('#ask_deg').textContent,"
     " sub:q('#ask_sub').textContent, cmds:c}); q('#ask_yes').click(); });", 500),
    ("cmds().then(function(c){rep({step:'ask_yes', ask:q('#ask').classList.contains('on'),"
     " cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})}); return fetch('/reset'); }).then(function(){"
     " var dl=q('#dial'); pey(dl,'pointerdown',100); setTimeout(function(){ pey(dl,'pointermove',140); },80);"
     " setTimeout(function(){ pey(dl,'pointermove',180); },160); setTimeout(function(){ pey(dl,'pointerup',180); },240); });", 700),
    ("rep({step:'asked_down', deg:q('#ask_deg').textContent}); q('#ask_no').click();"
     " setTimeout(function(){ cmds().then(function(c){ rep({step:'ask_no', ask:q('#ask').classList.contains('on'), cmds:c}); fetch('/reset'); }); }, 300);", 700),
    ("var dl=q('#dial'); pey(dl,'pointerdown',200); pey(dl,'pointerup',200);"
     " setTimeout(function(){ rep({step:'dial_tap', ask:q('#ask').classList.contains('on')}); }, 200);", 400),
    ("fetch('/set?phase=HOVER&armed=0').then(function(){ return until(function(){return !q('#t_sim').disabled},3000); })"
     ".then(function(){ fetch('/reset'); q('#t_menu').click(); q('#t_test').click(); setTimeout(function(){ rep({step:'sim_menu', dis:q('#t_sim').disabled,"
     " menu:q('#sh_menu').classList.contains('open'), test:q('#sh_test').classList.contains('open'),"
     " rows:[].map.call(d.querySelectorAll('#sh_menu .row'),function(x){return x.firstChild.firstChild.textContent}),"
     " indent:[].map.call(d.querySelectorAll('#sh_menu .row'),function(x){var rg=d.createRange(); rg.selectNodeContents(x.firstChild.firstChild);"
     " return Math.round(rg.getBoundingClientRect().left-x.getBoundingClientRect().left)})});"
     " q('#t_sim').click(); }, 200); });", 3400),
    ("until(function(){return d.body.classList.contains('sim')},3000).then(function(){ setTimeout(function(){"
     " var vb=q('#vbox').getBoundingClientRect(), gl=q('#s_gl').getBoundingClientRect(), gr=q('#s_gr').getBoundingClientRect(), x=q('#s_exit').getBoundingClientRect();"
     " rep({step:'sim_on', bar:W.getComputedStyle(q('#simctl')).display, img:W.getComputedStyle(q('#v')).visibility,"
     " inPic:q('#vbox').contains(q('#simctl')) && q('#vbox').contains(q('#s_exit')),"
     " wind:[gl.left-vb.left, gl.top-vb.top, gr.left-gl.right, gl.width], exit:[vb.right-x.right, x.top-vb.top, q('#s_exit').textContent],"
     " old:!!q('#simbar') || !!q('#s_wind') || !!q('#s_reset'), svg:!!q('#s_gl svg') && !!q('#s_gr svg'),"
     " sky:(function(){ var cv=q('#simv'), p=cv.getContext('2d').getImageData(Math.round(cv.width/2),Math.round(cv.height*0.2),1,1).data; return [p[0],p[1],p[2],p[3]]; })(),"
     " words:W.getComputedStyle(q('#badge')).display,"
     " canv:W.getComputedStyle(q('#simv')).display, sheet:q('#sh_test').classList.contains('open') || q('#sh_menu').classList.contains('open')});"
     " fetch('/reset').then(function(){ pe(q('#b_lock'),'pointerdown'); q('#s_gr').click(); }); }, 700); });", 1900),
    ("cmds().then(function(c){ rep({step:'sim_cmds', cmds:c}); q('#s_exit').click();"
     " setTimeout(function(){ rep({step:'sim_off', sim:d.body.classList.contains('sim'), bar:W.getComputedStyle(q('#simctl')).display, x:W.getComputedStyle(q('#s_exit')).display}); fetch('/reset'); }, 300); });", 900),
    ("fetch('/set?phase=HOVER&armed=1'); until(function(){return q('#t_sim').disabled},3000).then(function(){"
     " rep({step:'sim_armed', dis:q('#t_sim').disabled}); });", 3200),
    ("until(function(){return !q('#b_up').disabled},3000).then(function(){ return fetch('/reset'); })"
     ".then(function(){ pe(q('#b_up'),'pointerdown'); setTimeout(function(){"
     " rep({step:'held', cls:q('#b_up').className}); pe(q('#b_up'),'pointerup'); },1200); });", 4600),
    ("cmds().then(function(c){rep({step:'hold_up', n:c.filter(function(x){return x.indexOf('hold=up')>=0}).length,"
     " rel:c.some(function(x){return x.indexOf('release=')>=0}), pin:c.every(function(x){return x.indexOf('k=')<0})});"
     " fetch('/reset');});", 300),
    ("until(function(){return !q('#b_yl').disabled},3000).then(function(){ return fetch('/reset'); }).then(function(){"
     " pe(q('#b_yl'),'pointerdown'); setTimeout(function(){pe(q('#b_yl'),'pointerup');},120); });", 3300),
    ("cmds().then(function(c){rep({step:'tap', cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})}); fetch('/reset');});", 300),
    ("until(function(){return !q('#b_yr').disabled},3000).then(function(){ return fetch('/reset'); })"
     ".then(function(){ pe(q('#b_yr'),'pointerdown'); });", 3500),
    ("W.dispatchEvent(new W.Event('blur'));", 600),
    ("cmds().then(function(c){rep({step:'blur', n:c.filter(function(x){return x.indexOf('hold=yawr')>=0}).length,"
     " rel:c.some(function(x){return x.indexOf('release=')>=0})}); fetch('/reset');});", 700),
    ("cmds().then(function(c){rep({step:'after_blur', n:c.filter(function(x){return x.indexOf('hold=')>=0}).length}); fetch('/reset');});", 300),
    ("until(function(){return !q('#b_eng').disabled},3000).then(function(){ return fetch('/reset'); })"
     ".then(function(){ pe(q('#b_eng'),'pointerdown'); setTimeout(function(){pe(q('#b_eng'),'pointerup');},80); });", 3400),
    ("cmds().then(function(c){rep({step:'engage_tap', n:c.filter(function(x){return x.indexOf('engage=')>=0}).length}); fetch('/reset');});", 300),
    ("fetch('/set?phase=AI&engaged=1');", 100),
    ("until(function(){return q('#b_eng').classList.contains('go')},3000).then(function(){ rep({step:'ai', eng:q('#b_eng').className, engtxt:q('#engtxt').textContent, dis:q('#b_eng').disabled,"
     " up:q('#b_up').disabled}); fetch('/set?phase=HOVER&engaged=1'); });", 3200),
    ("until(function(){return d.body.getAttribute('data-phase')!=='AI'},3000);", 3100),
    ("rep({step:'latched', eng:q('#b_eng').className, dis:q('#b_eng').disabled, engtxt:q('#engtxt').textContent});"
     " fetch('/set?phase=HOVER&engaged=0');", 100),
    ("until(function(){return !q('#b_eng').classList.contains('go')},3000);", 3100),
    ("rep({step:'reset', eng:q('#b_eng').className, dis:q('#b_eng').disabled}); q('#t_menu').click(); q('#t_par').click();", 400),
    ("var sh=q('#sh_speed'), cs=W.getComputedStyle(sh); rep({step:'speeds', open:sh.classList.contains('open'),"
     " pos:cs.position, h:sh.getBoundingClientRect().height, winh:W.innerHeight,"
     " search:[].map.call(d.querySelectorAll('.seg[data-g=search] button'),function(x){return x.textContent}),"
     " searchOn:(q('.seg[data-g=search] button.on')||{}).textContent,"
     " fol:[].map.call(d.querySelectorAll('#folseg button'),function(x){return x.textContent}),"
     " folOn:(q('#folseg button.on')||{}).textContent,"
     " heads:[].map.call(d.querySelectorAll('#sh_speed .acch b'),function(x){return x.textContent}),"
     " nOpen:d.querySelectorAll('#sh_speed .ppane.on').length,"
     " homes:[q('#pp_semi').contains(q('.seg[data-g=search]')), q('#pp_auto').contains(q('#autoparams')) && q('#pp_auto').contains(q('#autoseg')),"
     " q('#pp_det').contains(q('#skyseg')) && q('#pp_det').contains(q('#folseg'))],"
     " sums:[q('#sum_semi').textContent, q('#sum_auto').textContent, q('#sum_det').textContent],"
     " acc:W.getComputedStyle(q('.seg[data-g=search] button.on')).backgroundColor, boxBg:W.getComputedStyle(q('#ptab_auto')).backgroundColor,"
     " unsaved:d.querySelectorAll('.unsaved.on').length,"
     " isdef:(q('.seg[data-g=vert] .isdef')||{getAttribute:function(){return null}}).getAttribute('data-lv'),"
     " ask:q('#defask').classList.contains('on')});"
     " q('#ptab_semi').click(); rep({step:'semiopen', semi:q('#pp_semi').classList.contains('on'), auto:q('#pp_auto').classList.contains('on'),"
     " det:q('#pp_det').classList.contains('on'), exp:q('#ptab_semi').getAttribute('aria-expanded')});"
     " q('.seg[data-g=vert] button[data-lv=FAST]').click();", 500),
    ("until(function(){return q('#defask').classList.contains('on')},3000).then(function(){ return cmds(); }).then(function(c){"
     " rep({step:'ask1', ask:q('#defask').classList.contains('on'), k:q('#defk').textContent, v:q('#defv').textContent,"
     " sub:q('#defsub').textContent, cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})}); q('#def_yes').click(); });", 3400),
    ("cmds().then(function(c){rep({step:'def1', ask:q('#defask').classList.contains('on'), cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})});"
     " fetch('/reset').then(function(){ q('.seg[data-g=search] button[data-lv=FAST]').click(); });});", 500),
    ("until(function(){return q('#defask').classList.contains('on')},3000).then(function(ok){"
     " rep({step:'ask2', ask:ok, v:q('#defv').textContent}); q('#def_no').click(); });", 3400),
    ("cmds().then(function(c){rep({step:'lvl', ask:q('#defask').classList.contains('on'), cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})});"
     " fetch('/reset').then(function(){ q('#folseg [data-fol=off]').click(); });});", 400),
    ("until(function(){return q('#defask').classList.contains('on')},3000).then(function(){ return cmds(); }).then(function(c){"
     " rep({step:'fol', ask:q('#defask').classList.contains('on'), k:q('#defk').textContent, v:q('#defv').textContent, cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})});"
     " q('#def_no').click(); return fetch('/set?phase=HOVER&lvl_vert=FAST'); }).then(function(){ return fetch('/reset'); })"
     ".then(function(){ return until(function(){return q('.unsaved[data-save=\"lvl:vert\"]').classList.contains('on')},3000); })"
     ".then(function(ok){ rep({step:'chip', on:ok, n:d.querySelectorAll('.unsaved.on').length}); q('.unsaved[data-save=\"lvl:vert\"]').click(); });", 3800),
    ("cmds().then(function(c){rep({step:'chipsave', cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})}); fetch('/set?phase=HOVER&lvl_vert=SLOW');"
     " q('#sh_speed [data-close]').click(); rep({step:'closed', open:q('#sh_speed').classList.contains('open')}); fetch('/reset');});", 400),
    ("fetch('/set?phase=HOVER&auto=0').then(function(){ return until(function(){return q('#b_auto').disabled},3000); })"
     ".then(function(){ var a=q('#b_auto').getBoundingClientRect(), l=q('#b_lock').getBoundingClientRect();"
     " rep({step:'auto_off', dis:q('#b_auto').disabled, above:a.bottom<=l.top, dl:Math.abs(a.left-l.left), w:a.width, h:a.height});"
     " fetch('/set?phase=HOVER&auto=1'); });", 3200),
    ("until(function(){return !q('#b_auto').disabled},3000).then(function(){ return fetch('/reset'); })"
     ".then(function(){ pe(q('#b_auto'),'pointerdown'); });", 3400),
    ("cmds().then(function(c){rep({step:'auto_tap', cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})});"
     " fetch('/set?phase=AUTO_HOLD&auto=1');});", 400),
    ("until(function(){return q('#b_auto').classList.contains('on')},3000).then(function(){"
     " rep({step:'auto_hold', cls:q('#b_auto').className, dis:q('#b_auto').disabled, txt:q('#autotxt').textContent,"
     " yl:q('#b_yl').disabled, yr:q('#b_yr').disabled, up:q('#b_up').disabled, lock:q('#b_lock').disabled, eng:q('#b_eng').disabled});"
     " fetch('/set?phase=AUTO_ALIGN&auto=1'); });", 3200),
    ("until(function(){return q('#autotxt').textContent.indexOf('LINING')>=0},3000).then(function(){"
     " rep({step:'auto_align', txt:q('#autotxt').textContent, yl:q('#b_yl').disabled});"
     " fetch('/set?phase=HOVER&auto=1'); });", 3200),
    ("until(function(){return !q('#b_auto').classList.contains('on')},3000).then(function(){ return fetch('/reset'); })"
     ".then(function(){ rep({step:'ai_btns', ai:[].map.call(d.querySelectorAll('.seg[data-g=ai] button'),function(x){return x.getAttribute('data-lv')}),"
     " push:[].map.call(d.querySelectorAll('.seg[data-g=push] button'),function(x){return x.getAttribute('data-lv')}),"
     " autoseg:[].map.call(d.querySelectorAll('#autoseg button'),function(x){return x.getAttribute('data-autoen')}),"
     " autoOn:(q('#autoseg button.on')||{getAttribute:function(){return null}}).getAttribute('data-autoen')});"
     " q('.seg[data-g=ai] button[data-lv=MAX]').click(); q('#autoseg [data-autoen=off]').click(); });", 3400),
    ("cmds().then(function(c){rep({step:'max_autoen', askMax:q('#defask').classList.contains('on'), cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})});"
     " q('#def_no').click(); fetch('/reset');});", 400),
    ("q('#t_menu').click(); q('#t_par').click(); setTimeout(function(){"
     " rep({step:'hub', heads:[].map.call(d.querySelectorAll('#sh_speed .acch b'),function(x){return x.textContent}),"
     " tag:q('#autotag').textContent, tagIn:q('#ptab_auto').contains(q('#autotag')), cats:d.querySelectorAll('.cat').length});"
     " q('#ptab_auto').click(); }, 300);", 700),
    ("rep({step:'autopage', auto:q('#pp_auto').classList.contains('on'), semi:q('#pp_semi').classList.contains('on'),"
     " exp:q('#ptab_auto').getAttribute('aria-expanded'),"
     " n:d.querySelectorAll('#pp_auto [data-ap]').length, secs:[].map.call(d.querySelectorAll('#pp_auto .pcard .sec'),function(x){return x.textContent}),"
     " on:[].map.call(d.querySelectorAll('#pp_auto [data-ap].on'),function(x){return x.getAttribute('data-ap')+'='+x.getAttribute('data-av')}),"
     " dflt:d.querySelectorAll('#pp_auto [data-ap].isdef').length, chips:d.querySelectorAll('#pp_auto .unsaved').length,"
     " autoenDef:(q('#autoseg .isdef')||{getAttribute:function(){return null}}).getAttribute('data-autoen'),"
     " skyDef:(q('#skyseg .isdef')||{getAttribute:function(){return null}}).getAttribute('data-sky')});"
     " q('#pp_auto [data-ap=height][data-av=\"15\"]').click(); q('#autoreset').click();", 600),
    ("cmds().then(function(c){rep({step:'autoset', ask:q('#defask').classList.contains('on'), cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})});"
     " fetch('/reset').then(function(){ q('#pp_auto [data-ap=height][data-av=\"20\"]').click(); });});", 400),
    ("until(function(){return q('#defask').classList.contains('on')},3000).then(function(ok){"
     " rep({step:'autoask', ask:ok, k:q('#defk').textContent, v:q('#defv').textContent, sub:q('#defsub').textContent}); q('#def_yes').click(); });", 3400),
    ("cmds().then(function(c){rep({step:'autodef', ask:q('#defask').classList.contains('on'), cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})});"
     " q('#ptab_semi').click(); rep({step:'autoback', semi:q('#pp_semi').classList.contains('on'), auto:q('#pp_auto').classList.contains('on')});"
     " q('#ptab_semi').click(); rep({step:'toggle', semi:q('#pp_semi').classList.contains('on'), exp:q('#ptab_semi').getAttribute('aria-expanded')});"
     " d.querySelectorAll('.sheet').forEach(function(x){x.classList.remove('open')}); fetch('/reset');});", 400),
    ("q('#t_log').click();", 400),
    ("rep({step:'console', lines:[].map.call(d.querySelectorAll('#log .ln'),function(l){return l.textContent})});"
     " var lg=q('#sh_log'); rep({step:'logsheet', open:lg.classList.contains('open'), pos:W.getComputedStyle(lg).position,"
     " h:lg.getBoundingClientRect().height, winh:W.innerHeight});"
     " q('#sh_log [data-close]').click(); rep({step:'logclosed', open:q('#sh_log').classList.contains('open')}); q('#sh_menu [data-close]').click();"
     " fetch('/set?phase=PUSH&engaged=1');", 100),
    ("until(function(){return d.body.getAttribute('data-phase')==='PUSH'},3000).then(function(){"
     " rep({step:'push', up:q('#b_up').disabled, eng:q('#b_eng').disabled, aimTop:q('#aim').style.top, lockdis:q('#b_lock').disabled});"
     " rep({step:'push_dial', off:q('#dial').classList.contains('off')}); var dl=q('#dial'); pey(dl,'pointerdown',300);"
     " pey(dl,'pointermove',200); pey(dl,'pointerup',200); setTimeout(function(){ rep({step:'push_dial2', ask:q('#ask').classList.contains('on')});"
     " fetch('/set?phase=HOVER&engaged=0'); fetch('/dead?1'); }, 200); });", 4800),
    ("rep({step:'dead', cover:q('#cover').classList.contains('on'), done:1});", 0),
]
HARNESS = ("""<!doctype html><html><body><iframe id=f src="/" width=390 height=844></iframe><script>
function rep(o){fetch('/report?'+encodeURIComponent(JSON.stringify(o)));}
var f=document.getElementById('f');
f.onload=function(){ var d=f.contentDocument, W=f.contentWindow, q=function(s){return d.querySelector(s)};
 function pe(el,t){ el.dispatchEvent(new W.PointerEvent(t,{bubbles:true,cancelable:true,pointerId:1,isPrimary:true})); }
 function pey(el,t,x){ var r=el.getBoundingClientRect(); el.dispatchEvent(new W.PointerEvent(t,{bubbles:true,cancelable:true,
   pointerId:2,isPrimary:true,clientX:x,clientY:r.top+r.height/2})); }
 function cmds(){ return fetch('/cmds').then(function(r){return r.json()}); }
 function until(fn, ms){ return new Promise(function(res){ var t0=Date.now();
   (function t(){ var ok=false; try{ ok=fn(); }catch(e){} if(ok||Date.now()-t0>ms) res(ok); else setTimeout(t,50); })(); }); }
 var S=%s; var i=0;
 function go(){ if(i>=S.length) return; try{ eval(S[i][0]); }catch(e){ rep({step:'error',err:String(e),at:i}); }
   var w=S[i][1]; i++; setTimeout(go, w); }
 (function ready(){ if(d.body.getAttribute('data-phase')) go(); else setTimeout(ready, 50); })(); };
</script></body></html>""" % json.dumps(STEPS)).encode()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _j(self, o, code=200):
        b = json.dumps(o).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def _raw(self, b):
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        p, _, q = self.path.partition("?")
        if p == "/h": return self._raw(HARNESS)
        if p == "/": return self._raw(PAGE)
        if p == "/img/ikd-logo.svg":
            b = open(os.path.join(HERE, "img", "ikd-logo.svg"), "rb").read()
            self.send_response(200); self.send_header("Content-Type", "image/svg+xml")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        if p == "/stream.mjpg": return self.send_error(404)
        if DEAD["on"] and p in ("/state", "/cmd", "/console"):
            time.sleep(4); return
        if p == "/state":
            if SLOW["on"]: time.sleep(0.8)
            return self._j(dict(ST, control="you"))
        if p == "/sim/state":
            SIMPOLLS.append(1)
            return self._j(dict(ST, control="you", real_armed=False, sim=dict(bf=2)))
        if p == "/sim/console":
            return self._j([])
        if p == "/sim/cmd":
            CMDS.append("SIM " + "&".join(x for x in q.split("&") if not x.startswith(("k=", "c="))))
            return self._j(dict(ST, control="you", refused="", real_armed=False, sim=dict(bf=2)))
        if p == "/cmd":
            CMDS.append(q)
            return self._j(dict(ST, control="you", refused=""))
        if p == "/console":
            since = int(parse_qs(q).get("since", ["0"])[0]); return self._j([r for r in CONSOLE if r[0] > since])
        if p == "/cmds": return self._j(CMDS)
        if p == "/reset": CMDS.clear(); return self._j({})
        if p == "/dead": DEAD["on"] = q == "1"; return self._j({})
        if p == "/set":
            a = parse_qs(q)
            ST["mission"]["phase"] = a.get("phase", [ST["mission"]["phase"]])[0]
            if "armed" in a:
                ST["armed"] = a["armed"][0] == "1"
            ST["mission"]["locked"] = a.get("locked", ["0"])[0] == "1"
            if "lock_hdg" in a:
                ST["mission"]["lock_hdg"] = float(a["lock_hdg"][0])
            if "heading" in a:
                ST["mission"]["heading_deg"] = float(a["heading"][0])
            if "auto" in a:
                ST["mission"]["auto_enabled"] = a["auto"][0] == "1"
            for k in a:
                if k.startswith("lvl_"):
                    ST["mission"]["levels"][k[4:]] = a[k][0]
            en = a.get("engaged", ["0"])[0] == "1"
            ST["mission"]["engaged"] = en
            ST["mission"]["ai_steering"] = en and ST["mission"]["phase"] in ("AI", "PUSH")
            return self._j({})
        if p == "/report": REP.append(json.loads(unquote(q))); return self._j({})
        self.send_error(404)


srv = ThreadingHTTPServer(("127.0.0.1", 8199), H); srv.daemon_threads = True
threading.Thread(target=srv.serve_forever, daemon=True).start()
prof = tempfile.mkdtemp()
ff = subprocess.Popen(["firefox", "--headless", "--no-remote", "--profile", prof, "--window-size", "390,844",
                       "http://127.0.0.1:8199/h"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
t0 = time.time()
while time.time() - t0 < 120 and not any(r.get("done") or r.get("step") == "error" for r in REP):
    time.sleep(0.3)
ff.kill(); shutil.rmtree(prof, ignore_errors=True)
R = {r["step"]: r for r in REP}
print("=== the phone app in headless Firefox ===")
check("page ran every step", "dead" in R and "error" not in R, str(R.get("error", "")))
b = R.get("boot", {})
check("no ABORT button on the page (removed on request)", b.get("abort") is False)
check("no PIN anywhere: no PIN box on the page", b"pin_in" not in PAGE and b"PIN" not in PAGE)
check("the whole page is in English (no Greek letters)",
      not any(0x370 <= ord(ch) <= 0x3ff or 0x1f00 <= ord(ch) <= 0x1fff for ch in PAGE.decode("utf-8")))
check("readiness chip: 'TensorRT 28 fps' in green", b.get("ai", "").startswith("TensorRT 28 fps")
      and "ok" in b.get("aicls", ""), f"{b.get('ai')} [{b.get('aicls')}]")
check("ENGAGE starts red and says 'ENGAGE'", "go" not in b.get("eng", "go") and "ENGAGE" in b.get("engtxt", ""))
check("no phase card (no PILOT / HOVER banner, no hint line): the picture gets the room", b.get("card") is False, str(b.get("card")))
check("LOCK / ENGAGE right of the pad, not in the picture", not b.get("engInVid") and b.get("engRightOfPad"), str(b))
check("LOCK right ABOVE ENGAGE, one column (same left edge)",
      b.get("lockB", 999) <= b.get("engT", 0) and abs(b.get("lockL", 0) - b.get("engL", 99)) <= 1,
      str({k: b.get(k) for k in ("lockL", "lockB", "engL", "engT")}))
check("both a bit bigger: 74 px round (was 66)", 72 <= b.get("engW", 0) <= 76 and 72 <= b.get("lockW", 0) <= 76,
      f"{b.get('engW')} / {b.get('lockW')}")
check("no message bar and no speed/message buttons under the picture; a ☰ menu top-left",
      b.get("ticker") is False and b.get("tools") is False and b.get("menu") and b.get("menuTL"), str(b))
check("light theme: light-grey page; controls, heading bar and height box a neutral GREY (no black)",
      b.get("bodyBg") == "rgb(236, 238, 241)" and b.get("btnBg") == "rgb(86, 93, 102)"
      and b.get("dialBg") == b.get("btnBg") and b.get("hbBg") == b.get("btnBg"),
      f"{b.get('bodyBg')} / {b.get('btnBg')} / {b.get('dialBg')} / {b.get('hbBg')}")
check("no pure black left in the page's styles (the video's own background aside)",
      b"#15181c" not in PAGE and b"#101317" not in PAGE and PAGE.count(b"#000") == 1)
check("no target-type button (always DRONE)", b.get("tgtbtn") is False)
check("the pad sits at the left edge, the buttons at the right edge (inside their 8 px plate)",
      b.get("padL", 99) <= 16 and b.get("engR", 0) >= 390 - 24, f"pad left {b.get('padL')}, ENGAGE right {b.get('engR')}")
tp, t3 = R.get("tape", {}), R.get("tape352", {})
check("compass tape: full width, along the top edge of the picture, ~26 px high",
      abs(tp.get("top", 99)) <= 2 and tp.get("wdiff", 99) <= 3 and 20 <= tp.get("h", 0) <= 30, str(tp))
check("compass tape draws marks on both sides at 123 deg, and across north at 352 deg",
      min(tp.get("left", 0), tp.get("right", 0), t3.get("left", 0), t3.get("right", 0)) > 25,
      f"{tp} {t3}")
check("LOCK: a white button of the same size as ENGAGE",
      abs(b.get("lockW", 0) - b.get("engW", 99)) <= 1 and b.get("lockBg") == "rgb(255, 255, 255)"
      and b.get("lockTxt") == "LOCK", str({k: b.get(k) for k in ("lockW", "lockBg", "lockTxt")}))
lo, ld, lf = R.get("lock_on", {}), R.get("locked", {}), R.get("lock_off", {})
check("LOCK pressed: sends lock=on", lo.get("cmds") == ["lock=on"], str(lo))
check("locked: ring on, shows the heading 352°", "on" in ld.get("cls", "") and "352" in ld.get("txt", "")
      and ld.get("phase") == "LOCKED", str(ld))
check("LOCK pressed again: sends lock=off (never a blind toggle)", lf.get("cmds") == ["lock=off"], str(lf))
check("no vibration anywhere in the page", b"vibrate" not in PAGE)
check("top right is empty: no UNLOCKED / LOCKED and no FREE / YOU CONTROL", b.get("pills") == 0, str(b.get("pills")))
un = R.get("unlocked", {})
check("unlocked again: the LOCK ring off, no heading", "on" not in un.get("cls", "").split() and un.get("txt") == "LOCK", str(un))
check("the buttons fit the phone (ENGAGE inside the 390 px width)", b.get("engR2", 999) <= 390, str(b.get("engR2")))
check("heading bar: BELOW the picture, from its left edge, ~46 px high, active while hovering",
      b.get("dialT", 0) >= b.get("vbB", 999) and abs(b.get("vbL", 0) - (b.get("dialR", 0) - b.get("dialW", 0))) <= 2
      and 40 <= b.get("dialH", 0) <= 50 and b.get("dialW", 0) > 180 and not b.get("dialOff"),
      str({k: b.get(k) for k in ("dialT", "vbB", "vbL", "dialR", "dialW", "dialH", "dialOff")}))
check("height box: right of the bar, same colour, shows the height (10.2 m)",
      b.get("hbL", 0) >= b.get("dialR", 999) and abs(b.get("hbT", 0) - b.get("dialT", 0)) <= 2
      and b.get("hbBg") == b.get("dialBg") and b.get("alt") == "10.2 m",
      str({k: b.get(k) for k in ("hbL", "dialR", "hbT", "dialT", "hbBg", "dialBg", "alt")}))
check("the picture stays 4:3", abs(b.get("vbW", 0) / max(1, b.get("vbH", 1)) - 4 / 3) < 0.02,
      f"{b.get('vbW')} x {b.get('vbH')}")
dg, ak, ay = R.get("dragging", {}), R.get("asked", {}), R.get("ask_yes", {})
check("while dragging: no pop-up yet, the scroll highlighted", dg.get("ask") is False and dg.get("drag"), str(dg))
try:
    picked = int(ak.get("deg", "x").rstrip("°"))
except ValueError:
    picked = None
check("finger up: the pop-up asks 'LOCK AT <deg>?', dragged LEFT = more degrees (from 123)",
      ak.get("ask") and picked is not None and 130 <= picked <= 160 and "turns" in ak.get("sub", ""), str(ak))
check("nothing is sent while scrolling or asking", ak.get("cmds") == [], str(ak.get("cmds")))
check("YES: sends lock=<that heading>, the pop-up closes",
      ay.get("cmds") == [f"lock={picked}"] and ay.get("ask") is False, str(ay))
ad, an = R.get("asked_down", {}), R.get("ask_no", {})
try:
    down = int(ad.get("deg", "x").rstrip("°"))
except ValueError:
    down = None
check("dragged RIGHT = fewer degrees", down is not None and 80 <= down < 123, str(ad))
check("NO: nothing sent, the pop-up closes", an.get("cmds") == [] and an.get("ask") is False, str(an))
check("a touch on the scroll without moving asks nothing", R.get("dial_tap", {}).get("ask") is False, str(R.get("dial_tap")))
check("PUSH: the degree scroll is off and a drag asks nothing",
      R.get("push_dial", {}).get("off") and R.get("push_dial2", {}).get("ask") is False,
      f"{R.get('push_dial')} {R.get('push_dial2')}")
so, sc, sf = R.get("sim_on", {}), R.get("sim_cmds", {}), R.get("sim_off", {})
sm = R.get("sim_menu", {})
check("☰ = a MENU with three sections: PARAMETERS, MESSAGES, TEST; TEST opens its own page",
      sm.get("rows") == ["PARAMETERS", "MESSAGES", "TEST"] and sm.get("menu") and sm.get("test"), str(sm))
check("menu rows: the text starts at the LEFT of each row (not centred)",
      sm.get("indent") and all(0 <= x <= 24 for x in sm.get("indent")), str(sm.get("indent")))
check("TEST → START TEST (drone disarmed): allowed; the menu closes, the TEST keys show, a drawn view instead of the camera",
      R.get("sim_menu", {}).get("dis") is False and so.get("bar") == "flex" and so.get("img") == "hidden"
      and so.get("canv") == "block" and so.get("sheet") is False and len(SIMPOLLS) > 2, f"{so} polls {len(SIMPOLLS)}")
sk = so.get("sky", [0, 0, 0, 0])
check("TEST: a drawn sky in the picture (blue, not black), and no words over it (no badge)",
      sk[3] == 255 and sk[2] > sk[0] + 40 and sk[2] > 120 and so.get("words") == "none", f"{sk} badge {so.get('words')}")
check("TEST: no bar over the picture any more (no GUST / WIND 0-3 / ↺ buttons)", so.get("old") is False, str(so.get("old")))
wd, ex = so.get("wind", [99, 99, 99, 0]), so.get("exit", [99, 99, ""])
check("TEST: two small WIND keys with icons INSIDE the picture, top-left under the compass tape",
      so.get("inPic") and so.get("svg") and 0 <= wd[0] <= 16 and 26 <= wd[1] <= 44 and 0 <= wd[2] <= 10 and 30 <= wd[3] <= 44, str(wd))
check("TEST: EXIT (just EXIT) inside the picture, top-right", so.get("inPic") and 0 <= ex[0] <= 16 and 26 <= ex[1] <= 44
      and ex[2].replace("\u2715", "").strip() == "EXIT", str(ex))
check("TEST look: plain black (no stripes, no purple); no az / el readout, no 'pretend drone' label",
      b"repeating-linear-gradient" not in PAGE and not any(c in PAGE for c in (b"#6d28d9", b"#5b21b6", b"#2e1065", b"#facc15"))
      and b'id="offs"' not in PAGE and b"TEST \xc2\xb7 pretend drone" not in PAGE)
check("in TEST the buttons go to the pretend drone ONLY (/sim/cmd): LOCK, a wind gust",
      sorted(sc.get("cmds", [])) == ["SIM gust=r", "SIM lock=on"], str(sc))
check("EXIT: back to the real drone, the TEST keys gone", sf.get("sim") is False and sf.get("bar") == "none" and sf.get("x") == "none", str(sf))
check("drone armed: TEST cannot be started", R.get("sim_armed", {}).get("dis") is True, str(R.get("sim_armed")))
hd = R.get("held", {}).get("cls", "")
check("while ▲ is held and the Jetson accepts it: highlighted (held) AND marked sending (tx)",
      "held" in hd and "tx" in hd, hd)
h = R.get("hold_up", {})
check("hold ▲ 1.2 s: ~10 keepalives, then a release; no PIN in any command",
      8 <= h.get("n", 0) <= 14 and h.get("rel") and h.get("pin"), str(h))
check("a 120 ms tap on ⟲ is ONE yaw step, not a hold", R.get("tap", {}).get("cmds") == ["step=l"], str(R.get("tap")))
check("holding ⟳ then the page losing focus: release sent, keepalives stop",
      R.get("blur", {}).get("rel") and R.get("after_blur", {}).get("n") == 0, f"{R.get('blur')} {R.get('after_blur')}")
check("ENGAGE: one quick tap = exactly one engage", R.get("engage_tap", {}).get("n") == 1, str(R.get("engage_tap")))
a = R.get("ai", {})
check("AI took over: ENGAGE green 'AI ✓', disabled; manual buttons disabled",
      "go" in a.get("eng", "") and "AI" in a.get("engtxt", "") and a.get("dis") and a.get("up"), str(a))
l = R.get("latched", {})
check("after the attempt: still green 'DONE ✓', still disabled",
      "go" in l.get("eng", "") and l.get("dis") and "DONE" in l.get("engtxt", ""), str(l))
check("pilot STAB -> ALT_HOLD: ENGAGE red and enabled again",
      "go" not in R.get("reset", {}).get("eng", "go") and R.get("reset", {}).get("dis") is False)
sp = R.get("speeds", {})
check("☰ → PARAMETERS opens FULL-SCREEN", sp.get("open") and sp.get("pos") == "fixed"
      and sp.get("h", 0) >= 0.95 * sp.get("winh", 1), str(sp))
so2 = R.get("semiopen", {})
check("PARAMETERS: three grey sections AUTO, SEMI-AUTO, DETECTION (in that order), all closed at first, each with what it has set; dark blue, not orange",
      sp.get("heads") == ["AUTO", "SEMI-AUTO", "DETECTION"] and sp.get("nOpen") == 0
      and "PUSH" in (sp.get("sums") or [""])[0] and "15 m" in (sp.get("sums") or ["", ""])[1] and "MAX" in (sp.get("sums") or ["", ""])[1]
      and sp.get("acc") == "rgb(31, 59, 99)" and sp.get("boxBg") == "rgb(226, 230, 235)"
      and "follower ON" in (sp.get("sums") or ["", "", ""])[2], str({k: sp.get(k) for k in ("heads", "nOpen", "sums", "acc", "boxBg")}))
check("...every setting lives in its own section (speeds in SEMI-AUTO, AUTO's in AUTO, sky gate + follower in DETECTION)",
      sp.get("homes") == [True, True, True], str(sp.get("homes")))
check("a tap on SEMI-AUTO opens it and only it", so2.get("semi") and not so2.get("auto") and not so2.get("det")
      and so2.get("exp") == "true", str(so2))
check("...each setting's DEFAULT marked (▲▼ SLOW here); nothing 'this flight only' yet; no question open",
      sp.get("isdef") == "SLOW" and sp.get("unsaved") == 0 and sp.get("ask") is False, str(sp))
a1, d1 = R.get("ask1", {}), R.get("def1", {})
check("a speed button sends lvl=vert:FAST, then asks: make it the default? (name, value, the default now)",
      a1.get("cmds") == ["lvl=vert:FAST"] and a1.get("ask") and "HEIGHT SPEED" in (a1.get("k") or "")
      and (a1.get("v") or "").startswith("FAST") and "SLOW" in (a1.get("sub") or ""), str(a1))
check("YES: sends def=lvl:vert, the question closes", d1.get("cmds") == ["lvl=vert:FAST", "def=lvl:vert"]
      and d1.get("ask") is False, str(d1))
check("SEARCH: 4 s / 6 s / 8 s with the distance at PUSH 20°, 4 s selected; 8 s sends lvl=search:FAST",
      sp.get("search") == ["4 s~19 m", "6 s~42 m", "8 s~69 m"] and sp.get("searchOn") == "4 s~19 m"
      and R.get("lvl", {}).get("cmds") == ["lvl=search:FAST"], f"{sp.get('search')} {sp.get('searchOn')} {R.get('lvl')}")
check("ONLY THIS FLIGHT: nothing more is sent, the question closes",
      R.get("ask2", {}).get("ask") and R.get("lvl", {}).get("ask") is False and "def=" not in str(R.get("lvl")), str(R.get("lvl")))
fo = R.get("fol", {})
check("FOLLOWER: ON / OFF after the SKY GATE, ON lit from the status; OFF sends follow=off, then asks too",
      sp.get("fol") == ["ONdetector + follower", "OFFdetector only"] and sp.get("folOn") == "ONdetector + follower"
      and fo.get("cmds") == ["follow=off"] and fo.get("ask") and fo.get("k") == "FOLLOWER" and fo.get("v") == "OFF",
      f"{sp.get('fol')} {sp.get('folOn')} {fo}")
check("a level that differs from its default shows THIS FLIGHT ONLY · SAVE; a tap sends def=lvl:vert; ✕ closes",
      R.get("chip", {}).get("on") and R.get("chip", {}).get("n") == 1
      and R.get("chipsave", {}).get("cmds") == ["def=lvl:vert"] and R.get("closed", {}).get("open") is False,
      f"{R.get('chip')} {R.get('chipsave')}")
check("console shows the tracker's printed lines", len(R.get("console", {}).get("lines", [])) == 2)
lg = R.get("logsheet", {})
check("MESSAGES opens FULL-SCREEN (not a pop-up) and ✕ closes it",
      lg.get("open") and lg.get("pos") == "fixed" and lg.get("h", 0) >= 0.95 * lg.get("winh", 1)
      and R.get("logclosed", {}).get("open") is False, str(lg))
pu = R.get("push", {})
check("PUSH: ▲, ENGAGE and LOCK disabled; the aim mark moves to the centre", pu.get("up") and pu.get("eng") and pu.get("lockdis")
      and pu.get("aimTop") == "50%", str(pu))
ao, at, ah = R.get("auto_off", {}), R.get("auto_tap", {}), R.get("auto_hold", {})
check("AUTO: a pill right above LOCK, same column", ao.get("above") is True and ao.get("dl", 99) < 2
      and (ao.get("w") or 0) >= 60 and 30 <= (ao.get("h") or 0) <= 44, str(ao))
check("AUTO greyed out while OFF in PARAMETERS", ao.get("dis") is True, str(ao))
check("AUTO ON: a tap sends auto=on", at.get("cmds") == ["auto=on"], str(at))
check("AUTO flying: orange, says WATCHING, tap again would stop it (enabled)",
      "on" in (ah.get("cls") or "") and "WATCHING" in (ah.get("txt") or "") and ah.get("dis") is False, str(ah))
check("...⟲ ⟳ usable to look around, ▲ and LOCK not; ENGAGE (go now) usable",
      ah.get("yl") is False and ah.get("yr") is False and ah.get("up") is True and ah.get("lock") is True
      and ah.get("eng") is False, str(ah))
check("AUTO lining up: says LINING UP, ⟲ ⟳ off", "LINING UP" in (R.get("auto_align", {}).get("txt") or "")
      and R.get("auto_align", {}).get("yl") is True, str(R.get("auto_align")))
ab = R.get("ai_btns", {})
check("AI row has SLOW / MEDIUM / FAST / MAX; PUSH row has no MAX",
      ab.get("ai") == ["SLOW", "MEDIUM", "FAST", "MAX"] and ab.get("push") == ["SLOW", "MEDIUM", "FAST"], str(ab))
check("PARAMETERS has the AUTO ON / OFF switch, showing ON", ab.get("autoseg") == ["on", "off"]
      and ab.get("autoOn") == "on", str(ab))
check("MAX sends lvl=ai:MAX (and asks about the default); the AUTO switch sends autoen=off and asks nothing more",
      sorted(R.get("max_autoen", {}).get("cmds") or []) == ["autoen=off", "lvl=ai:MAX"]
      and R.get("max_autoen", {}).get("askMax"),
      str(R.get("max_autoen")))
hb, apg = R.get("hub", {}), R.get("autopage", {})
check("PARAMETERS: no sub-pages any more; the AUTO section carries the ON tag",
      hb.get("heads") == ["AUTO", "SEMI-AUTO", "DETECTION"] and hb.get("cats") == 0 and hb.get("tag") == "ON" and hb.get("tagIn"), str(hb))
check("a tap on AUTO opens it and closes SEMI-AUTO; SEMI-AUTO again swaps back; a second tap closes it",
      apg.get("auto") and not apg.get("semi") and apg.get("exp") == "true"
      and R.get("autoback", {}).get("semi") and not R.get("autoback", {}).get("auto")
      and R.get("toggle", {}).get("semi") is False and R.get("toggle", {}).get("exp") == "false",
      str(apg) + str(R.get("autoback")) + str(R.get("toggle")))
check("AUTO section: the AUTO button, then 12 settings in three sections, each lit and its default marked, each with a SAVE chip",
      apg.get("secs") == ["AUTO BUTTON", "LEG 1 · CLIMB AND WATCH", "LINING UP · AFTER THE FIRST DETECTION", "LEG 2 · THE RUN"]
      and apg.get("n", 0) >= 36 and len(apg.get("on", [])) == 12 and apg.get("dflt") == 12 and apg.get("chips") == 12, str(apg))
check("AUTO button (OFF) and SKY GATE (ON) are marked as fixed power-up values",
      apg.get("autoenDef") == "off" and apg.get("skyDef") == "on", str(apg))
check("a tap on the default value sends autop=height:15 and asks nothing; RESET sends autoreset=1 (factory = default: no question)",
      sorted(R.get("autoset", {}).get("cmds") or []) == ["autop=height:15", "autoreset=1"]
      and R.get("autoset", {}).get("ask") is False,
      str(R.get("autoset")))
aa, adf = R.get("autoask", {}), R.get("autodef", {})
check("HEIGHT 20 m: asks to make it the default (default now 15 m); YES sends def=auto:height",
      aa.get("ask") and aa.get("k") == "HEIGHT" and aa.get("v") == "20 m" and "15 m" in (aa.get("sub") or "")
      and adf.get("cmds") == ["autop=height:20", "def=auto:height"] and adf.get("ask") is False, f"{aa} {adf}")
check("link dead: the red cover shows", R.get("dead", {}).get("cover") is True)

DEAD["on"] = False
SIZES = [("tablet upright", 820, 1180), ("tablet upright small", 768, 1024), ("tablet sideways", 1180, 820),
         ("tablet sideways small", 1024, 768), ("laptop", 1366, 657), ("desktop", 1920, 1080)]
LAYOUT = ("""<!doctype html><html><body style="margin:0">%s<script>
function rep(o){fetch('/report?'+encodeURIComponent(JSON.stringify(o)));}
function box(e){var r=e.getBoundingClientRect();return {l:r.left,t:r.top,r:r.right,b:r.bottom,w:r.width,h:r.height};}
[].forEach.call(document.querySelectorAll('iframe'), function(f){ f.onload=function(){ var d=f.contentDocument, W=f.contentWindow,
 q=function(s){return d.querySelector(s)};
 (function ready(){ if(!d.body.getAttribute('data-phase')) return setTimeout(ready,50);
  setTimeout(function(){ var all=[].map.call(d.querySelectorAll('.app button, .tile, #vbox, .belt'), box);
   rep({step:'layout_'+f.id, vw:W.innerWidth, vh:W.innerHeight, vb:box(q('#vbox')), tiles:box(q('.tiles')), ctl:box(q('.ctl')),
     tile0:box(q('.tile')), tile5:box(d.querySelectorAll('.tile')[5]), btn:box(q('#b_up')), eng:box(q('#b_eng')), pad:box(q('.pad')), lock:box(q('#b_lock')), keys:['#b_up','#b_dn','#b_yl','#b_yr'].map(function(k){return box(q(k));}), belt:box(q('.belt')), top:box(q('.brand')), logo:box(q('.logo')), logoOk:q('.logo').complete && q('.logo').naturalWidth+q('.logo').naturalHeight>0, logoNat:[q('.logo').complete,q('.logo').naturalWidth,q('.logo').naturalHeight], inVb:q('#vbox').contains(q('.ctl')), plates:[W.getComputedStyle(q('.pad')).backgroundColor, W.getComputedStyle(q('.acts')).backgroundColor],
     out:all.filter(function(r){return r.l<-1||r.t<-1||r.r>W.innerWidth+1||r.b>W.innerHeight+1;}).length,
     sw:d.documentElement.scrollWidth});
  }, 400); })(); }; });
</script></body></html>""" % "".join(f'<iframe id=f{i} src="/" width={w} height={h} style="border:0;display:block"></iframe>'
                                     for i, (_, w, h) in enumerate(SIZES))).encode()
HARNESS = LAYOUT
REP.clear()
prof = tempfile.mkdtemp()
ff = subprocess.Popen(["firefox", "--headless", "--no-remote", "--profile", prof, "--window-size", "1920,1080",
                       "http://127.0.0.1:8199/h"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
t0 = time.time()
while time.time() - t0 < 90 and len({r.get("step") for r in REP if r.get("step", "").startswith("layout_")}) < len(SIZES):
    time.sleep(0.3)
ff.kill(); shutil.rmtree(prof, ignore_errors=True)
R = {r["step"]: r for r in REP}
print("=== the same page on a tablet and a desktop ===")
for i, (name, w, h) in enumerate(SIZES):
    L = R.get(f"layout_f{i}")
    if not L:
        check(f"{name} {w}x{h}: page ran", False); continue
    vb, ctl, tl = L["vb"], L["ctl"], L["tiles"]
    check(f"{name} {w}x{h}: every control, tile and the picture inside the screen, no sideways scroll",
          L["out"] == 0 and L["sw"] <= w, f"out {L['out']} scrollW {L['sw']}")
    check(f"{name}: the picture stays 4:3", abs(vb["w"] / vb["h"] - 4 / 3) < 0.02, f"{vb['w']:.0f}x{vb['h']:.0f}")
    pad, eng, lk, ks = L["pad"], L["eng"], L["lock"], L["keys"]
    check(f"{name}: one line on top: ☰ + name, then the six status cells in one row; the picture under it",
          abs(L["tile0"]["t"] - L["tile5"]["t"]) < 1 and tl["l"] >= L["top"]["r"] and abs((tl["t"] + tl["b"]) - (L["top"]["t"] + L["top"]["b"])) < 4
          and tl["b"] <= vb["t"], f"top right {L['top']['r']:.0f}, tiles {tl['l']:.0f}..{tl['r']:.0f}")
    check(f"{name}: the IKnowHow Defence logo loads, at the far right of the top line",
          L["logoOk"] and L["logo"]["l"] >= tl["r"] and L["logo"]["r"] <= w and abs((L["logo"]["t"] + L["logo"]["b"]) - (tl["t"] + tl["b"])) < 4,
          f"logo {L['logo']['l']:.0f}..{L['logo']['r']:.0f} t{L['logo']['t']:.0f}-{L['logo']['b']:.0f}, tiles t{tl['t']:.0f}-{tl['b']:.0f} end {tl['r']:.0f} nat {L['logoNat']}")
    check(f"{name}: the picture as big as it fits (full width or full height)",
          vb["w"] >= w - 50 or vb["b"] >= h - 120, f"{vb['w']:.0f}x{vb['h']:.0f} of {w}x{h}")
    check(f"{name}: the heading bar right under the picture", 0 <= L["belt"]["t"] - vb["b"] <= 10, f"{L['belt']['t'] - vb['b']:.0f}")
    check(f"{name}: the pad INSIDE the picture, bottom-left",
          L["inVb"] and pad["l"] >= vb["l"] and pad["l"] - vb["l"] <= 40 and pad["b"] <= vb["b"] and vb["b"] - pad["b"] <= 40,
          f"pad {pad['l'] - vb['l']:.0f} from the left, {vb['b'] - pad['b']:.0f} from the bottom")
    check(f"{name}: LOCK above ENGAGE INSIDE the picture, bottom-right",
          eng["r"] <= vb["r"] and vb["r"] - eng["r"] <= 50 and eng["b"] <= vb["b"] and vb["b"] - eng["b"] <= 50
          and lk["b"] <= eng["t"] and abs(lk["l"] - eng["l"]) <= 1, f"engage {vb['r'] - eng['r']:.0f} from the right")
    check(f"{name}: ▲ ▼ ⟲ ⟳ all the SAME size (square), big enough ({'>= 60' if w < h or name.startswith('tablet') else '>= 48'} px)",
          max(k["w"] for k in ks) - min(k["w"] for k in ks) < 1 and max(k["h"] for k in ks) - min(k["h"] for k in ks) < 1
          and abs(ks[0]["w"] - ks[0]["h"]) < 1 and ks[0]["h"] >= (60 if name.startswith("tablet") else 48),
          " ".join(f"{k['w']:.0f}x{k['h']:.0f}" for k in ks))
    check(f"{name}: a dark plate behind the pad and behind LOCK / ENGAGE",
          all(c.startswith("rgba(14, 17, 21") for c in L["plates"]), str(L["plates"]))
print()
if FAILED:
    print(f"{len(FAILED)} FAILED:")
    for f in FAILED:
        print("  - " + f)
    sys.exit(1)
print("all passed")
