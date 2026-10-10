#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from browser_support import available, launch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = open(os.path.join(HERE, "mission_ui.html"), "rb").read()
FAILED = []


def check(label, ok, detail=""):

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

check("same settings on the page as in the mission", sorted(_ui) == sorted(_ms), f"{sorted(_ui)} vs {sorted(_ms)}")
check("same choices and the same default for every one",
      all(_norm(_ui[k][0]) == _norm(_ms[k][0]) and _norm([_ui[k][1]]) == _norm([_ms[k][1]]) for k in _ms if k in _ui),
      str({k: (_ui.get(k), _ms[k]) for k in _ms if k in _ui and (_norm(_ui[k][0]) != _norm(_ms[k][0]) or _norm([_ui[k][1]]) != _norm([_ms[k][1]]))}))

if not available():

    sys.exit(0)

ST = dict(linked=True, armed=True, fc_mode="ALT_HOLD", target="DRONE", tilt_deg=30.0,
          detector="TensorRT", fps=28.0, follower=True,
          mission=dict(app="mission", phase="HOVER", engaged=False, ai_steering=False,
                       levels=dict(vert="SLOW", yaw="MEDIUM", push="SLOW", ai="MEDIUM"),
                       speeds=dict(climb=1.0, descent=0.5, yaw=20, step=5, push=20), min_conf=0.2,
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
CMDS, REP = [], []
SLOW, DEAD = {"on": False}, {"on": False}

STEPS = [
    ("rep({step:'startup', chooser:!!q('#modechooser'), popup:q('#modechange').classList.contains('on'), inert:q('.app').inert});", 600),
    ("fetch('/reset');", 200),
    ("rep({step:'boot', phase:d.body.getAttribute('data-phase'), alt:q('#alt').textContent, card:!!q('#hdr')||!!q('#phase')||!!q('#hint'), hbL:q('#hbox').getBoundingClientRect().left, hbT:q('#hbox').getBoundingClientRect().top, hbBg:W.getComputedStyle(q('#hbox')).backgroundColor, dialBg:W.getComputedStyle(q('#dial')).backgroundColor, dialT:q('#dial').getBoundingClientRect().top, dialR:q('#dial').getBoundingClientRect().right, dialW:q('#dial').getBoundingClientRect().width, vbB:q('#vbox').getBoundingClientRect().bottom, vbL:q('#vbox').getBoundingClientRect().left, lockL:q('#b_lock').getBoundingClientRect().left, lockT:q('#b_lock').getBoundingClientRect().top, engL:q('#b_eng').getBoundingClientRect().left, engT:q('#b_eng').getBoundingClientRect().top, engR:q('#b_eng').getBoundingClientRect().right, padL:q('.pad').getBoundingClientRect().left, abort:q('#b_wasp_abort').getClientRects().length>0, eng:q('#b_eng').className, engtxt:q('#engtxt').textContent, aimTop:q('#aim').style.top, badge:q('#badge').textContent, ai:q('#c_ai_k').textContent+' '+q('#c_ai').textContent, aicls:q('#c_ai').className, tgtbtn:!!q('#t_tgt'), engInVid:q('#vbox').contains(q('#b_eng')), engW:q('#b_eng').getBoundingClientRect().width, engRightOfPad:q('#b_eng').getBoundingClientRect().left>=q('.pad').getBoundingClientRect().right, engVsPad:Math.abs((q('#b_eng').getBoundingClientRect().top+q('#b_lock').getBoundingClientRect().bottom)/2-(q('.pad').getBoundingClientRect().top+q('.pad').getBoundingClientRect().bottom)/2), ticker:!!q('#ticker'), tools:!!q('.tools'), menu:!!q('#t_menu'), menuTL:q('#t_menu').getBoundingClientRect().left<40 && q('#t_menu').getBoundingClientRect().top<60, bodyBg:W.getComputedStyle(d.body).backgroundColor, btnBg:W.getComputedStyle(q('#b_up')).backgroundColor, grpC:Math.abs((q('.pad').getBoundingClientRect().left+q('#b_eng').getBoundingClientRect().right)/2-W.innerWidth/2), lockAbove:q('#b_lock').getBoundingClientRect().bottom<=q('#b_eng').getBoundingClientRect().top, lockW:q('#b_lock').getBoundingClientRect().width, lockBg:W.getComputedStyle(q('#b_lock')).backgroundColor, lockTxt:q('#b_lock').textContent, pills:d.querySelectorAll('.top .pill, #c_lock, #c_ctl').length, lockB:q('#b_lock').getBoundingClientRect().bottom, lockR:q('#b_lock').getBoundingClientRect().right, engR2:q('#b_eng').getBoundingClientRect().right, dialL:q('#dial').getBoundingClientRect().left, vbR:q('#vbox').getBoundingClientRect().right, dialH:q('#dial').getBoundingClientRect().height, dialOff:q('#dial').classList.contains('off'), vbL:q('#vbox').getBoundingClientRect().left, vbH:q('#vbox').getBoundingClientRect().height, vbW:q('#vbox').getBoundingClientRect().width, padC:Math.abs((q('.pad').getBoundingClientRect().left+q('.pad').getBoundingClientRect().right)/2-W.innerWidth/2)}); fetch('/reset');", 400),
    ("var cv=q('#tape'), vb=q('#vbox').getBoundingClientRect(), r=cv.getBoundingClientRect(); window._ink=function(x0,x1){var c=cv.getContext('2d'), d=c.getImageData(Math.round(cv.width*x0),0,Math.max(1,Math.round(cv.width*(x1-x0))),cv.height).data, n=0; for(var i=3;i<d.length;i+=4) if(d[i]>100) n++; return n;}; rep({step:'tape', top:r.top-vb.top, wdiff:Math.abs(r.width-vb.width), h:r.height, left:_ink(0,0.35), right:_ink(0.65,1)}); fetch('/set?phase=HOVER&heading=352');", 900),
    ("rep({step:'tape352', left:_ink(0,0.35), right:_ink(0.65,1)}); fetch('/set?phase=HOVER&heading=123'); fetch('/reset');", 500),
    ("until(function(){return !q('#b_lock').disabled},3000).then(function(){ return fetch('/reset'); }).then(function(){ pe(q('#b_lock'),'pointerdown'); });", 3400),
    ("cmds().then(function(c){rep({step:'lock_on', cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})}); fetch('/set?phase=HOVER&locked=1&lock_hdg=352');});", 800),
    ("until(function(){return q('#b_lock').classList.contains('on')},3000).then(function(){ rep({step:'locked', cls:q('#b_lock').className, txt:q('#locktxt').textContent, phase:d.body.getAttribute('data-phase'),}); fetch('/reset').then(function(){ pe(q('#b_lock'),'pointerdown'); }); });", 3600),
    ("cmds().then(function(c){rep({step:'lock_off', cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})}); fetch('/set?phase=HOVER&locked=0').then(function(){ fetch('/reset'); });});", 400),
    ("until(function(){return !q('#b_lock').classList.contains('on') && !q('#b_yl').disabled},3000);", 3100),
    ("rep({step:'unlocked', cls:q('#b_lock').className, txt:q('#locktxt').textContent}); fetch('/reset');", 300),
    ("var dl=q('#dial'); pey(dl,'pointerdown',300); [1,2,3,4,5,6].forEach(function(k){ setTimeout(function(){ pey(dl,'pointermove',300-10*k); }, 80*k); }); setTimeout(function(){ rep({step:'dragging', ask:q('#ask').classList.contains('on'), drag:dl.classList.contains('drag')}); pey(dl,'pointerup',240); }, 600);", 900),
    ("cmds().then(function(c){ rep({step:'asked', ask:q('#ask').classList.contains('on'), deg:q('#ask_deg').textContent, sub:q('#ask_sub').textContent, cmds:c}); q('#ask_yes').click(); });", 500),
    ("cmds().then(function(c){rep({step:'ask_yes', ask:q('#ask').classList.contains('on'), cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})}); return fetch('/reset'); }).then(function(){ var dl=q('#dial'); pey(dl,'pointerdown',100); setTimeout(function(){ pey(dl,'pointermove',140); },80); setTimeout(function(){ pey(dl,'pointermove',180); },160); setTimeout(function(){ pey(dl,'pointerup',180); },240); });", 700),
    ("rep({step:'asked_down', deg:q('#ask_deg').textContent}); q('#ask_no').click(); setTimeout(function(){ cmds().then(function(c){ rep({step:'ask_no', ask:q('#ask').classList.contains('on'), cmds:c}); fetch('/reset'); }); }, 300);", 700),
    ("var dl=q('#dial'); pey(dl,'pointerdown',200); pey(dl,'pointerup',200); setTimeout(function(){ rep({step:'dial_tap', ask:q('#ask').classList.contains('on')}); }, 200);", 400),
    ("until(function(){return !q('#b_up').disabled},3000).then(function(){ return fetch('/reset'); }).then(function(){ pe(q('#b_up'),'pointerdown'); setTimeout(function(){ rep({step:'held', cls:q('#b_up').className}); pe(q('#b_up'),'pointerup'); },1200); });", 4600),
    ("cmds().then(function(c){rep({step:'hold_up', n:c.filter(function(x){return x.indexOf('hold=up')>=0}).length, rel:c.some(function(x){return x.indexOf('release=')>=0}), pin:c.every(function(x){return x.indexOf('k=')<0})}); fetch('/reset');});", 300),
    ("until(function(){return !q('#b_yl').disabled},3000).then(function(){ return fetch('/reset'); }).then(function(){ pe(q('#b_yl'),'pointerdown'); setTimeout(function(){pe(q('#b_yl'),'pointerup');},120); });", 3300),
    ("cmds().then(function(c){rep({step:'tap', cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})}); fetch('/reset');});", 300),
    ("until(function(){return !q('#b_yr').disabled},3000).then(function(){ return fetch('/reset'); }).then(function(){ pe(q('#b_yr'),'pointerdown'); });", 3500),
    ("W.dispatchEvent(new W.Event('blur'));", 600),
    ("cmds().then(function(c){rep({step:'blur', n:c.filter(function(x){return x.indexOf('hold=yawr')>=0}).length, rel:c.some(function(x){return x.indexOf('release=')>=0})}); fetch('/reset');});", 700),
    ("cmds().then(function(c){rep({step:'after_blur', n:c.filter(function(x){return x.indexOf('hold=')>=0}).length}); fetch('/reset');});", 300),
    ("until(function(){return !q('#b_eng').disabled},3000).then(function(){ return fetch('/reset'); }).then(function(){ pe(q('#b_eng'),'pointerdown'); setTimeout(function(){pe(q('#b_eng'),'pointerup');},80); });", 3400),
    ("cmds().then(function(c){rep({step:'engage_tap', n:c.filter(function(x){return x.indexOf('engage=')>=0}).length}); fetch('/reset');});", 300),
    ("fetch('/set?phase=AI&engaged=1');", 100),
    ("until(function(){return q('#b_eng').classList.contains('go')},3000).then(function(){ rep({step:'ai', eng:q('#b_eng').className, engtxt:q('#engtxt').textContent, dis:q('#b_eng').disabled, up:q('#b_up').disabled}); fetch('/set?phase=HOVER&engaged=1'); });", 3200),
    ("until(function(){return d.body.getAttribute('data-phase')!=='AI'},3000);", 3100),
    ("rep({step:'latched', eng:q('#b_eng').className, dis:q('#b_eng').disabled, engtxt:q('#engtxt').textContent}); fetch('/set?phase=HOVER&engaged=0');", 100),
    ("until(function(){return !q('#b_eng').classList.contains('go')},3000);", 3100),
    ("rep({step:'reset', eng:q('#b_eng').className, dis:q('#b_eng').disabled});", 400),
    ("fetch('/set?phase=HOVER&auto=0').then(function(){ return until(function(){return q('#engtxt').textContent==='ENGAGE'},3000); }).then(function(){ rep({step:'auto_off', dis:q('#b_eng').disabled, txt:q('#engtxt').textContent, auto:!!q('#b_auto'), buttons:[].filter.call(d.querySelectorAll('.acts .rbtn'),function(b){return b.getClientRects().length>0}).map(function(b){return b.id})}); fetch('/set?phase=HOVER&auto=1'); });", 3200),
    ("until(function(){return q('#engtxt').textContent==='AUTO ENGAGE'&&!q('#b_eng').disabled},3000).then(function(){ return fetch('/reset'); }).then(function(){ pe(q('#b_eng'),'pointerdown'); });", 3400),
    ("cmds().then(function(c){rep({step:'auto_tap', cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})}); fetch('/set?phase=AUTO_HOLD&auto=1');});", 400),
    ("until(function(){return d.body.getAttribute('data-phase')==='AUTO_HOLD'},3000).then(function(){ rep({step:'auto_hold', cls:q('#b_eng').className, dis:q('#b_eng').disabled, txt:q('#engtxt').textContent, yl:q('#b_yl').disabled, yr:q('#b_yr').disabled, up:q('#b_up').disabled, lock:q('#b_lock').disabled, eng:q('#b_eng').disabled}); fetch('/set?phase=AUTO_ALIGN&auto=1'); });", 3200),
    ("until(function(){return d.body.getAttribute('data-phase')==='AUTO_ALIGN'},3000).then(function(){ rep({step:'auto_align', txt:q('#engtxt').textContent, yl:q('#b_yl').disabled}); fetch('/set?phase=HOVER&auto=1'); });", 3200),
    ("q('#t_log').click();", 400),
    ("rep({step:'console', lines:[].map.call(d.querySelectorAll('#log .ln'),function(l){return l.textContent})}); var lg=q('#sh_log'); rep({step:'logsheet', open:lg.classList.contains('open'), pos:W.getComputedStyle(lg).position, h:lg.getBoundingClientRect().height, winh:W.innerHeight}); q('#sh_log [data-close]').click(); rep({step:'logclosed', open:q('#sh_log').classList.contains('open')}); q('#sh_menu [data-close]').click(); fetch('/set?phase=PUSH&engaged=1');", 100),
    ("until(function(){return d.body.getAttribute('data-phase')==='PUSH'},3000).then(function(){ rep({step:'push', up:q('#b_up').disabled, eng:q('#b_eng').disabled, aimTop:q('#aim').style.top, lockdis:q('#b_lock').disabled}); rep({step:'push_dial', off:q('#dial').classList.contains('off')}); var dl=q('#dial'); pey(dl,'pointerdown',300); pey(dl,'pointermove',200); pey(dl,'pointerup',200); setTimeout(function(){ rep({step:'push_dial2', ask:q('#ask').classList.contains('on')}); fetch('/set?phase=HOVER&engaged=0'); fetch('/dead?1'); }, 200); });", 4800),
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
prof = tempfile.mkdtemp(prefix=".ui-browser-", dir=HERE)
ff = launch(prof, "http://127.0.0.1:8199/h")
t0 = time.time()
while time.time() - t0 < 120 and not any(r.get("done") or r.get("step") == "error" for r in REP):
    time.sleep(0.3)
ff.terminate(); ff.wait(timeout=10); shutil.rmtree(prof, ignore_errors=True)
R = {r["step"]: r for r in REP}

check("page ran every step", "dead" in R and "error" not in R, str(R.get("error", "")))
b = R.get("boot", {})
check("ABORT is hidden in Seeker", b.get("abort") is False)
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
check("Test mode removed from the phone app", b'id="t_test"' not in PAGE and b'/sim' not in PAGE)
check("opening the page uses the active mode without a prompt", R.get("startup", {}).get("chooser") is False and R.get("startup", {}).get("popup") is False and R.get("startup", {}).get("inert") is False)
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
check("console shows the tracker's printed lines", len(R.get("console", {}).get("lines", [])) == 2)
lg = R.get("logsheet", {})
check("MESSAGES opens FULL-SCREEN (not a pop-up) and ✕ closes it",
      lg.get("open") and lg.get("pos") == "fixed" and lg.get("h", 0) >= 0.95 * lg.get("winh", 1)
      and R.get("logclosed", {}).get("open") is False, str(lg))
pu = R.get("push", {})
check("PUSH: ▲, ENGAGE and LOCK disabled; the aim mark moves to the centre", pu.get("up") and pu.get("eng") and pu.get("lockdis")
      and pu.get("aimTop") == "50%", str(pu))
ao, at, ah = R.get("auto_off", {}), R.get("auto_tap", {}), R.get("auto_hold", {})
check("Seeker has only Lock and Engage actions", ao.get("buttons") == ["b_lock", "b_eng"] and ao.get("auto") is False, str(ao))
check("Semi-auto keeps Engage ready", ao.get("dis") is False and ao.get("txt") == "ENGAGE", str(ao))
check("AUTO ENGAGE sends one auto=on", at.get("cmds") == ["auto=on"], str(at))
check("AUTO keeps Auto Engage available", ah.get("txt") == "AUTO ENGAGE" and ah.get("dis") is False, str(ah))
check("...⟲ ⟳ usable to look around, ▲ and LOCK not; ENGAGE (go now) usable",
      ah.get("yl") is False and ah.get("yr") is False and ah.get("up") is True and ah.get("lock") is True
      and ah.get("eng") is False, str(ah))
check("AUTO lining up: Auto Engage remains labelled, ⟲ ⟳ off", R.get("auto_align", {}).get("txt") == "AUTO ENGAGE"
      and R.get("auto_align", {}).get("yl") is True, str(R.get("auto_align")))
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
prof = tempfile.mkdtemp(prefix=".ui-browser-", dir=HERE)
ff = launch(prof, "http://127.0.0.1:8199/h", 1920, 1080)
t0 = time.time()
while time.time() - t0 < 90 and len({r.get("step") for r in REP if r.get("step", "").startswith("layout_")}) < len(SIZES):
    time.sleep(0.3)
ff.terminate(); ff.wait(timeout=10); shutil.rmtree(prof, ignore_errors=True)
R = {r["step"]: r for r in REP}

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

if FAILED:

    for f in FAILED:
        pass
    sys.exit(1)
