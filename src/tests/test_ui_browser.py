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
check("Seeker actions are CHECK and GO",
      b'id="b_rfd"' in PAGE and b'id="b_go"' in PAGE and b'id="rfd_read"' in PAGE
      and b'id="b_lock"' not in PAGE and b'id="b_up"' not in PAGE and b'id="ptab_auto"' not in PAGE)
check("the whole page is in English (no Greek letters)",
      not any(0x370 <= ord(ch) <= 0x3ff or 0x1f00 <= ord(ch) <= 0x1fff for ch in PAGE.decode("utf-8")))
check("no PIN anywhere: no PIN box on the page", b"pin_in" not in PAGE and b"PIN" not in PAGE)
check("no pure black left in the page's styles (the video's own background aside)",
      b"#15181c" not in PAGE and b"#101317" not in PAGE and PAGE.count(b"#000") == 1)

if FAILED:
    for f in FAILED:
        print("FAIL", f)
    sys.exit(1)

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
    ("rep({step:'boot', phase:d.body.getAttribute('data-phase'), alt:q('#alt').textContent, card:!!q('#hdr')||!!q('#phase')||!!q('#hint'), hbL:q('#hbox').getBoundingClientRect().left, hbT:q('#hbox').getBoundingClientRect().top, hbBg:W.getComputedStyle(q('#hbox')).backgroundColor, dialBg:W.getComputedStyle(q('#dial')).backgroundColor, dialT:q('#dial').getBoundingClientRect().top, dialR:q('#dial').getBoundingClientRect().right, dialW:q('#dial').getBoundingClientRect().width, dialH:q('#dial').getBoundingClientRect().height, dialOff:q('#dial').classList.contains('off'), vbB:q('#vbox').getBoundingClientRect().bottom, vbL:q('#vbox').getBoundingClientRect().left, vbR:q('#vbox').getBoundingClientRect().right, vbH:q('#vbox').getBoundingClientRect().height, vbW:q('#vbox').getBoundingClientRect().width, chkL:q('#b_rfd').getBoundingClientRect().left, chkT:q('#b_rfd').getBoundingClientRect().top, chkB:q('#b_rfd').getBoundingClientRect().bottom, chkR:q('#b_rfd').getBoundingClientRect().right, chkW:q('#b_rfd').getBoundingClientRect().width, chkH:q('#b_rfd').getBoundingClientRect().height, chkBg:W.getComputedStyle(q('#b_rfd')).backgroundColor, chkTxt:q('#b_rfd').textContent.trim(), goL:q('#b_go').getBoundingClientRect().left, goT:q('#b_go').getBoundingClientRect().top, goR:q('#b_go').getBoundingClientRect().right, goW:q('#b_go').getBoundingClientRect().width, goH:q('#b_go').getBoundingClientRect().height, goTxt:q('#gotxt').textContent, goCls:q('#b_go').className, goDis:q('#b_go').disabled, goBg:W.getComputedStyle(q('#b_go')).backgroundColor, padL:q('#normal_pad').getBoundingClientRect().left, abort:q('#b_safedown').getClientRects().length>0, aimTop:q('#aim').style.top, badge:q('#badge').textContent, ai:q('#c_ai_k').textContent+' '+q('#c_ai').textContent, aicls:q('#c_ai').className, tgtbtn:!!q('#t_tgt'), goInVid:q('#vbox').contains(q('#b_go')), goRightOfPad:q('#b_go').getBoundingClientRect().left>=q('#normal_pad').getBoundingClientRect().right, ticker:!!q('#ticker'), tools:!!q('.tools'), menu:!!q('#t_menu'), menuTL:q('#t_menu').getBoundingClientRect().left<40 && q('#t_menu').getBoundingClientRect().top<60, bodyBg:W.getComputedStyle(d.body).backgroundColor, pills:d.querySelectorAll('.top .pill, #c_lock, #c_ctl').length, buttons:[].filter.call(d.querySelectorAll('.acts .rbtn'),function(b){return b.getClientRects().length>0}).map(function(b){return b.id}), rfd:q('#rfd_read').textContent, engHidden:q('#b_eng').hidden}); fetch('/reset');", 400),
    ("var cv=q('#tape'), vb=q('#vbox').getBoundingClientRect(), r=cv.getBoundingClientRect(); window._ink=function(x0,x1){var c=cv.getContext('2d'), d=c.getImageData(Math.round(cv.width*x0),0,Math.max(1,Math.round(cv.width*(x1-x0))),cv.height).data, n=0; for(var i=3;i<d.length;i+=4) if(d[i]>100) n++; return n;}; rep({step:'tape', top:r.top-vb.top, wdiff:Math.abs(r.width-vb.width), h:r.height, left:_ink(0,0.35), right:_ink(0.65,1)}); fetch('/set?phase=HOVER&heading=352');", 900),
    ("rep({step:'tape352', left:_ink(0,0.35), right:_ink(0.65,1)}); fetch('/set?phase=HOVER&heading=123'); fetch('/reset');", 500),
    ("until(function(){return !q('#b_rfd').disabled},3000).then(function(){ return fetch('/reset'); }).then(function(){ pe(q('#b_rfd'),'pointerdown'); });", 3400),
    ("cmds().then(function(c){rep({step:'check', cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})}); fetch('/reset');});", 400),
    ("until(function(){return !q('#b_go').disabled && q('#gotxt').textContent==='GO'},3000).then(function(){ return fetch('/reset'); }).then(function(){ pe(q('#b_go'),'pointerdown'); });", 3400),
    ("cmds().then(function(c){rep({step:'go', cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})}); fetch('/reset');});", 400),
    ("var dl=q('#dial'); pey(dl,'pointerdown',300); pey(dl,'pointermove',200); pey(dl,'pointerup',200); setTimeout(function(){ rep({step:'dial', ask:q('#ask').classList.contains('on'), off:dl.classList.contains('off')}); }, 250);", 700),
    ("cmds().then(function(c){rep({step:'dial_cmds', cmds:c}); fetch('/reset'); fetch('/set?phase=GPS_GOTO&fresh=0');});", 400),
    ("until(function(){return d.body.getAttribute('data-phase')==='GPS_GOTO'},3000).then(function(){ rep({step:'gps', txt:q('#gotxt').textContent, dis:q('#b_go').disabled, cls:q('#b_go').className, badge:q('#badge').textContent, aim:q('#aim').style.top, label:q('#b_go').getAttribute('aria-label')}); return fetch('/reset'); }).then(function(){ pe(q('#b_go'),'pointerdown'); });", 3400),
    ("cmds().then(function(c){rep({step:'stop', cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})}); fetch('/set?phase=AI&engaged=1&fresh=1');});", 400),
    ("until(function(){return q('#b_go').disabled && q('#aim').style.top==='50%'},3000).then(function(){ rep({step:'ai', dis:q('#b_go').disabled, txt:q('#gotxt').textContent, aim:q('#aim').style.top}); fetch('/set?phase=HOVER&engaged=0&fresh=1'); });", 3200),
    ("until(function(){return !q('#b_go').disabled && q('#gotxt').textContent==='GO'},3000).then(function(){ rep({step:'hover', dis:q('#b_go').disabled, txt:q('#gotxt').textContent, badge:q('#badge').textContent}); });", 3200),
    ("q('#t_log').click();", 400),
    ("rep({step:'console', lines:[].map.call(d.querySelectorAll('#log .ln'),function(l){return l.textContent})}); var lg=q('#sh_log'); rep({step:'logsheet', open:lg.classList.contains('open'), pos:W.getComputedStyle(lg).position, h:lg.getBoundingClientRect().height, winh:W.innerHeight}); q('#sh_log [data-close]').click(); rep({step:'logclosed', open:q('#sh_log').classList.contains('open')}); q('#sh_menu [data-close]').click(); fetch('/set?phase=PUSH&engaged=1');", 100),
    ("until(function(){return d.body.getAttribute('data-phase')==='PUSH'},3000).then(function(){ rep({step:'push', dis:q('#b_go').disabled, aim:q('#aim').style.top}); rep({step:'push_dial', off:q('#dial').classList.contains('off')}); var dl=q('#dial'); pey(dl,'pointerdown',300); pey(dl,'pointermove',200); pey(dl,'pointerup',200); setTimeout(function(){ rep({step:'push_dial2', ask:q('#ask').classList.contains('on')}); fetch('/set?phase=HOVER&engaged=0'); fetch('/dead?1'); }, 200); });", 4800),
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
            if "fresh" in a:
                ST["mission"]["target_fresh"] = a["fresh"][0] == "1"
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
check("GO starts red and says GO", "push" not in b.get("goCls", "push") and b.get("goTxt") == "GO"
      and b.get("goBg") == "rgb(214, 40, 40)" and b.get("goDis") is False, str(b))
check("the hidden WASP launch button stays off the Seeker screen", b.get("engHidden") is True and b.get("buttons") == ["b_rfd", "b_go"], str(b))
check("no phase card (no PILOT / HOVER banner, no hint line): the picture gets the room", b.get("card") is False, str(b.get("card")))
check("CHECK / GO right of the RFD card, not in the picture", not b.get("goInVid") and b.get("goRightOfPad"), str(b))
check("CHECK right ABOVE GO, one column (same left edge)",
      b.get("chkB", 999) <= b.get("goT", 0) and abs(b.get("chkL", 0) - b.get("goL", 99)) <= 1,
      str({k: b.get(k) for k in ("chkL", "chkB", "goL", "goT")}))
check("both a bit bigger: 74 px round (was 66)",
      72 <= b.get("goW", 0) <= 76 and 72 <= b.get("chkW", 0) <= 76
      and abs(b.get("goW", 0) - b.get("goH", 99)) <= 1 and abs(b.get("chkW", 0) - b.get("chkH", 99)) <= 1,
      f"{b.get('goW')}x{b.get('goH')} / {b.get('chkW')}x{b.get('chkH')}")
check("no message bar and no speed/message buttons under the picture; a ☰ menu top-left",
      b.get("ticker") is False and b.get("tools") is False and b.get("menu") and b.get("menuTL"), str(b))
check("light theme: light-grey page; heading bar and height box a neutral GREY (no black)",
      b.get("bodyBg") == "rgb(236, 238, 241)" and b.get("dialBg") == "rgb(86, 93, 102)"
      and b.get("hbBg") == b.get("dialBg"),
      f"{b.get('bodyBg')} / {b.get('dialBg')} / {b.get('hbBg')}")
check("no pure black left in the page's styles (the video's own background aside)",
      b"#15181c" not in PAGE and b"#101317" not in PAGE and PAGE.count(b"#000") == 1)
check("no target-type button (always DRONE)", b.get("tgtbtn") is False)
check("the RFD card sits at the left edge, GO at the right edge (inside their 8 px plate)",
      b.get("padL", 99) <= 16 and b.get("goR", 0) >= 390 - 24, f"pad left {b.get('padL')}, GO right {b.get('goR')}")
tp, t3 = R.get("tape", {}), R.get("tape352", {})
check("compass tape: full width, along the top edge of the picture, ~26 px high",
      abs(tp.get("top", 99)) <= 2 and tp.get("wdiff", 99) <= 3 and 20 <= tp.get("h", 0) <= 30, str(tp))
check("compass tape draws marks on both sides at 123 deg, and across north at 352 deg",
      min(tp.get("left", 0), tp.get("right", 0), t3.get("left", 0), t3.get("right", 0)) > 25,
      f"{tp} {t3}")
check("CHECK: a white button of the same size as GO",
      abs(b.get("chkW", 0) - b.get("goW", 99)) <= 1 and b.get("chkBg") == "rgb(255, 255, 255)"
      and b.get("chkTxt") == "CHECK", str({k: b.get(k) for k in ("chkW", "goW", "chkBg", "chkTxt")}))
check("CHECK pressed: sends rfd=1", R.get("check", {}).get("cmds") == ["rfd=1"], str(R.get("check")))
check("GO pressed: sends go=on", R.get("go", {}).get("cmds") == ["go=on"], str(R.get("go")))
check("no vibration anywhere in the page", b"vibrate" not in PAGE)
check("top right is empty: no UNLOCKED / LOCKED and no FREE / YOU CONTROL", b.get("pills") == 0, str(b.get("pills")))
check("the buttons fit the phone (GO inside the 390 px width)", b.get("goR", 999) <= 390 and b.get("chkR", 999) <= 390,
      str((b.get("goR"), b.get("chkR"))))
check("heading bar: BELOW the picture, from its left edge, ~46 px high, and the lock gesture is off",
      b.get("dialT", 0) >= b.get("vbB", 999) and abs(b.get("vbL", 0) - (b.get("dialR", 0) - b.get("dialW", 0))) <= 2
      and 40 <= b.get("dialH", 0) <= 50 and b.get("dialW", 0) > 180 and b.get("dialOff") is True,
      str({k: b.get(k) for k in ("dialT", "vbB", "vbL", "dialR", "dialW", "dialH", "dialOff")}))
check("height box: right of the bar, same colour, shows the height (10.2 m)",
      b.get("hbL", 0) >= b.get("dialR", 999) and abs(b.get("hbT", 0) - b.get("dialT", 0)) <= 2
      and b.get("hbBg") == b.get("dialBg") and b.get("alt") == "10.2 m",
      str({k: b.get(k) for k in ("hbL", "dialR", "hbT", "dialT", "hbBg", "dialBg", "alt")}))
check("the picture stays 4:3", abs(b.get("vbW", 0) / max(1, b.get("vbH", 1)) - 4 / 3) < 0.02,
      f"{b.get('vbW')} x {b.get('vbH')}")
check("a drag on the heading bar asks nothing and sends nothing",
      R.get("dial", {}).get("ask") is False and R.get("dial", {}).get("off") is True and R.get("dial_cmds", {}).get("cmds") == [],
      f"{R.get('dial')} {R.get('dial_cmds')}")
gps = R.get("gps", {})
check("on the way to the RFD point, GO becomes STOP and the badge says where it is going",
      gps.get("txt") == "STOP" and gps.get("label") == "STOP" and "push" in gps.get("cls", "")
      and gps.get("dis") is False and gps.get("badge") == "GPS · GO TO" and gps.get("aim") != "50%", str(gps))
check("STOP sends go=off", R.get("stop", {}).get("cmds") == ["go=off"], str(R.get("stop")))
a = R.get("ai", {})
check("AI took over: GO is disabled and the aim mark is in the centre",
      a.get("dis") is True and a.get("txt") == "GO" and a.get("aim") == "50%", str(a))
check("back in HOVER: GO is ready again and a centred target reads ON TARGET",
      R.get("hover", {}).get("dis") is False and R.get("hover", {}).get("txt") == "GO"
      and "ON TARGET" in R.get("hover", {}).get("badge", ""), str(R.get("hover")))
check("PUSH: the degree scroll is off and a drag asks nothing",
      R.get("push_dial", {}).get("off") and R.get("push_dial2", {}).get("ask") is False,
      f"{R.get('push_dial')} {R.get('push_dial2')}")
check("Test mode removed from the phone app", b'id="t_test"' not in PAGE and b'/sim' not in PAGE)
check("opening the page uses the active mode without a prompt", R.get("startup", {}).get("chooser") is False and R.get("startup", {}).get("popup") is False and R.get("startup", {}).get("inert") is False)
check("console shows the tracker's printed lines", len(R.get("console", {}).get("lines", [])) == 2)
lg = R.get("logsheet", {})
check("MESSAGES opens FULL-SCREEN (not a pop-up) and ✕ closes it",
      lg.get("open") and lg.get("pos") == "fixed" and lg.get("h", 0) >= 0.95 * lg.get("winh", 1)
      and R.get("logclosed", {}).get("open") is False, str(lg))
pu = R.get("push", {})
check("PUSH: GO is disabled and the aim mark moves to the centre", pu.get("dis") is True and pu.get("aim") == "50%", str(pu))
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
     tile0:box(q('.tile')), tile5:box(d.querySelectorAll('.tile')[5]), chk:box(q('#b_rfd')), go:box(q('#b_go')), pad:box(q('#normal_pad')), belt:box(q('.belt')), top:box(q('.brand')), logo:box(q('.logo')), logoOk:q('.logo').complete && q('.logo').naturalWidth+q('.logo').naturalHeight>0, logoNat:[q('.logo').complete,q('.logo').naturalWidth,q('.logo').naturalHeight], inVb:q('#vbox').contains(q('.ctl')), plates:[W.getComputedStyle(q('#normal_pad')).backgroundColor, W.getComputedStyle(q('.acts')).backgroundColor],
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
    pad, chk, go = L["pad"], L["chk"], L["go"]
    check(f"{name}: one line on top: ☰ + name, then the six status cells in one row; the picture under it",
          abs(L["tile0"]["t"] - L["tile5"]["t"]) < 1 and tl["l"] >= L["top"]["r"] and abs((tl["t"] + tl["b"]) - (L["top"]["t"] + L["top"]["b"])) < 4
          and tl["b"] <= vb["t"], f"top right {L['top']['r']:.0f}, tiles {tl['l']:.0f}..{tl['r']:.0f}")
    check(f"{name}: the IKnowHow Defence logo loads, at the far right of the top line",
          L["logoOk"] and L["logo"]["l"] >= tl["r"] and L["logo"]["r"] <= w and abs((L["logo"]["t"] + L["logo"]["b"]) - (tl["t"] + tl["b"])) < 4,
          f"logo {L['logo']['l']:.0f}..{L['logo']['r']:.0f} t{L['logo']['t']:.0f}-{L['logo']['b']:.0f}, tiles t{tl['t']:.0f}-{tl['b']:.0f} end {tl['r']:.0f} nat {L['logoNat']}")
    check(f"{name}: the picture as big as it fits (full width or full height)",
          vb["w"] >= w - 50 or vb["b"] >= h - 120, f"{vb['w']:.0f}x{vb['h']:.0f} of {w}x{h}")
    check(f"{name}: the heading bar right under the picture", 0 <= L["belt"]["t"] - vb["b"] <= 10, f"{L['belt']['t'] - vb['b']:.0f}")
    check(f"{name}: the RFD card INSIDE the picture, bottom-left",
          L["inVb"] and pad["l"] >= vb["l"] and pad["l"] - vb["l"] <= 40 and pad["b"] <= vb["b"] and vb["b"] - pad["b"] <= 40,
          f"card {pad['l'] - vb['l']:.0f} from the left, {vb['b'] - pad['b']:.0f} from the bottom")
    check(f"{name}: CHECK above GO INSIDE the picture, bottom-right",
          go["r"] <= vb["r"] and vb["r"] - go["r"] <= 50 and go["b"] <= vb["b"] and vb["b"] - go["b"] <= 50
          and chk["b"] <= go["t"] + 1 and abs(chk["l"] - go["l"]) <= 1, f"GO {vb['r'] - go['r']:.0f} from the right")
    check(f"{name}: CHECK and GO are the same round size, at least 48 px",
          abs(chk["w"] - go["w"]) < 1 and abs(chk["h"] - go["h"]) < 1
          and abs(go["w"] - go["h"]) < 1 and go["h"] >= 48,
          f"CHECK {chk['w']:.0f}x{chk['h']:.0f} GO {go['w']:.0f}x{go['h']:.0f}")
    check(f"{name}: a dark plate behind the RFD card and behind CHECK / GO",
          all(c.startswith("rgba(14, 17, 21") for c in L["plates"]), str(L["plates"]))

if FAILED:
    for f in FAILED:
        print("FAIL", f)
    sys.exit(1)
print("ok", "ui")
