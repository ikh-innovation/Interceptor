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
    ("var vis=function(s){var e=q(s);return !!e&&e.getClientRects().length>0&&W.getComputedStyle(e).display!=='none'}; window._vis=vis; rep({step:'boot', mode:d.body.getAttribute('data-flight-mode'), card:!!q('#hdr')||!!q('#phase')||!!q('#hint'), tiles:[].filter.call(d.querySelectorAll('.tile'),function(t){return t.getClientRects().length>0}).map(function(t){return t.querySelector('.v').id}), tools:[].filter.call(d.querySelectorAll('#seeker_tools .sktool'),function(t){return t.getClientRects().length>0}).map(function(t){return t.querySelector('.tl').textContent}), acts:[].filter.call(d.querySelectorAll('#seeker_actions .big'),function(t){return t.getClientRects().length>0}).map(function(t){return t.id}), pad:vis('#normal_pad'), dial:vis('#dial'), hbox:vis('#hbox'), tape:vis('#tape'), abort:vis('#s_abort'), safedown:vis('#b_safedown'), eng:q('#s_main').className, engtxt:q('#s_maintxt').textContent, ai:q('#c_ai_k').textContent+' '+q('#c_ai').textContent, aicls:q('#c_ai').className, tgtbtn:!!q('#t_tgt'), ticker:!!q('#ticker'), menu:!!q('#t_menu'), menuTL:q('#t_menu').getBoundingClientRect().left<40 && q('#t_menu').getBoundingClientRect().top<60, bodyBg:W.getComputedStyle(d.body).backgroundColor, pills:d.querySelectorAll('.top .pill, #c_lock, #c_ctl').length, vbW:q('#vbox').getBoundingClientRect().width, vbH:q('#vbox').getBoundingClientRect().height, engR:q('#s_main').getBoundingClientRect().right, cue:q('#cue_txt')?q('#cue_txt').textContent:null, ast:q('#c_ast').textContent}); fetch('/reset');", 400),
    ("until(function(){return !q('#s_main').disabled},3000).then(function(){ return fetch('/reset'); }).then(function(){ pe(q('#s_main'),'pointerdown'); setTimeout(function(){pe(q('#s_main'),'pointerup');},80); });", 3400),
    ("cmds().then(function(c){rep({step:'engage_tap', n:c.filter(function(x){return x.indexOf('engage=')>=0}).length}); fetch('/reset');});", 300),
    ("fetch('/set?phase=AI&engaged=1');", 100),
    ("until(function(){return q('#s_main').classList.contains('flying')},3000).then(function(){ rep({step:'ai', eng:q('#s_main').className, engtxt:q('#s_maintxt').textContent, dis:q('#s_main').disabled}); fetch('/set?phase=HOVER&engaged=1'); });", 3200),
    ("until(function(){return d.body.getAttribute('data-phase')!=='AI'},3000);", 3100),
    ("rep({step:'latched', eng:q('#s_main').className, dis:q('#s_main').disabled, engtxt:q('#s_maintxt').textContent}); fetch('/set?phase=HOVER&engaged=0');", 100),
    ("until(function(){return !q('#s_main').classList.contains('flying')},3000);", 3100),
    ("rep({step:'reset', eng:q('#s_main').className, dis:q('#s_main').disabled});", 400),
    ("fetch('/set?phase=HOVER&auto=0').then(function(){ return until(function(){return q('#s_maintxt').textContent==='ENGAGE'},3000); }).then(function(){ rep({step:'auto_off', dis:q('#s_main').disabled, txt:q('#s_maintxt').textContent, auto:!!q('#b_auto')}); fetch('/set?phase=HOVER&auto=1'); });", 3200),
    ("until(function(){return q('#s_maintxt').textContent==='AUTO ENGAGE'&&!q('#s_main').disabled},3000).then(function(){ return fetch('/reset'); }).then(function(){ pe(q('#s_main'),'pointerdown'); });", 3400),
    ("cmds().then(function(c){rep({step:'auto_tap', cmds:c.map(function(x){return x.replace(/^c=\\w+&/,'')})}); fetch('/set?phase=AUTO_HOLD&auto=1');});", 400),
    ("until(function(){return d.body.getAttribute('data-phase')==='AUTO_HOLD'},3000).then(function(){ rep({step:'auto_hold', dis:q('#s_main').disabled, txt:q('#s_maintxt').textContent}); fetch('/set?phase=HOVER&auto=0'); });", 3200),
    ("q('#t_menu').click();", 400),
    ("rep({step:'menu', open:q('#sh_menu').classList.contains('open'), cards:[].map.call(d.querySelectorAll('#sh_menu .modecard'),function(b){return [b.getAttribute('data-flightmode'), !!b.querySelector('svg'), b.classList.contains('on')]}), messages:!!q('#t_log'), logo:!!q('#sh_menu .menu-logo')}); q('#sh_menu [data-close]').click(); fetch('/dead?1');", 4800),
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
check("page ran every step", "dead" in R and "error" not in R, str(R.get("error", "")))
check("no PIN anywhere: no PIN box on the page", b"pin_in" not in PAGE and b"PIN" not in PAGE)
check("the whole page is in English (no Greek letters)",
      not any(0x370 <= ord(ch) <= 0x3ff or 0x1f00 <= ord(ch) <= 0x1fff for ch in PAGE.decode("utf-8")))
check("Seeker: three status tiles, FC mode, GPS / Compass, ASTERIX", b.get("tiles") == ["c_mode", "c_sgps", "c_ast"], str(b.get("tiles")))
check("Seeker: four tools, Detection, Flight, MISSIONS, Geofence", b.get("tools") == ["Detection", "Flight", "MISSIONS", "Geofence"], str(b.get("tools")))
check("Seeker: ABORT left, the main action right (like WASP's SAFEDOWN and ARM)", b.get("acts") == ["s_abort", "s_main"]
      and b.get("safedown") is False, str(b.get("acts")))
check("Seeker: no arrow pad, no heading bar, no compass tape, no height box",
      not b.get("pad") and not b.get("dial") and not b.get("tape") and not b.get("hbox"), str(b))
check("Seeker: its own light blue", b.get("mode") == "normal" and b.get("bodyBg") == "rgb(226, 237, 247)", str(b.get("bodyBg")))
check("in the air in ALT_HOLD the right button is ENGAGE, red", "engage" in b.get("eng", "") and b.get("engtxt") == "ENGAGE", str(b))
check("no phase card (no PILOT / HOVER banner, no hint line): the picture gets the room", b.get("card") is False, str(b.get("card")))
check("no message bar under the picture; a ☰ menu top-left", b.get("ticker") is False and b.get("menu") and b.get("menuTL"), str(b))
check("no target-type button (always DRONE)", b.get("tgtbtn") is False)
check("no vibration anywhere in the page", b"vibrate" not in PAGE)
check("top right is empty: no UNLOCKED / LOCKED and no FREE / YOU CONTROL", b.get("pills") == 0, str(b.get("pills")))
check("the buttons fit the phone (the right button inside the 390 px width)", b.get("engR", 999) <= 390, str(b.get("engR")))
check("the picture stays 4:3", abs(b.get("vbW", 0) / max(1, b.get("vbH", 1)) - 4 / 3) < 0.02, f"{b.get('vbW')} x {b.get('vbH')}")
check("ASTERIX: no feed says so on the tile (no cue strip)", b.get("ast") == "NO DATA" and b.get("cue") is None, f"{b.get('ast')} / {b.get('cue')}")
check("Test mode removed from the phone app", b'id="t_test"' not in PAGE and b'/sim' not in PAGE)
check("opening the page uses the active mode without a prompt", R.get("startup", {}).get("chooser") is False and R.get("startup", {}).get("popup") is False and R.get("startup", {}).get("inert") is False)
check("ENGAGE: one quick tap = exactly one engage", R.get("engage_tap", {}).get("n") == 1, str(R.get("engage_tap")))
a = R.get("ai", {})
check("AI took over: the button green 'AI ✓', disabled",
      "flying" in a.get("eng", "") and "AI" in a.get("engtxt", "") and a.get("dis"), str(a))
l = R.get("latched", {})
check("after the attempt: still green 'DONE ✓', still disabled",
      "flying" in l.get("eng", "") and l.get("dis") and "DONE" in l.get("engtxt", ""), str(l))
check("pilot STAB -> ALT_HOLD: ENGAGE red and enabled again",
      "flying" not in R.get("reset", {}).get("eng", "flying") and R.get("reset", {}).get("dis") is False)
mn = R.get("menu", {})
check("☰ menu: the logo and two picture buttons, SEEKER (on) and WASP; no MESSAGES",
      mn.get("open") and mn.get("logo") and mn.get("cards") == [["normal", True, True], ["wasp", True, False]]
      and mn.get("messages") is False, str(mn))
ao, at, ah = R.get("auto_off", {}), R.get("auto_tap", {}), R.get("auto_hold", {})
check("Semi-auto keeps Engage ready", ao.get("dis") is False and ao.get("txt") == "ENGAGE" and ao.get("auto") is False, str(ao))
check("AUTO ENGAGE sends one auto=on", at.get("cmds") == ["auto=on"], str(at))
check("AUTO keeps Auto Engage available", ah.get("txt") == "AUTO ENGAGE" and ah.get("dis") is False, str(ah))
check("link dead: the red cover shows", R.get("dead", {}).get("cover") is True)

DEAD["on"] = False
SIZES = [("small phone", 320, 568), ("phone 360", 360, 640), ("phone 390", 390, 844), ("tablet upright", 820, 1180), ("tablet upright small", 768, 1024), ("tablet sideways", 1180, 820),
         ("tablet sideways small", 1024, 768), ("laptop", 1366, 657), ("desktop", 1920, 1080)]
LAYOUT = ("""<!doctype html><html><body style="margin:0">%s<script>
function rep(o){fetch('/report?'+encodeURIComponent(JSON.stringify(o)));}
function box(e){var r=e.getBoundingClientRect();return {l:r.left,t:r.top,r:r.right,b:r.bottom,w:r.width,h:r.height};}
[].forEach.call(document.querySelectorAll('iframe'), function(f){ f.onload=function(){ var d=f.contentDocument, W=f.contentWindow,
 q=function(s){return d.querySelector(s)};
 (function ready(){ if(!d.body.getAttribute('data-phase')) return setTimeout(ready,50);
  setTimeout(function(){ var all=[].filter.call(d.querySelectorAll('.app button, .tile, #vbox'),function(e){return e.getClientRects().length>0}).map(box);
   rep({step:'layout_'+f.id, vw:W.innerWidth, vh:W.innerHeight, vb:box(q('#vbox')), tiles:box(q('.tiles')),
     tileTops:[].filter.call(d.querySelectorAll('.tile'),function(t){return t.getClientRects().length>0}).map(function(t){return box(t).t}),
     ui:box(q('#seeker_ui')), tools:[].filter.call(d.querySelectorAll('#seeker_tools .sktool'),function(t){return t.getClientRects().length>0}).map(box),
     acts:[].filter.call(d.querySelectorAll('#seeker_actions .big'),function(t){return t.getClientRects().length>0}).map(box),
     top:box(q('.brand')), logo:box(q('.logo')), logoOk:q('.logo').complete && q('.logo').naturalWidth+q('.logo').naturalHeight>0,
     out:all.filter(function(r){return r.w>0&&(r.l<-1||r.t<-1||r.r>W.innerWidth+1||r.b>W.innerHeight+1);}).length,
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
    vb, tl = L["vb"], L["tiles"]
    check(f"{name} {w}x{h}: every control, tile and the picture inside the screen, no sideways scroll",
          L["out"] == 0 and L["sw"] <= w, f"out {L['out']} scrollW {L['sw']}")
    check(f"{name}: the picture stays 4:3", abs(vb["w"] / vb["h"] - 4 / 3) < 0.02, f"{vb['w']:.0f}x{vb['h']:.0f}")
    check(f"{name}: three status tiles in one row", len(L["tileTops"]) == 3 and max(L["tileTops"]) - min(L["tileTops"]) < 1,
          str(L["tileTops"]))
    check(f"{name}: the IKnowHow Defence logo loads", L["logoOk"])
    side = w >= 700 or (w >= 560 and w > h)
    ui = L["ui"]
    check(f"{name}: the Seeker panel {'beside' if side else 'under'} the picture",
          ui["l"] >= vb["r"] - 1 if side else ui["t"] >= vb["b"] - 1, f"panel {ui}, picture {vb}")
    check(f"{name}: four tools and the two big buttons, all big enough to tap",
          len(L["tools"]) == 4 and len(L["acts"]) == 2 and all(t["h"] >= 44 for t in L["tools"] + L["acts"]),
          " ".join(f"{t['w']:.0f}x{t['h']:.0f}" for t in L["tools"] + L["acts"]))

if FAILED:

    for f in FAILED:
        print("FAIL", f)
    sys.exit(1)
