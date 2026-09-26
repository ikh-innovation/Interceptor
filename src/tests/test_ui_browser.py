#!/usr/bin/env python3
"""The phone app (src/mission_ui.html) driven in a REAL browser.

Serves the page from a fake server that records every /cmd, opens it in
headless Firefox inside a same-origin harness, synthesises pointer events and
reads the DOM back.  SKIPS (exit 0, says so) when Firefox is not installed.

    ../.env/yolo/bin/python tests/test_ui_browser.py
"""
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
TOKEN = "424242"
FAILED = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


if not shutil.which("firefox"):
    print("  [SKIP] firefox not installed -- browser test not run")
    sys.exit(0)

ST = dict(linked=True, armed=True, fc_mode="ALT_HOLD", target="DRONE", tilt_deg=30.0,
          detector="TensorRT", fps=28.0,
          mission=dict(app="mission", phase="HOVER", engaged=False, ai_steering=False,
                       levels=dict(vert="SLOW", yaw="MEDIUM", push="SLOW", ai="MEDIUM"),
                       speeds=dict(climb=1.0, descent=0.5, yaw=20, step=5, push=20), min_conf=0.2,
                       height_m=10.2, climb_mps=0.0, heading_deg=123,
                       vibe=dict(x=9, y=8, z=14, clip=0), det_conf=0.62, det_age_s=0.05,
                       target_fresh=True, target_az_deg=1.2, target_el_deg=-18.5),
          console_seq=2)
CONSOLE = [[1, "12:00:00", "[MISSION] PILOT -> HOVER  (armed in ALT_HOLD)"],
           [2, "12:00:01", "[FC] override CONFIRMED"]]
CMDS, REP = [], []
SLOW, DEAD = {"on": False}, {"on": False}

# Each step: JS run in the harness, then a wait (ms).  rep(...) sends a report.
STEPS = [
    ("rep({step:'boot', phase:q('#phase').textContent, alt:q('#alt').textContent,"
     " abort:!!q('#b_abort'), eng:q('#b_eng').className, engtxt:q('#engtxt').textContent,"
     " aimTop:q('#aim').style.top, badge:q('#badge').textContent}); fetch('/reset');", 400),
    ("pe(q('#b_up'),'pointerdown');", 1200),
    ("pe(q('#b_up'),'pointerup');", 300),
    ("cmds().then(function(c){rep({step:'hold_up', n:c.filter(function(x){return x.indexOf('hold=up')>=0}).length,"
     " rel:c.some(function(x){return x.indexOf('release=')>=0}), pin:c.every(function(x){return x.indexOf('k=424242')>=0})});"
     " fetch('/reset');});", 300),
    ("pe(q('#b_yl'),'pointerdown'); setTimeout(function(){pe(q('#b_yl'),'pointerup');},120);", 500),
    ("cmds().then(function(c){rep({step:'tap', cmds:c.map(function(x){return x.replace(/k=\\d+&c=\\w+&/,'')})}); fetch('/reset');});", 300),
    ("pe(q('#b_yr'),'pointerdown');", 700),
    ("W.dispatchEvent(new W.Event('blur'));", 600),
    ("cmds().then(function(c){rep({step:'blur', n:c.filter(function(x){return x.indexOf('hold=yawr')>=0}).length,"
     " rel:c.some(function(x){return x.indexOf('release=')>=0})}); fetch('/reset');});", 700),
    ("cmds().then(function(c){rep({step:'after_blur', n:c.filter(function(x){return x.indexOf('hold=')>=0}).length}); fetch('/reset');});", 300),
    ("pe(q('#b_eng'),'pointerdown'); setTimeout(function(){pe(q('#b_eng'),'pointerup');},80);", 500),
    ("cmds().then(function(c){rep({step:'engage_tap', n:c.filter(function(x){return x.indexOf('engage=')>=0}).length}); fetch('/reset');});", 300),
    ("fetch('/set?phase=AI&engaged=1');", 700),
    ("rep({step:'ai', eng:q('#b_eng').className, engtxt:q('#engtxt').textContent, dis:q('#b_eng').disabled,"
     " up:q('#b_up').disabled, hdr:q('#hdr').style.background}); fetch('/set?phase=HOVER&engaged=1');", 700),
    ("rep({step:'latched', eng:q('#b_eng').className, dis:q('#b_eng').disabled, hint:q('#hint').textContent});"
     " fetch('/set?phase=HOVER&engaged=0');", 700),
    ("rep({step:'reset', eng:q('#b_eng').className, dis:q('#b_eng').disabled}); q('#t_speed').click();", 400),
    ("var sh=q('#sh_speed'), cs=W.getComputedStyle(sh); rep({step:'speeds', open:sh.classList.contains('open'),"
     " pos:cs.position, h:sh.getBoundingClientRect().height, winh:W.innerHeight});"
     " q('.seg[data-g=vert] button[data-lv=FAST]').click();", 500),
    ("cmds().then(function(c){rep({step:'lvl', cmds:c.map(function(x){return x.replace(/k=\\d+&c=\\w+&/,'')})});"
     " q('#sh_speed [data-close]').click(); rep({step:'closed', open:q('#sh_speed').classList.contains('open')}); fetch('/reset');});", 400),
    ("q('#t_log').click();", 400),
    ("rep({step:'console', lines:[].map.call(d.querySelectorAll('#log .ln'),function(l){return l.textContent})});"
     " q('#sh_log [data-close]').click(); fetch('/set?phase=PUSH&engaged=1');", 700),
    ("rep({step:'push', up:q('#b_up').disabled, eng:q('#b_eng').disabled, aimTop:q('#aim').style.top});"
     " fetch('/set?phase=HOVER&engaged=0'); fetch('/dead?1');", 1900),
    ("rep({step:'dead', cover:q('#cover').classList.contains('on'), done:1});", 0),
]
HARNESS = ("""<!doctype html><html><body><iframe id=f src="/?k=424242" width=390 height=844></iframe><script>
function rep(o){fetch('/report?'+encodeURIComponent(JSON.stringify(o)));}
var f=document.getElementById('f');
f.onload=function(){ var d=f.contentDocument, W=f.contentWindow, q=function(s){return d.querySelector(s)};
 function pe(el,t){ el.dispatchEvent(new W.PointerEvent(t,{bubbles:true,cancelable:true,pointerId:1,isPrimary:true})); }
 function cmds(){ return fetch('/cmds').then(function(r){return r.json()}); }
 var S=%s; var i=0;
 (function go(){ if(i>=S.length) return; try{ eval(S[i][0]); }catch(e){ rep({step:'error',err:String(e),at:i}); }
   var w=S[i][1]; i++; setTimeout(go, w); })(); };
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
        if p == "/stream.mjpg": return self.send_error(404)
        if DEAD["on"] and p in ("/state", "/cmd", "/console"):
            time.sleep(4); return
        if p == "/state":
            if SLOW["on"]: time.sleep(0.8)
            return self._j(dict(ST, control="you"))
        if p == "/cmd":
            CMDS.append(q)
            if parse_qs(q).get("k", [""])[0] != TOKEN: return self._j(dict(error="bad PIN"), 403)
            return self._j(dict(ST, control="you", refused=""))
        if p == "/console":
            since = int(parse_qs(q).get("since", ["0"])[0]); return self._j([r for r in CONSOLE if r[0] > since])
        if p == "/cmds": return self._j(CMDS)
        if p == "/reset": CMDS.clear(); return self._j({})
        if p == "/dead": DEAD["on"] = q == "1"; return self._j({})
        if p == "/set":
            a = parse_qs(q)
            ST["mission"]["phase"] = a.get("phase", [ST["mission"]["phase"]])[0]
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
check("ENGAGE starts red and says 'πάτα' (instant)", "go" not in b.get("eng", "go") and "πάτα" in b.get("engtxt", ""))
h = R.get("hold_up", {})
check("hold ▲ 1.2 s: ~10 keepalives, then a release, every command with the PIN",
      8 <= h.get("n", 0) <= 14 and h.get("rel") and h.get("pin"), str(h))
check("a 120 ms tap on ⟲ is ONE yaw step, not a hold", R.get("tap", {}).get("cmds") == ["step=l"], str(R.get("tap")))
check("holding ⟳ then the page losing focus: release sent, keepalives stop",
      R.get("blur", {}).get("rel") and R.get("after_blur", {}).get("n") == 0, f"{R.get('blur')} {R.get('after_blur')}")
check("ENGAGE: one quick tap = exactly one engage", R.get("engage_tap", {}).get("n") == 1, str(R.get("engage_tap")))
a = R.get("ai", {})
check("AI took over: ENGAGE green 'AI ✓', disabled; manual buttons disabled; header red",
      "go" in a.get("eng", "") and "AI" in a.get("engtxt", "") and a.get("dis") and a.get("up")
      and "--ai" in a.get("hdr", ""), str(a))
l = R.get("latched", {})
check("after the attempt: still green, still disabled, hint says STAB -> ALT_HOLD",
      "go" in l.get("eng", "") and l.get("dis") and "STAB" in l.get("hint", ""), str(l))
check("pilot STAB -> ALT_HOLD: ENGAGE red and enabled again",
      "go" not in R.get("reset", {}).get("eng", "go") and R.get("reset", {}).get("dis") is False)
sp = R.get("speeds", {})
check("ΤΑΧΥΤΗΤΕΣ opens FULL-SCREEN", sp.get("open") and sp.get("pos") == "fixed"
      and sp.get("h", 0) >= 0.95 * sp.get("winh", 1), str(sp))
check("a speed button sends lvl=vert:FAST; ✕ closes the page",
      R.get("lvl", {}).get("cmds") == ["lvl=vert:FAST"] and R.get("closed", {}).get("open") is False)
check("console shows the tracker's printed lines", len(R.get("console", {}).get("lines", [])) == 2)
pu = R.get("push", {})
check("PUSH: ▲ and ENGAGE disabled; the aim mark moves to the centre", pu.get("up") and pu.get("eng")
      and pu.get("aimTop") == "50%", str(pu))
check("link dead: the red cover shows", R.get("dead", {}).get("cover") is True)
print()
if FAILED:
    print(f"{len(FAILED)} FAILED:")
    for f in FAILED:
        print("  - " + f)
    sys.exit(1)
print("all passed")
