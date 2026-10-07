"""Real Firefox checks of mode selection, refresh and Wasp controls."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission import Mission, phone_command
from browser_support import available, launch

ROOT = Path(__file__).resolve().parents[1]


class Mapper:
    armed = False
    camera_mount_deg = 90

    def wasp_pwm(self, rates, throttle):
        return {}


mapper = Mapper()
m = Mission(mapper, log=lambda *_: None)
commands, reports = [], []
detector_settings = dict(sky_gate=True, follower=True, follower_default=True)
done = threading.Event()
lock = threading.Lock()


def state():
    s = m.snapshot(time.perf_counter())
    s.update(gate_why="" if mapper.armed else "disarmed", height_m=0,
             climb_mps=0, heading_deg=25, det_conf=.9, det_age_s=.01,
             target_fresh=True, target_az_deg=0, target_el_deg=0)
    return dict(mission=s, armed=mapper.armed, linked=True,
                fc_mode="ACRO" if m.wasp.enabled else "ALT_HOLD", control="you",
                detector="TensorRT", fps=30, **detector_settings)


HARNESS = r'''<!doctype html><html><head><meta charset="utf-8"></head><body><iframe id="f" width="390" height="844"></iframe><script>
const f=document.getElementById('f');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const req=async(path)=>{const r=await fetch(path);return r.json()};
const q=s=>f.contentDocument.querySelector(s);
const visible=s=>q(s)&&q(s).getClientRects().length>0&&f.contentWindow.getComputedStyle(q(s)).display!=='none';
async function check(name,ok){await req('/report?data='+encodeURIComponent(JSON.stringify({name,ok:!!ok})));if(!ok)throw Error(name)}
async function until(fn){for(let i=0;i<80;i++){if(fn())return;await sleep(50)}throw Error('timeout: '+fn)}
async function load(){f.src='/?t='+Date.now();await new Promise(r=>f.onload=r);await until(()=>!q('#choose_wasp').disabled || !q('#choose_normal').disabled)}
function tap(s){q(s).dispatchEvent(new f.contentWindow.PointerEvent('pointerdown',{bubbles:true,cancelable:true,pointerId:1}))}
async function main(){
 await load();
 await check('chooser on first open',visible('#modechooser')&&q('.app').inert);
 await check('no test controls or simulator canvas',!q('#t_test')&&!q('#simv')&&!q('#sh_test'));
 q('#choose_normal').click(); await until(()=>!visible('#modechooser'));
 await check('normal parameters only',!visible('#ptab_wasp')&&visible('#ptab_auto')&&visible('#ptab_semi'));
 q('[data-mount="30"]').click(); await until(()=>q('#mountstatus').textContent.includes('30°'));
 q('[data-autoen="on"]').click(); await until(()=>q('#autotag').textContent==='ON');
 q('#sh_speed [data-close]').click();q('#t_menu').click();q('#t_mode').click();q('#choose_wasp').click();
 await until(()=>!visible('#modechooser')&&q('#wasptag').textContent==='ON');
 await check('Wasp selection disables Auto and shows Wasp parameters',q('#autotag').textContent==='OFF'&&visible('#ptab_wasp')&&!visible('#ptab_auto')&&!visible('#ptab_semi'));
 await check('90 degree reminder',q('#mountstatus').textContent.includes('confirm the 90°'));
 await check('launch defaults and throttle choices',q('[data-wp="straight"] [data-wv="1"]').classList.contains('on')&&q('[data-wp="blend"] [data-wv="0.5"]').classList.contains('on')&&Array.from(q('[data-wp="throttle"]').children).map(b=>b.textContent).join(',')==='25%,50%,100%');
 q('[data-wp="straight"] [data-wv="0.5"]').click();await until(()=>q('[data-wp="straight"] [data-wv="0.5"]').classList.contains('on'));
 q('[data-wp="blend"] [data-wv="0.25"]').click();await until(()=>q('[data-wp="blend"] [data-wv="0.25"]').classList.contains('on'));
 await check('straight delay and blend editable', (await req('/state')).mission.wasp_params.straight===0.5&&(await req('/state')).mission.wasp_params.blend===0.25);
 q('[data-wp="straight"] [data-wv="1"]').click();q('[data-wp="blend"] [data-wv="0.5"]').click();
 q('#ptab_det').click();await until(()=>visible('#skyseg')&&visible('#folseg'));
 await check('ground rejection and follower visible in Wasp',visible('#ptab_det')&&visible('#skyseg')&&visible('#folseg'));
 q('[data-sky="off"]').click();await until(()=>q('[data-sky="off"]').classList.contains('on'));
 q('[data-fol="off"]').click();await until(()=>q('[data-fol="off"]').classList.contains('on'));
 await check('Wasp ground and follower toggles reach backend',!(await req('/state')).sky_gate&&!(await req('/state')).follower);
 q('#defask .no').click();q('#ptab_wasp').click();
 q('[data-wp="throttle"] [data-wv="0.25"]').click();await until(()=>q('[data-wp="throttle"] [data-wv="0.25"]').classList.contains('on'));
 q('[data-mount="90"]').click();await until(()=>q('#mountstatus').textContent.includes('90° confirmed'));
 await load();await check('refresh asks again',visible('#modechooser')&&q('.app').inert);
 q('#choose_wasp').click();await until(()=>!visible('#modechooser'));
 await check('same-mode refresh preserves settings and confirmation',q('[data-wp="throttle"] [data-wv="0.25"]').classList.contains('on')&&q('#mountstatus').textContent.includes('90° confirmed'));
 await req('/set?armed=1&phase=WASP_READY');await sleep(400);
 await load();await check('armed refresh permits only active mode',q('#choose_normal').disabled&&!q('#choose_wasp').disabled);
 q('#choose_wasp').click();await until(()=>!visible('#modechooser'));
 await check('Wasp uses centre aim and disables manual pad',q('#aim').style.top==='50%'&&!visible('#arc')&&q('#b_up').disabled&&q('#b_yl').disabled);
 await check('armed parameters locked',q('[data-wp="throttle"] button').disabled&&q('[data-mount="90"]').disabled);
 await check('launch ready with live target',!q('#b_eng').disabled&&q('#engtxt').textContent==='LAUNCH WASP');
 tap('#b_eng');await until(()=>q('#engtxt').textContent==='WASP FLYING');
 await check('straight launch stage visible',q('#badge').textContent.includes('STRAIGHT'));
 await req('/set?armed=1&phase=WASP_FLIGHT&elapsed=1.25');await until(()=>q('#badge').textContent.includes('BLENDING'));
 await req('/set?armed=1&phase=WASP_FLIGHT&elapsed=1.75');await until(()=>q('#badge').textContent.includes('GUIDING'));
 await check('blend then guidance stage visible',q('#badge').textContent.includes('GUIDING'));
 await load();await check('refresh during flight asks without resetting flight',(await req('/state')).mission.phase==='WASP_FLIGHT');
 q('#choose_wasp').click();await until(()=>!visible('#modechooser'));
 await check('reopening active flight does not relaunch or reset',(await req('/state')).mission.phase==='WASP_FLIGHT');
 tap('#b_wasp_abort');await until(()=>q('#engtxt').textContent==='DONE · DISARM');
 await check('abort sent once and launch remains locked',q('#b_eng').disabled&&(await req('/commands')).filter(x=>x[0]==='abort').length===1);
 await req('/set?armed=0&phase=PILOT');await sleep(350);q('#t_menu').click();q('#t_mode').click();q('#choose_normal').click();
 await until(()=>!visible('#modechooser'));
 await check('returning to normal requires physical 30 degree mount',q('#mountstatus').textContent.includes('30° required'));
 for (const size of [[390,844],[820,1180],[1180,820],[1366,657]]){
   f.width=size[0];f.height=size[1];await sleep(100);
   const b=q('#choose_wasp').getBoundingClientRect();
   await check('no horizontal overflow '+size,q('html').scrollWidth<=size[0]);
 }
 await req('/done');
}
main().catch(async e=>{await req('/report?data='+encodeURIComponent(JSON.stringify({name:String(e),ok:false})));await req('/done')});
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, data, mime="application/json"):
        body = json.dumps(data).encode() if mime == "application/json" else data
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        with lock:
            if u.path == "/h":
                return self.send(HARNESS.encode(), "text/html")
            if u.path == "/":
                return self.send((ROOT / "mission_ui.html").read_bytes(), "text/html")
            if u.path == "/state":
                return self.send(state())
            if u.path == "/cmd":
                why = ""
                for k, v in q.items():
                    if k == "c":
                        continue
                    commands.append((k, v[0]))
                    if k in ("sky", "follow"):
                        detector_settings["sky_gate" if k == "sky" else "follower"] = v[0] == "on"
                    else:
                        why = phone_command(m, k, v[0], time.perf_counter())
                    if k == "engage" and not why:
                        m.phase = "WASP_FLIGHT"
                        m.wasp.begin(time.perf_counter())
                    elif k == "abort":
                        m.phase = "WASP_DONE"
                        m.wasp.started = None
                return self.send(dict(state(), refused=why))
            if u.path == "/set":
                mapper.armed = q.get("armed", ["0"])[0] == "1"
                m.phase = q.get("phase", [m.phase])[0]
                if "elapsed" in q:
                    m.wasp.started = time.perf_counter()-float(q["elapsed"][0])
                return self.send({})
            if u.path == "/commands":
                return self.send(commands)
            if u.path == "/report":
                r = json.loads(q["data"][0])
                reports.append(r)
                print(r, flush=True)
                return self.send({})
            if u.path == "/done":
                done.set()
                return self.send({})
            if u.path == "/console":
                return self.send([])
            if u.path.startswith("/font/") or u.path.startswith("/img/"):
                path = ROOT / ("fonts" if u.path.startswith("/font/") else "img") / Path(u.path).name
                if path.is_file():
                    return self.send(path.read_bytes(), "font/ttf" if u.path.startswith("/font/") else "image/svg+xml")
            self.send_error(404)


if __name__ == "__main__":
    if not available():
        sys.exit("A headless browser is required for this test")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    # Snap Firefox has a private /tmp; use a profile it can actually see.
    with tempfile.TemporaryDirectory(prefix=".wasp-browser-", dir=ROOT) as profile:
        with open("/tmp/wasp-firefox.log", "w") as log:
            proc = launch(profile, f"http://127.0.0.1:{srv.server_port}/h", output=log)
            try:
                finished = done.wait(50)
            finally:
                proc.terminate()
                proc.wait(timeout=10)
    srv.shutdown()
    assert finished, "Browser timed out; see /tmp/wasp-firefox.log"
    assert reports and all(r["ok"] for r in reports), reports
    print(f"All {len(reports)} browser checks passed")
