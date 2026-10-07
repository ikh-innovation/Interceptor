"""Browser checks of persistent mode, compact parameters and Wasp controls."""
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
detector_settings = dict(sky_gate=True, follower=True, follower_default=True, cam_fps=30, cam_default=30, cam_busy=False)
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
async function until(fn){for(let i=0;i<100;i++){if(fn())return;await sleep(50)}throw Error('timeout: '+fn)}
async function load(){f.src='/?t='+Date.now();await new Promise(r=>f.onload=r);await until(()=>q('[data-wp="throttle"] button.on')&&q('#c_ai').classList.contains('ok'))}
function parameters(){q('#t_menu').click();q('#t_par').click()}
function tap(s){q(s).dispatchEvent(new f.contentWindow.PointerEvent('pointerdown',{bubbles:true,cancelable:true,pointerId:1}))}
async function switchMode(mode){q('[data-flightmode="'+mode+'"]').click();await until(()=>visible('#modechange'));await check('camera warning '+mode,q('#mode_degrees').textContent===(mode==='wasp'?'90°':'30°'));q('#mode_confirm').click();await until(()=>!visible('#modechange'))}
async function layout(mode){
 for(const size of [[320,640],[390,844],[820,1180],[1180,820],[1366,657]]){
  f.width=size[0];f.height=size[1];await sleep(150);
  const d=f.contentDocument, blocks=[...d.querySelectorAll('.param-block')].map(x=>x.getBoundingClientRect());
  const segs=[...d.querySelectorAll('#detection_params .seg')].map(x=>x.getBoundingClientRect());
  await check(mode+' section edges '+size,blocks.every(x=>Math.abs(x.left-blocks[0].left)<1&&Math.abs(x.width-blocks[0].width)<1));
  await check(mode+' detection alignment '+size,segs.every(x=>Math.abs(x.left-segs[0].left)<1&&Math.abs(x.width-segs[0].width)<1));
  await check(mode+' no horizontal overflow '+size,d.documentElement.scrollWidth<=size[0]&&q('#sh_speed .body').scrollWidth<=q('#sh_speed .body').clientWidth);
  const buttons=[...d.querySelectorAll('#sh_speed .seg button')].filter(x=>x.getClientRects().length);
  await check(mode+' choice text fits '+size,buttons.every(x=>x.scrollWidth<=x.clientWidth+1));
  await check(mode+' equal choices '+size,[...d.querySelectorAll('#sh_speed .seg')].filter(x=>x.getClientRects().length).every(x=>{const a=[...x.children].map(b=>b.getBoundingClientRect().width);return Math.max(...a)-Math.min(...a)<1}));
 }
 f.width=390;f.height=844;await sleep(150);
}
async function main(){
 await load();
 await check('startup has no chooser or blocking popup',!q('#modechooser')&&!visible('#modechange')&&!q('.app').inert);
 await check('startup is read-only',(await req('/commands')).length===0);
 await check('TensorRT FPS stays green',q('#c_ai_k').textContent==='TensorRT'&&q('#c_ai').classList.contains('ok'));
 parameters();
 await check('three parameter sections in order',[...f.contentDocument.querySelectorAll('.param-title')].map(x=>x.textContent).join('|')==='1Flight mode|2Detection|3Flight parametersSEEKER');
 await check('Seeker ON and Wasp OFF',q('[data-flightmode="normal"]').textContent.startsWith('Seeker drone')&&q('[data-flightmode="normal"]').getAttribute('aria-pressed')==='true'&&q('[data-flightmode="wasp"]').getAttribute('aria-pressed')==='false');
 await check('detection always visible',visible('#skyseg')&&visible('#folseg')&&visible('#camseg'));
 const start=(await req('/commands')).length;
 q('[data-flightmode="wasp"]').click();await until(()=>visible('#modechange'));
 await check('switch waits for physical-camera confirmation',(await req('/commands')).length===start&&q('#mode_degrees').textContent==='90°');
 q('#mode_cancel').click();
 await check('Cancel preserves mode and sends nothing',!visible('#modechange')&&!(await req('/state')).mission.wasp_enabled&&(await req('/commands')).length===start);
 await switchMode('normal');
 await check('normal confirmation sets 30 degrees',(await req('/state')).mission.camera_mount_deg===30&&q('#mountstatus').textContent==='30° ✓');
 await layout('normal');
 q('#ptab_auto').click();
 await check('Auto tab replaces Manual controls',visible('#pp_auto')&&!visible('#pp_semi'));
 q('[data-autoen="on"]').click();await until(()=>q('#autotag').textContent==='ON');
 q('[data-ap="height"][data-av="20"]').click();await until(()=>q('[data-ap="height"][data-av="20"]').classList.contains('on'));
 await check('choice applies without a default popup',!q('#defask')&&(await req('/state')).mission.auto_params.height===20);
 q('[data-save="auto:height"]').click();await sleep(150);
 await check('explicit save sends the correct default command',(await req('/commands')).some(x=>x[0]==='def'&&x[1]==='auto:height'));
 q('#ptab_semi').click();q('.seg[data-g="vert"] [data-lv="FAST"]').click();await until(()=>q('.seg[data-g="vert"] [data-lv="FAST"]').classList.contains('on'));
 await check('Manual speed changes apply',(await req('/state')).mission.levels.vert==='FAST');
 await switchMode('wasp');
 await check('Wasp switch disables Auto and confirms 90 degrees',(await req('/state')).mission.wasp_camera_confirmed&&!(await req('/state')).mission.auto_enabled&&q('#mountstatus').textContent==='90° ✓');
 await check('mode and mount sent in the correct order',JSON.stringify((await req('/commands')).slice(-2))===JSON.stringify([['flightmode','wasp'],['mount','90']]));
 await check('Wasp flight parameters replace Normal',visible('#pp_wasp')&&!visible('#normal_params')&&q('#flight_mode_tag').textContent==='WASP');
 await check('Wasp has all eight parameter groups',q('#pp_wasp').querySelectorAll('[data-wp]').length===8);
 await check('Wasp defaults are 10 seconds and gain 2',q('[data-wp="duration"] [data-wv="10"]').classList.contains('on')&&q('[data-wp="gain"] [data-wv="2"]').classList.contains('on'));
 q('[data-wp="duration"] [data-wv="5"]').click();await until(()=>q('[data-wp="duration"] [data-wv="5"]').classList.contains('on'));
 q('[data-wp="duration"] [data-wv="10"]').click();await until(()=>q('[data-wp="duration"] [data-wv="10"]').classList.contains('on'));
 q('[data-wp="rate"] [data-wv="45"]').click();await until(()=>q('[data-wp="rate"] [data-wv="45"]').classList.contains('on'));
 await check('10-second and 45-degree choices apply to backend',(await req('/state')).mission.wasp_params.duration===10&&(await req('/state')).mission.wasp_params.rate===45);
 await layout('wasp');
 q('[data-sky="off"]').click();await until(()=>q('[data-sky="off"]').classList.contains('on'));
 q('[data-fol="off"]').click();await until(()=>q('[data-fol="off"]').classList.contains('on'));
 q('[data-cam="60"]').click();await until(()=>q('[data-cam="60"]').classList.contains('on'));
 await check('detection controls work in Wasp',!(await req('/state')).sky_gate&&!(await req('/state')).follower&&(await req('/state')).cam_fps===60);
 q('[data-wp="throttle"] [data-wv="0.25"]').click();await until(()=>q('[data-wp="throttle"] [data-wv="0.25"]').classList.contains('on'));
 q('[data-wp="straight"] [data-wv="0.5"]').click();await until(()=>q('[data-wp="straight"] [data-wv="0.5"]').classList.contains('on'));
 q('[data-wp="blend"] [data-wv="0.25"]').click();
 await until(()=>q('[data-wp="blend"] [data-wv="0.25"]').classList.contains('on'));
 const count=(await req('/commands')).length;
 await load();
 await check('refresh has no popup and sends no mode command',!visible('#modechange')&&!q('.app').inert&&(await req('/commands')).length===count);
 parameters();
 await check('refresh preserves Wasp mode',q('[data-flightmode="wasp"]').classList.contains('on'));
 await check('refresh preserves throttle',q('[data-wp="throttle"] [data-wv="0.25"]').classList.contains('on'));
 await check('refresh preserves duration, rate and gain',q('[data-wp="duration"] [data-wv="10"]').classList.contains('on')&&q('[data-wp="rate"] [data-wv="45"]').classList.contains('on')&&q('[data-wp="gain"] [data-wv="2"]').classList.contains('on'));
 await check('refresh preserves camera confirmation',q('#mountstatus').textContent==='90° ✓');
 await req('/set?armed=1&phase=WASP_READY');await until(()=>q('[data-flightmode="normal"]').disabled);
 await check('mode and Wasp parameters locked while armed',[...f.contentDocument.querySelectorAll('[data-flightmode], [data-wp] button')].every(x=>x.disabled)&&q('#mountstatus').disabled);
 q('[data-flightmode="normal"]').click();await sleep(100);
 await check('armed mode click sends nothing',(await req('/commands')).length===count&&!visible('#modechange'));
 q('#sh_speed [data-close]').click();
 await check('Wasp aim and launch controls remain correct',q('#aim').style.top==='50%'&&!visible('#arc')&&q('#b_up').disabled&&!q('#b_eng').disabled);
 tap('#b_eng');await until(()=>q('#engtxt').textContent==='WASP FLYING');
 await req('/set?armed=1&phase=WASP_FLIGHT&elapsed=1.25');await until(()=>q('#badge').textContent.includes('GUIDING'));
 await check('launch sent once',(await req('/commands')).filter(x=>x[0]==='engage').length===1);
 await load();
 await check('armed refresh preserves the active flight',!visible('#modechange')&&!q('.app').inert&&(await req('/state')).mission.phase==='WASP_FLIGHT'&&(await req('/commands')).filter(x=>x[0]==='engage').length===1);
 parameters();await check('camera switching blocked during Wasp flight',q('[data-cam="30"]').disabled&&q('[data-cam="60"]').disabled);
 q('#sh_speed [data-close]').click();tap('#b_wasp_abort');await until(()=>q('#engtxt').textContent==='DONE · DISARM');
 await check('abort remains available and sends once',(await req('/commands')).filter(x=>x[0]==='abort').length===1&&q('#b_eng').disabled);
 await req('/set?armed=0&phase=PILOT');await sleep(350);parameters();
 await switchMode('normal');
 await check('return to Normal confirms 30 degrees and changes flight parameters',q('#mountstatus').textContent==='30° ✓'&&visible('#normal_params')&&!visible('#pp_wasp'));
 q('[data-sky="on"]').click();q('[data-fol="on"]').click();q('[data-cam="30"]').click();await until(()=>q('[data-cam="30"]').classList.contains('on'));
 await check('detection controls work in Normal',(await req('/state')).sky_gate&&(await req('/state')).follower&&(await req('/state')).cam_fps===30);
 q('[data-flightmode="wasp"]').click();await until(()=>visible('#modechange'));
 await req('/set?armed=1&phase=PILOT');await until(()=>q('#mode_confirm').disabled);
 await check('arming during confirmation blocks the change',q('#mode_note').textContent.includes('Disarm')&&!q('#mode_cancel').disabled);
 q('#mode_cancel').click();await req('/set?armed=0&phase=PILOT&mode=wasp');await until(()=>q('[data-flightmode="wasp"]').classList.contains('on'));
 await check('another client changing mode updates UI without a chooser',!visible('#modechange')&&!q('.app').inert&&visible('#pp_wasp'));
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
                page=(ROOT / "mission_ui.html").read_text()
                probe="<script>window.addEventListener('error',e=>fetch('/report?data='+encodeURIComponent(JSON.stringify({name:'JavaScript: '+e.message,ok:false}))));</script>"
                return self.send(page.replace("<head>","<head>"+probe).encode(), "text/html")
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
                    elif k == "cam":
                        detector_settings["cam_fps"] = int(v[0])
                    elif k == "def":
                        kind, name = v[0].split(":")
                        if kind == "auto":
                            m.auto_defaults[name] = m.auto_params()[name]
                        elif kind == "lvl":
                            m.level_defaults[name] = m.level[name]
                        elif name == "follow":
                            detector_settings["follower_default"] = detector_settings["follower"]
                        elif name == "cam":
                            detector_settings["cam_default"] = detector_settings["cam_fps"]
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
                if "mode" in q:
                    m.set_wasp_enabled(q["mode"][0] == "wasp", selection=True)
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
                finished = done.wait(80)
            finally:
                proc.terminate()
                proc.wait(timeout=10)
    srv.shutdown()
    assert finished, "Browser timed out; see /tmp/wasp-firefox.log"
    assert reports and all(r["ok"] for r in reports), reports
    print(f"All {len(reports)} browser checks passed")
