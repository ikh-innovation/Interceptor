
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

    def idle_pwm(self, yaw, climb):
        return {}

    def idle_request_alt_hold(self):
        return True

    def request_mode(self, name):
        return True

    def request_disarm(self, force=False):
        return True

    # The app's ARM: the FC arms when every check passes.
    armed_t = None

    def wasp_arm_ready(self):
        if self.armed:
            return "already armed"
        return simulation["arm_why"]

    def request_arm(self):
        self.armed, self.armed_t = True, time.perf_counter()
        return True


mapper = Mapper()
settings_tmp = tempfile.TemporaryDirectory(prefix="wasp-ui-settings-")
m = Mission(mapper, log=lambda *_: None, wasp_path=str(Path(settings_tmp.name)/"wasp.json"))
commands, reports = [], []
detector_settings = dict(sky_gate=True, follower=True, follower_default=True, cam_fps=30, cam_default=30, cam_busy=False,
                         video_stabilization=False, engine=640, engines=[640, 960], engine_default=640,
                         engine_busy=False)
simulation = dict(target=True, attitude=True, gate_why="", arm_why="", gps=True, compass=True,
                  position=[37.98, 23.72], home=[37.98, 23.72])
done = threading.Event()
lock = threading.Lock()


def context():
    return dict(gate_ok=mapper.armed and not simulation["gate_why"],
                att=dict(roll=0, pitch=0, yaw=0) if simulation["attitude"] else None,
                height=0, climb=0, idle_ok=True, airborne=True, safedown_ok=mapper.armed,
                fc_mode="ALT_HOLD" if m.idle.active or m.safedown.active else "ACRO",
                ai_cmd=None, det_t=None, det_conf=None, position=simulation["position"],
                armed_s=None if not mapper.armed or mapper.armed_t is None else time.perf_counter()-mapper.armed_t)


def state():
    if m.wasp.launch_t is not None:
        m.cycle(time.perf_counter(), context())
    m._fence_state = m.fence.check(simulation["position"]) if m.fence.points else (None, None)
    m._armed_s = context()["armed_s"]  # the real app's loop refreshes it 50 times a second
    s = m.snapshot(time.perf_counter())
    gate = simulation["gate_why"] or ("" if simulation["attitude"] else "no fresh attitude")
    s.update(gate_why=gate if mapper.armed else "disarmed", height_m=0,
             climb_mps=0, heading_deg=25,
             det_conf=.9 if simulation["target"] else None,
             det_age_s=.01 if simulation["target"] else None,
             target_fresh=simulation["target"], target_az_deg=0, target_el_deg=0,
             wasp_attitude_fresh=simulation["attitude"],
             wasp_arm_why=mapper.wasp_arm_ready(),
             armed_s=context()["armed_s"],
             gps=dict(fix=3 if simulation["gps"] else 0, sats=12 if simulation["gps"] else 0, ok=simulation["gps"]),
             compass_ok=simulation["compass"], position_ok=simulation["position"] is not None,
             position=simulation["position"], home=simulation["home"])
    return dict(mission=s, armed=mapper.armed, linked=True,
                fc_mode="ACRO" if m.wasp.enabled else "ALT_HOLD", control="you",
                detector="TensorRT", fps=30,
                cam_aspect=16/9 if detector_settings["cam_fps"] == 60 else 4/3,
                cam_frame={21: [3280, 2464], 30: [1280, 960], 60: [1280, 720]}[detector_settings["cam_fps"]],
                cam_capture="VIC", cam_error="", **detector_settings)


HARNESS = r'''<!doctype html><html><head><meta charset="utf-8"></head><body><iframe id="f" width="390" height="844"></iframe><script>
const f=document.getElementById('f');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const req=async(path)=>{const r=await fetch(path);return r.json()};
const q=s=>f.contentDocument.querySelector(s);
const visible=s=>q(s)&&q(s).getClientRects().length>0&&f.contentWindow.getComputedStyle(q(s)).display!=='none';
async function check(name,ok){await req('/report?data='+encodeURIComponent(JSON.stringify({name,ok:!!ok})));if(!ok)throw Error(name)}
async function until(fn){for(let i=0;i<100;i++){if(await fn())return;await sleep(50)}throw Error('timeout: '+fn)}
async function load(){f.src='/?t='+Date.now();await new Promise(r=>f.onload=r);await until(()=>q('[data-wp="throttle"] button.on')&&q('#c_ai').classList.contains('ok'))}
function parameters(){q('#t_menu').click();q('#t_par').click()}
function tap(s){q(s).dispatchEvent(new f.contentWindow.PointerEvent('pointerdown',{bubbles:true,cancelable:true,pointerId:1}))}
async function switchMode(mode){q('[data-flightmode="'+mode+'"]').click();await until(()=>visible('#modechange'));await check('camera warning '+mode,q('#mode_degrees').textContent===(mode==='wasp'?'90°':'30°'));q('#mode_confirm').click();await until(()=>!visible('#modechange'))}
async function layout(mode){
 for(const size of [[320,640],[390,844],[820,1180],[1180,820],[1366,657]]){
  f.width=size[0];f.height=size[1];await sleep(150);
  const d=f.contentDocument, blocks=[...d.querySelectorAll('.param-block')].filter(x=>x.getClientRects().length).map(x=>x.getBoundingClientRect());
  // Only rows that are shown take part: WASP hides the Idle row (data-only="normal"),
  // and a hidden row's box is 0 wide at x=0.
  const segs=[...d.querySelectorAll('#detection_params .seg, #detection_params .idle-input')].filter(x=>x.getClientRects().length).map(x=>x.getBoundingClientRect());
  await check(mode+' Idle row shown only in Seeker '+size,mode==='wasp'?!visible('#idle_seconds')&&segs.length===q('#detection_params').querySelectorAll('.seg').length:visible('#idle_seconds'));
  await check(mode+' section edges '+size,blocks.every(x=>Math.abs(x.left-blocks[0].left)<1&&Math.abs(x.width-blocks[0].width)<1));
  await check(mode+' detection alignment '+size,segs.every(x=>Math.abs(x.left-segs[0].left)<1&&Math.abs(x.width-segs[0].width)<1));
  await check(mode+' no horizontal overflow '+size,d.documentElement.scrollWidth<=size[0]&&q('#sh_speed .body').scrollWidth<=q('#sh_speed .body').clientWidth);
  const buttons=[...d.querySelectorAll('#sh_speed .seg button')].filter(x=>x.getClientRects().length);
  await check(mode+' choice text fits '+size,buttons.every(x=>x.scrollWidth<=x.clientWidth+1));
  if(mode==='wasp') await check('Wasp matrix fits '+size,q('#wasp_profile_matrix').scrollWidth<=q('#wasp_profile_matrix').clientWidth+1&&[...q('#wasp_profile_matrix').querySelectorAll('th,td')].every(x=>x.scrollWidth<=x.clientWidth+1));
  await check(mode+' equal choices '+size,[...d.querySelectorAll('#sh_speed .seg')].filter(x=>x.getClientRects().length).every(x=>{const a=[...x.children].map(b=>b.getBoundingClientRect().width);return Math.max(...a)-Math.min(...a)<1}));
  await check('Video Stabilization sits directly below Camera '+size,q('#stabilizationseg').closest('.detection-row').previousElementSibling.contains(q('#camseg'))&&q('#stabilizationseg').getBoundingClientRect().top>=q('#camseg').getBoundingClientRect().bottom);
  if(mode!=='wasp') await check('Idle stays below Video Stabilization '+size,q('#idle_seconds').getBoundingClientRect().top>=q('#stabilizationseg').getBoundingClientRect().bottom);
 }
 f.width=390;f.height=844;await sleep(150);
}
function openPop(id){q('[data-pop="'+id+'"]').click()}
function closePop(){q('.pop.open [data-popclose]').click()}
async function popLayout(){
 for(const size of [[320,640],[390,844],[844,390],[1180,820]]){
  f.width=size[0];f.height=size[1];await sleep(150);
  for(const id of ['pop_det','pop_flight','pop_test','pop_fence']){
   openPop(id);await until(()=>visible('#'+id+' .card'));
   const card=q('#'+id+' .card').getBoundingClientRect(), body=q('#'+id+' .pbody');
   await check(id+' fits the screen '+size,card.left>=0&&card.right<=size[0]+1&&card.top>=0&&card.bottom<=size[1]+1&&body.scrollWidth<=body.clientWidth+1);
   const buttons=[...q('#'+id).querySelectorAll('.seg button,.fbtn')].filter(x=>x.getClientRects().length);
   await check(id+' choice text fits '+size,buttons.every(x=>x.scrollWidth<=x.clientWidth+1));
   if(id==='pop_flight') await check('Wasp matrix fits '+size,q('#wasp_profile_matrix').scrollWidth<=q('#wasp_profile_matrix').clientWidth+1);
   closePop();await until(()=>!visible('#'+id+' .card'));
  }
 }
 f.width=390;f.height=844;await sleep(150);
}
async function flightLayout(){
 for(const size of [[320,640],[390,844],[568,320],[667,375],[844,390],[820,1180],[1180,820],[1366,657]]){
  f.width=size[0];f.height=size[1];await sleep(150);
  const tiles=[...f.contentDocument.querySelectorAll('.tile')].filter(x=>x.getClientRects().length);
  await check('Wasp shows three tiles: FC mode, TensorRT, GPS / Compass '+size,tiles.length===3&&tiles[0].contains(q('#c_mode'))&&tiles[1].contains(q('#c_ai'))&&tiles[2].contains(q('#c_gps')));
  const tb=tiles.map(x=>x.getBoundingClientRect());
  await check('Wasp tiles in one row '+size,tb.every((x,i)=>Math.abs(x.top-tb[0].top)<1&&(!i||x.left>=tb[i-1].right-1)));
  await check('Wasp tile text fits '+size+' '+tiles.map(x=>[...x.children].map(c=>c.textContent+':'+c.scrollWidth+'/'+c.clientWidth).join(',')).join(' | '),tiles.every(x=>[...x.children].every(c=>c.scrollWidth<=c.clientWidth+1)));
  const tools=[...f.contentDocument.querySelectorAll('.tool')], sd=q('#b_safedown').getBoundingClientRect(), arm=q('#b_arm').getBoundingClientRect();
  await check('Wasp has four settings buttons with icons '+size,tools.length===4&&tools.every(x=>x.getClientRects().length&&x.querySelector('svg')));
  await check('Wasp hides Seeker controls and the old panel '+size,!visible('.ctl')&&!visible('#normal_pad')&&!visible('#b_lock')&&!visible('#b_eng')&&!visible('#dial')&&!visible('#hbox')&&!visible('#tape')&&!q('#wasp_launch_main')&&!q('#wasp_pad'));
  await check('SAFEDOWN left, ARM / LAUNCH right, same row and size '+size,sd.right<=arm.left&&Math.abs(sd.top-arm.top)<1&&Math.abs(sd.height-arm.height)<1&&Math.abs(sd.width-arm.width)<1&&sd.height>=50);
  await check('SAFEDOWN is always red '+size,f.contentWindow.getComputedStyle(q('#b_safedown')).backgroundColor==='rgb(198, 40, 40)');
  await check('Wasp big button labels fit '+size,[q('#b_safedown'),q('#b_arm')].every(x=>[...x.querySelectorAll('b')].every(c=>c.scrollWidth<=c.clientWidth+1)));
  const camera=q('#vbox').getBoundingClientRect(), ui=q('#wasp_ui').getBoundingClientRect(), tbox=q('.wtools').getBoundingClientRect();
  await check('Wasp controls stay inside viewport '+size,[q('#wasp_ui'),q('#vbox')].every(x=>{const b=x.getBoundingClientRect();return b.left>=0&&b.right<=size[0]+1&&b.top>=0&&b.bottom<=size[1]+1}));
  const side=size[0]>=700||(size[0]>=560&&size[0]>size[1]);
  await check('Wasp layout matches orientation '+size,side?camera.right<=ui.left:ui.top>=camera.bottom-1);
  await check('settings buttons sit between the image and the big buttons '+size,tbox.bottom<=Math.min(sd.top,arm.top)+1&&(side||tbox.top>=camera.bottom-1));
  if(!side&&size[1]>=700) await check('settings buttons fill the space below the image '+size,tbox.top-camera.bottom<40||tbox.height>=300);
  await check('Wasp camera uses the full frame '+size,Math.abs(q('#v').getBoundingClientRect().width-camera.width)<3&&Math.abs(q('#v').getBoundingClientRect().height-camera.height)<3);
 }
 f.width=390;f.height=844;await sleep(150);
}
async function main(){
 await load();
 await check('startup has no chooser or blocking popup',!q('#modechooser')&&!visible('#modechange')&&!q('.app').inert);
 await check('startup is read-only',(await req('/commands')).length===0);
 await check('Seeker keeps Engage and Lock without Auto or Abort',visible('#normal_pad')&&visible('#b_lock')&&visible('#b_eng')&&!q('#b_auto')&&!visible('#b_safedown')&&visible('#dial')&&visible('#hbox')&&visible('#tape')&&!visible('#wasp_ui'));
 await check('TensorRT FPS stays green',q('#c_ai_k').textContent==='TensorRT 640'&&q('#c_ai').classList.contains('ok'));
 parameters();
 await check('three Seeker parameter sections in order',[...f.contentDocument.querySelectorAll('.param-title')].filter(x=>x.getClientRects().length).map(x=>x.textContent).join('|')==='1Flight mode|2Detection|3Flight parametersSEEKER');
 await check('Seeker ON and Wasp OFF',q('[data-flightmode="normal"]').textContent.startsWith('Seeker drone')&&q('[data-flightmode="normal"]').getAttribute('aria-pressed')==='true'&&q('[data-flightmode="wasp"]').getAttribute('aria-pressed')==='false');
 await check('detection always visible',visible('#skyseg')&&visible('#folseg')&&visible('#camseg'));
 await check('Idle default is 10 seconds and Search removed',q('#idle_seconds').value==='10'&&!q('[data-g="search"]'));
 const field=q('#idle_seconds');field.focus();field.value='12.5';await sleep(400);
 await check('polling preserves typed Idle value',field.value==='12.5');
 field.dispatchEvent(new f.contentWindow.Event('change',{bubbles:true}));field.blur();
 await until(async()=>q('#idle_seconds').value==='12.5'&&(await req('/state')).mission.idle_s===12.5);
 await check('custom Idle applies',(await req('/state')).mission.idle_s===12.5);
 const idleCommands=(await req('/commands')).length;
 field.value='0';field.dispatchEvent(new f.contentWindow.Event('change',{bubbles:true}));
 await check('invalid Idle sends nothing and restores saved value',field.value==='12.5'&&(await req('/commands')).length===idleCommands);
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
 await check('Seeker Auto labels the primary action Auto Engage',q('#engtxt').textContent==='AUTO ENGAGE'&&q('#b_eng').getAttribute('aria-label')==='AUTO ENGAGE'&&!visible('#b_safedown')&&!q('#b_auto'));
 q('[data-ap="height"][data-av="20"]').click();await until(()=>q('[data-ap="height"][data-av="20"]').classList.contains('on'));
 await check('choice applies without a default popup',!q('#defask')&&(await req('/state')).mission.auto_params.height===20);
 q('[data-save="auto:height"]').click();await sleep(150);
 await check('explicit save sends the correct default command',(await req('/commands')).some(x=>x[0]==='def'&&x[1]==='auto:height'));
 q('#ptab_semi').click();q('.seg[data-g="vert"] [data-lv="FAST"]').click();await until(()=>q('.seg[data-g="vert"] [data-lv="FAST"]').classList.contains('on'));
 await check('Manual speed changes apply',(await req('/state')).mission.levels.vert==='FAST');
 await check('no camera prompt in Seeker',!visible('#mount_flash'));
 await req('/cmd?c=test&flightmode=wasp');
 await until(()=>visible('#mount_flash'));
 await check('unconfirmed Wasp camera flashes a prompt on the video',q('#mount_flash').textContent==='90°Turn the camera to 90°, then tap here to confirm'
   &&!q('#mount_flash').disabled&&getComputedStyle(q('#mount_flash')).animationName==='mountflash'&&q('#mountstatus').textContent==='90° required');
 q('#mount_flash').click();await until(()=>visible('#modechange'));
 await check('camera prompt opens the 90 degree confirmation',q('#mode_degrees').textContent==='90°');
 q('#mode_confirm').click();await until(()=>!visible('#modechange'));
 await until(()=>!visible('#mount_flash'));
 await check('Wasp switch disables Auto and confirms 90 degrees',(await req('/state')).mission.wasp_camera_confirmed&&!(await req('/state')).mission.auto_enabled&&q('#mountstatus').textContent==='90° ✓');
 await check('mode and mount sent in the correct order',JSON.stringify((await req('/commands')).slice(-2))===JSON.stringify([['flightmode','wasp'],['mount','90']]));
 parameters();
 await check('PARAMETERS in WASP keeps only the flight mode',[...q('#sh_speed').querySelectorAll('.param-block')].map(x=>x.id).join()==='flight_mode_params'&&!q('#sh_speed').contains(q('#pp_wasp')));
 q('#sh_speed [data-close]').click();
 openPop('pop_flight');await until(()=>visible('#pp_wasp'));
 await check('Flight popup holds the WASP flight parameters',visible('#pp_wasp')&&!visible('#normal_params')&&q('#flight_mode_tag').textContent==='WASP');
 await check('simple steering is visible and advanced controls start collapsed',visible('#wasp_profiles')&&!q('#wasp_advanced').open&&!visible('[data-wp="blend"]')&&!visible('[data-wp="gain"]')&&visible('[data-wp="throttle"]'));
 await check('no launch timer anywhere',!q('#wasp_launch_main')&&!q('.main-countdown')&&(await req('/state')).mission.wasp_params.launch_s===0);
 await check('Standard is selected from actual backend settings',q('[data-wasp-profile="standard"]').getAttribute('aria-pressed')==='true'&&q('#wasp_profile_matrix th[data-profile-col="standard"]').getAttribute('aria-current')==='true');
 await check('matrix shows all profile rates and lead',q('[data-profile-col="smooth"][data-profile-setting="rate"]').textContent==='45'&&q('[data-profile-col="extra"][data-profile-setting="rate"]').textContent==='120'&&q('[data-profile-col="extra"][data-profile-setting="lead"]').textContent==='80');
 const profileStart=(await req('/commands')).length;
 q('[data-wasp-profile="smooth"]').click();await until(()=>q('[data-wasp-profile="smooth"]').classList.contains('on'));
 const smooth=(await req('/state')).mission.wasp_params;
 await check('one profile command applies all three steering values',smooth.blend===1&&smooth.rate===45&&smooth.gain===2&&(await req('/commands')).length===profileStart+1);
 q('[data-wasp-profile="extra"]').click();await until(()=>q('[data-wasp-profile="extra"]').classList.contains('on'));
 const extra=(await req('/state')).mission.wasp_params;
 await check('Extra applies exact steering settings',extra.blend===0.25&&extra.rate===120&&extra.gain===6&&extra.lead===0.08&&extra.camera_roll===0);
 q('[data-wasp-profile="standard"]').click();await until(()=>q('[data-wasp-profile="standard"]').classList.contains('on'));
 q('#wasp_advanced').open=true;
 await until(()=>(q('#align_img').getAttribute('src')||'').startsWith('/stream.mjpg'));
 const nose=()=>[parseFloat(q('#align_nose').style.left),parseFloat(q('#align_nose').style.top)];
 await check('camera alignment shows a live view in the popup',visible('#align_view')&&nose()[1]>80&&q('#align_help').textContent.startsWith('Nose is at the bottom of the picture'));
 q('[data-wp="camera_roll"] [data-wv="90"]').click();await until(()=>q('#align_help').textContent.startsWith('Nose is at the right'));
 await check('changing camera alignment moves the nose marker',nose()[0]>80&&(await req('/state')).mission.wasp_params.camera_roll===90);
 q('[data-wp="camera_roll"] [data-wv="0"]').click();await until(()=>q('#align_help').textContent.startsWith('Nose is at the bottom'));
 closePop();await until(()=>(q('#align_img').getAttribute('src')||'').startsWith('data:'));
 await check('closing the popup stops the alignment stream',!(q('#align_img').getAttribute('src')||'').startsWith('/stream'));
 openPop('pop_flight');
 await check('Wasp has eight parameter groups including the throttle ramp',q('#pp_wasp').querySelectorAll('[data-wp]').length===8&&q('#pp_wasp [data-wp="ramp"] [data-wv="0.1"]').classList.contains('on'));
 q('[data-wp="rate"] [data-wv="45"]').click();await until(()=>q('[data-wp="rate"] [data-wv="45"]').classList.contains('on'));
 await check('45-degree choice applies to backend',(await req('/state')).mission.wasp_params.rate===45);
 await check('Wasp throttle choices are 75, 85 and 100 percent',[...q('[data-wp="throttle"]').children].map(b=>b.textContent).join(',')==='75%,85%,100%');
 q('[data-wp="throttle"] [data-wv="0.75"]').click();await until(()=>q('[data-wp="throttle"] [data-wv="0.75"]').classList.contains('on'));
 closePop();
 openPop('pop_det');await until(()=>visible('#skyseg'));
 await check('Detection popup holds the detection settings and no Idle',visible('#skyseg')&&visible('#folseg')&&visible('#camseg')&&visible('#engseg')&&!visible('#idle_seconds'));
 q('[data-sky="off"]').click();await until(()=>q('[data-sky="off"]').classList.contains('on'));
 q('[data-fol="off"]').click();await until(()=>q('[data-fol="off"]').classList.contains('on'));
 q('[data-cam="60"]').click();await until(()=>q('[data-cam="60"]').classList.contains('on'));
 await check('detection controls work in Wasp',!(await req('/state')).sky_gate&&!(await req('/state')).follower&&(await req('/state')).cam_fps===60);
 q('[data-stabilization="on"]').click();await until(()=>q('[data-stabilization="on"]').getAttribute('aria-pressed')==='true');
 q('[data-stabilization="off"]').click();await until(()=>q('[data-stabilization="off"]').getAttribute('aria-pressed')==='true');
 closePop();
 await popLayout();
 const count=(await req('/commands')).length;
 await load();
 await check('refresh has no popup and sends no mode command',!visible('#modechange')&&!q('.app').inert&&!q('.pop.open')&&(await req('/commands')).length===count);
 await check('refresh preserves throttle, rate and profile',q('[data-wp="throttle"] [data-wv="0.75"]').classList.contains('on')&&q('[data-wp="rate"] [data-wv="45"]').classList.contains('on')&&q('#wasp_profiles [data-wasp-profile="standard"]').classList.contains('on'));
 // ---- GPS / compass tile
 await check('GPS / Compass tile reads YES / YES in green',q('#c_gps').textContent==='YES / YES'&&q('#c_gps').classList.contains('ok'));
 await req('/set?armed=0&phase=PILOT&compass=0');await until(()=>q('#c_gps').textContent==='YES / NO');
 await check('a missing compass shows on the tile',q('#c_gps').classList.contains('warn'));
 await req('/set?armed=0&phase=PILOT&gps=0&compass=0&armwhy=waiting for a GPS fix: SafeDown flies back to the launch point');
 await until(()=>q('#c_gps').textContent==='NO / NO'&&q('#b_arm').disabled);
 await check('no GPS: red tile and ARM refused with the reason',q('#c_gps').classList.contains('bad')&&q('#armsub').textContent==='waiting for GPS');
 await req('/set?armed=0&phase=PILOT&gps=1&compass=1&armwhy=');await until(()=>!q('#b_arm').disabled);
 await flightLayout();
 // ---- ARM, then LAUNCH WASP after 2 s
 await check('disarmed: the right button says ARM',q('#armtxt').textContent==='ARM'&&q('#b_arm').getAttribute('data-action')==='arm');
 const before=(await req('/commands')).length;
 tap('#b_arm');await until(()=>q('#armtxt').textContent.startsWith('LAUNCH IN')&&parseFloat(q('#armprog').style.width)>0);
 await check('ARM asks the app to arm, once',(await req('/commands')).slice(before).filter(x=>x[0]==='arm').length===1&&(await req('/state')).armed);
 const cd1=parseFloat(q('#armtxt').textContent.slice(10));await sleep(400);const cd2=parseFloat(q('#armtxt').textContent.slice(10));
 await check('while the FC spools up: an amber 2.5 s countdown in the button, no launch',cd1<=2.5&&cd2<cd1&&q('#b_arm').disabled&&q('#b_arm').classList.contains('arming')&&f.contentWindow.getComputedStyle(q('#b_arm')).backgroundColor==='rgb(224, 123, 0)');
 await until(()=>q('#armtxt').textContent==='LAUNCH WASP');await sleep(350);  // the colour fades in
 await check('2.5 s after arming the button turns blue: LAUNCH WASP',q('#b_arm').classList.contains('launch')&&!q('#b_arm').disabled&&f.contentWindow.getComputedStyle(q('#b_arm')).backgroundColor==='rgb(47, 102, 184)');
 await check('SAFEDOWN on the stand says it cancels',q('#sdsub').textContent==='cancel · disarm');
 await req('/set?armed=1&attitude=0');await until(()=>q('#b_arm').disabled);
 await check('launch still requires fresh attitude',q('#b_arm').disabled);
 await req('/set?armed=1&attitude=1&gate=wrong-mode');await until(()=>q('#badge').textContent==='wrong-mode');
 await check('launch still requires the controller gate',q('#b_arm').disabled&&q('#armsub').textContent==='wrong-mode');
 await req('/set?armed=1&gate=');await until(()=>!q('#b_arm').disabled);
 await req('/set?armed=1&target=0');await until(()=>q('#armtxt').textContent==='LAUNCH WASP');
 await check('launch is available without an initial detection',!q('#b_arm').disabled);
 // ---- Test mode popup
 openPop('pop_test');await until(()=>visible('#test_centre_timer'));
 await check('Centre timer defaults to 3 and is editable while armed',q('#test_centre_timer').value==='3'&&!q('#test_centre_timer').disabled);
 q('#test_centre_timer').value='1.5';q('#test_centre_timer').dispatchEvent(new f.contentWindow.Event('change',{bubbles:true}));
 await until(async()=>(await req('/state')).mission.wasp_params.centre_timer===1.5);
 q('[data-testmode="on"]').click();await until(()=>!q('.pop.open')&&visible('#test_prompt'));
 await check('Test mode returns to the image, chip ON, launch needs a target',(await req('/state')).mission.test_mode&&q('#chip_test').textContent==='ON'&&q('#b_arm').disabled);
 function point(x,y){const r=q('#v').getBoundingClientRect();q('#vbox').dispatchEvent(new f.contentWindow.PointerEvent('pointerdown',{bubbles:true,cancelable:true,pointerId:3,clientX:r.left+r.width*x,clientY:r.top+r.height*y}))}
 point(.25,.75);await until(()=>q('#test_target').style.left==='25%'&&q('#test_target').style.top==='75%');
 await until(()=>!q('#b_arm').disabled);
 tap('#b_arm');await until(()=>q('#armtxt').textContent==='WASP FLYING');
 await check('LAUNCH WASP launches at once: no countdown',(await req('/commands')).filter(x=>x[0]==='engage').length===1&&(await req('/state')).mission.phase==='WASP_FLIGHT'&&q('#b_arm').disabled);
 await check('flight badge tells the pilot how to take over',q('#badge').textContent==='TEST · STRAIGHT · SAFEDOWN OR SWITCH UP = RETURN');
 await until(()=>parseFloat(q('#test_target').style.left)>25&&parseFloat(q('#test_target').style.left)<50);
 await check('Test target moves toward the center only after launch',parseFloat(q('#test_target').style.top)<75);
 openPop('pop_det');
 await check('camera and engine switching blocked during Wasp flight',q('[data-cam="30"]').disabled&&q('[data-engine="640"]').disabled);
 closePop();
 await check('SafeDown button says what it does',q('#b_safedown b').textContent==='SAFEDOWN'&&q('#b_safedown').getAttribute('aria-label')==='SafeDown: level, fly back to the launch point and land');
 tap('#b_safedown');await until(()=>q('#badge').textContent.startsWith('SAFEDOWN'));
 const landing=(await req('/state')).mission;
 await check('SafeDown in flight starts the automatic return',landing.phase==='SAFEDOWN'&&landing.safedown.active===true&&q('#armtxt').textContent==='SAFEDOWN'&&q('#b_arm').disabled);
 await check('SafeDown badge shows the stage and Plan B',q('#badge').textContent==='SAFEDOWN · SETTLING · MIDDLE = YOU FLY');
 await req('/set?armed=1&phase=SAFEDOWN_FAILED&idle=stop');await until(()=>q('#badge').textContent.startsWith('SAFEDOWN FAILED'));
 await check('failed SafeDown tells the pilot to fly it',q('#badge').textContent==='SAFEDOWN FAILED · THROTTLE MID, SWITCH MIDDLE, FLY IT');
 await req('/set?armed=0&phase=PILOT');await sleep(350);
 openPop('pop_test');q('[data-testmode="off"]').click();await until(()=>!(q('#test_target').getClientRects().length));
 closePop();
 // ---- Geofence: draw it with a finger, save, leave it (GPS on)
 openPop('pop_fence');await until(()=>visible('#fmapc')&&q('#fm_save').textContent==='Saved');
 await check('no fence yet: chip NONE, nothing to save, no ON / OFF buttons, GPS live so no preview banner',q('#chip_fence').textContent==='NONE'&&!visible('#fence_dummy')&&q('#fm_save').disabled&&!q('[data-fence]'));
 const fm=()=>q('#fmap').dataset, near=(a,b,eps)=>Math.abs(Number(a)-b)<eps;
 await check('the map opens on the drone',near(fm().lat,37.98,1e-4)&&near(fm().lon,23.72,1e-4));
 const z0=Number(fm().z), scale0=q('#fmap_scale').textContent;
 q('#fm_zin').click();await check('+ zooms in',Number(fm().z)===z0+1&&q('#fmap_scale').textContent!==scale0);
 q('#fm_zout').click();q('#fm_zout').click();await check('- zooms out',Number(fm().z)===z0-1);
 q('#fm_zin').click();
 const c=q('#fmapc'), r=c.getBoundingClientRect(), W=f.contentWindow;
 const pe=(type,x,y,id)=>c.dispatchEvent(new W.PointerEvent(type,{bubbles:true,cancelable:true,pointerId:id||7,clientX:r.left+r.width*x,clientY:r.top+r.height*y}));
 pe('pointerdown',.5,.5);for(let k=1;k<=6;k++)pe('pointermove',.5-k*.05,.5-k*.05);pe('pointerup',.2,.2);
 await check('Move: dragging pans the map',!near(fm().lat,37.98,1e-5)&&!near(fm().lon,23.72,1e-5));
 q('#fm_here').click();
 await check('◎ brings the map back to the drone',near(fm().lat,37.98,1e-6)&&near(fm().lon,23.72,1e-6));
 q('[data-fm="draw"]').click();
 await check('Draw is selected',q('[data-fm="draw"]').classList.contains('on')&&!q('[data-fm="move"]').classList.contains('on'));
 const loop=[[.2,.2],[.5,.15],[.8,.2],[.85,.5],[.8,.8],[.5,.85],[.2,.8],[.15,.5],[.2,.22]];
 pe('pointerdown',...loop[0]);
 for(let i=1;i<loop.length;i++){const a=loop[i-1],b=loop[i];for(let k=1;k<=5;k++)pe('pointermove',a[0]+(b[0]-a[0])*k/5,a[1]+(b[1]-a[1])*k/5)}
 pe('pointerup',...loop[loop.length-1]);
 await until(()=>!q('#fm_save').disabled);
 await check('a drawn loop becomes a fence ready to save',q('#fm_save').textContent==='Save geofence');
 q('#fm_undo').click();await check('Undo removes the last corner and leaves a fence',q('#fm_save').textContent==='Save geofence'&&!q('#fm_undo').disabled);
 q('#fm_save').click();await until(async()=>(await req('/state')).mission.fence.armed);
 const fence=(await req('/state')).mission.fence;
 await check('saved fence is live with the drone inside it',fence.enabled&&fence.points.length>=4&&fence.points.length<=40&&fence.inside===true);
 await until(()=>q('#chip_fence').textContent==='ON');
 await check('fence status and chip say ON and inside',q('#fence_status').textContent.startsWith('Drone inside')&&q('#fm_save').textContent==='Saved');
 closePop();
 await req('/set?armed=0&phase=PILOT&compass=0');await until(()=>q('#chip_fence').textContent==='NO GPS');
 await check('no compass: the fence is not ON and the popup says preview only',q('#chip_fence').classList.contains('warn'));
 await req('/set?armed=0&phase=PILOT&compass=1');await until(()=>q('#chip_fence').textContent==='ON');
 await req('/set?armed=0&phase=PILOT&pos=38.5,23.72');await until(()=>q('#chip_fence').textContent==='OUT');
 await check('outside the fence: chip OUT and ARM refused',q('#chip_fence').classList.contains('bad')&&q('#b_arm').disabled&&q('#armsub').textContent==='fence: drone outside');
 openPop('pop_fence');await until(()=>q('#fence_status').textContent.startsWith('Drone OUTSIDE'));
 q('#fm_radius').value='300';q('#fm_circle').click();
 await check('Make: a circle around the drone, to be saved',!q('#fm_save').disabled&&q('#fm_radius_l').textContent==='Circle around the drone');
 q('#fm_save').click();await until(async()=>{const x=(await req('/state')).mission.fence;return x.armed&&x.points.length===24&&x.inside===true});
 await until(()=>q('#chip_fence').textContent==='ON'&&!q('#b_arm').disabled);
 await check('the drone is inside the new circle and ARM is allowed again',!q('#b_arm').disabled);
 q('#fm_points').value='38.4990, 23.7190\n38.5010, 23.7190\n38.5010, 23.7210\n38.4990, 23.7210';
 q('#fm_points').closest('details').open=true;q('#fm_usepts').click();
 await check('typed points replace the drawing',q('#fm_save').textContent==='Save geofence');
 q('#fm_save').click();await until(async()=>(await req('/state')).mission.fence.points.length===4);
 await check('typed points are saved as the fence',(await req('/state')).mission.fence.inside===true);
 // ---- No GPS: the map and every button still work
 closePop();await req('/set?armed=0&phase=PILOT&pos=none&gps=0&compass=0&armwhy=waiting for a GPS fix');
 openPop('pop_fence');await until(()=>q('#fence_status').textContent.startsWith('Geofence saved · waiting for a GPS'));
 await check('no GPS: chip NO GPS, PREVIEW ONLY banner, the circle goes at the map centre',q('#chip_fence').textContent==='NO GPS'&&q('#chip_fence').classList.contains('warn')&&visible('#fence_dummy')&&q('#fence_dummy').textContent.includes('NOT WORKING')&&q('#fm_radius_l').textContent==='Circle at the map centre (no GPS)');
 pe('pointerdown',.5,.5);for(let k=1;k<=6;k++)pe('pointermove',.5+k*.04,.5);pe('pointerup',.74,.5);
 q('[data-fm="move"]').click();pe('pointerdown',.5,.5);for(let k=1;k<=6;k++)pe('pointermove',.5+k*.04,.5);pe('pointerup',.74,.5);
 const moved=[Number(fm().lat),Number(fm().lon)];
 q('#fm_here').click();
 await check('no GPS: ◎ centres on the fence',near(fm().lat,38.5,2e-4)&&near(fm().lon,23.72,2e-4)&&!near(fm().lon,moved[1],1e-5));
 const centre=[Number(fm().lat),Number(fm().lon)];
 q('#fm_radius').value='150';q('#fm_circle').click();
 q('#fm_save').click();await until(async()=>(await req('/state')).mission.fence.points.length===24);
 const circle=(await req('/state')).mission.fence.points, mid=[circle.reduce((a,p)=>a+p[0],0)/24,circle.reduce((a,p)=>a+p[1],0)/24];
 await check('no GPS: Make draws the circle at the map centre',near(mid[0],centre[0],1e-5)&&near(mid[1],centre[1],1e-5));
 q('#fm_zin').click();await check('no GPS: + still zooms',Number(fm().z)>=1);
 q('#fm_clear').click();q('#fm_save').click();
 await until(async()=>!(await req('/state')).mission.fence.points.length);
 await until(()=>q('#chip_fence').textContent==='NONE');
 await check('Clear and save removes the fence: chip NONE',q('#chip_fence').textContent==='NONE'&&q('#fence_status').textContent.startsWith('No geofence'));
 closePop();
 await req('/set?armed=0&phase=PILOT&pos=37.98,23.72&gps=1&compass=1&armwhy=');

 parameters();
 await switchMode('normal');
 await check('leaving WASP disables Test mode and hides the WASP screen',!(await req('/state')).mission.test_mode&&!visible('#test_target')&&!visible('#wasp_ui')&&!visible('#b_safedown'));
 await check('returning to Seeker restores flight controls and its six tiles',visible('#normal_pad')&&visible('#b_lock')&&visible('#dial')&&visible('#hbox')&&visible('#tape')&&[...f.contentDocument.querySelectorAll('.tile')].filter(x=>x.getClientRects().length).length===6);
 await check('Seeker sections go back into PARAMETERS',q('#sh_speed').contains(q('#detection_params'))&&q('#sh_speed').contains(q('#flight_params'))&&visible('#normal_params')&&!visible('#pp_wasp'));
 q('[data-sky="on"]').click();q('[data-fol="on"]').click();q('[data-cam="30"]').click();
 await until(async()=>{const s=await req('/state');return s.sky_gate&&s.follower&&s.cam_fps===30});
 await check('detection controls work in Normal',(await req('/state')).sky_gate&&(await req('/state')).follower&&(await req('/state')).cam_fps===30);
 q('[data-flightmode="wasp"]').click();await until(()=>visible('#modechange'));
 await req('/set?armed=1&phase=PILOT');await until(()=>q('#mode_confirm').disabled);
 await check('arming during confirmation blocks the change',q('#mode_note').textContent.includes('Disarm')&&!q('#mode_cancel').disabled);
 q('#mode_cancel').click();await req('/set?armed=0&phase=PILOT&mode=wasp');await until(()=>q('[data-flightmode="wasp"]').classList.contains('on'));
 await check('another client changing mode updates UI without a chooser',!visible('#modechange')&&!q('.app').inert&&visible('#wasp_ui'));
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
        q = parse_qs(u.query, keep_blank_values=True)
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
                    elif k == "engine":
                        detector_settings["engine"] = int(v[0])
                    elif k == "stabilization":
                        detector_settings["video_stabilization"] = v[0] == "on"
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
                    if k in ("engage", "arm") and not why:
                        m.cycle(time.perf_counter(), context())
                        if k == "arm":  # the next cycle sees the FC armed: WASP_READY
                            m.cycle(time.perf_counter(), context())
                    elif k in ("abort", "safedown"):
                        m.cycle(time.perf_counter(), context())
                        if m.safedown.active:
                            m.safedown.stage = "STOP"
                return self.send(dict(state(), refused=why))
            if u.path == "/set":
                mapper.armed = q.get("armed", ["0"])[0] == "1"
                if not mapper.armed or q.get("idle", [""])[0] == "stop":
                    m.idle.stop()
                    m.safedown.stop()
                    m.wasp.used = False
                    m.engaged = False
                m.phase = q.get("phase", [m.phase])[0]
                if "mode" in q:
                    m.set_wasp_enabled(q["mode"][0] == "wasp", selection=True)
                if "elapsed" in q:
                    m.wasp.started = time.perf_counter()-float(q["elapsed"][0])
                    if m.wasp.acquired_t is not None:
                        m.wasp.acquired_t = m.wasp.started
                for key in ("target", "attitude"):
                    if key in q:
                        simulation[key] = q[key][0] == "1"
                if "gate" in q:
                    simulation["gate_why"] = q["gate"][0]
                for key in ("gps", "compass"):
                    if key in q:
                        simulation[key] = q[key][0] == "1"
                if "armwhy" in q:
                    simulation["arm_why"] = q["armwhy"][0]
                if "pos" in q:
                    simulation["position"] = (None if q["pos"][0] == "none" else
                                              [float(v) for v in q["pos"][0].split(",")])
                    # No GPS: the FC has no home either.
                    simulation["home"] = None if simulation["position"] is None else [37.98, 23.72]
                if not mapper.armed:
                    mapper.armed_t = None
                if "acquired" in q:
                    m.wasp.acquired_t = time.perf_counter() if q["acquired"][0] == "1" else None
                return self.send({})
            if u.path == "/commands":
                return self.send(commands)
            if u.path == "/report":
                r = json.loads(q["data"][0])
                reports.append(r)

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

    with tempfile.TemporaryDirectory(prefix=".wasp-browser-", dir=ROOT) as profile:
        with open("/tmp/wasp-firefox.log", "w") as log:
            proc = launch(profile, f"http://127.0.0.1:{srv.server_port}/h", output=log)
            try:
                finished = done.wait(80)
            finally:
                proc.terminate()
                proc.wait(timeout=10)


                for _ in range(20):
                    try:
                        shutil.rmtree(profile)
                        break
                    except OSError:
                        time.sleep(.05)
    srv.shutdown()
    assert finished, "Browser timed out; see /tmp/wasp-firefox.log"
    assert reports and all(r["ok"] for r in reports), [r for r in reports if not r["ok"]]
