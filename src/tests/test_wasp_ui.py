
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


mapper = Mapper()
settings_tmp = tempfile.TemporaryDirectory(prefix="wasp-ui-settings-")
m = Mission(mapper, log=lambda *_: None, wasp_path=str(Path(settings_tmp.name)/"wasp.json"))
commands, reports = [], []
detector_settings = dict(sky_gate=True, follower=True, follower_default=True, cam_fps=30, cam_default=30, cam_busy=False)
simulation = dict(target=True, attitude=True, gate_why="")
done = threading.Event()
lock = threading.Lock()


def context():
    return dict(gate_ok=mapper.armed and not simulation["gate_why"],
                att=dict(roll=0, pitch=0, yaw=0) if simulation["attitude"] else None,
                height=0, climb=0, idle_ok=True, airborne=True,
                fc_mode="ALT_HOLD" if m.idle.active else "ACRO",
                ai_cmd=None, det_t=None, det_conf=None)


def state():
    if m.wasp.launch_t is not None:
        m.cycle(time.perf_counter(), context())
    s = m.snapshot(time.perf_counter())
    gate = simulation["gate_why"] or ("" if simulation["attitude"] else "no fresh attitude")
    s.update(gate_why=gate if mapper.armed else "disarmed", height_m=0,
             climb_mps=0, heading_deg=25,
             det_conf=.9 if simulation["target"] else None,
             det_age_s=.01 if simulation["target"] else None,
             target_fresh=simulation["target"], target_az_deg=0, target_el_deg=0,
             wasp_attitude_fresh=simulation["attitude"])
    return dict(mission=s, armed=mapper.armed, linked=True,
                fc_mode="ACRO" if m.wasp.enabled else "ALT_HOLD", control="you",
                detector="TensorRT", fps=30,
                cam_aspect=16/9 if detector_settings["cam_fps"] == 60 else 4/3, **detector_settings)


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
async function countdown(seconds){
 const wasOpen=q('#sh_speed').classList.contains('open');
 if(wasOpen)q('#sh_speed [data-close]').click();
 q('#wasp_launch_main [data-seconds="'+seconds+'"]').click();
 await until(async()=>q('#wasp_launch_main').value===String(seconds)&&(await req('/state')).mission.wasp_params.launch_s===seconds);
 if(wasOpen)parameters();
}
function tap(s){q(s).dispatchEvent(new f.contentWindow.PointerEvent('pointerdown',{bubbles:true,cancelable:true,pointerId:1}))}
async function switchMode(mode){q('[data-flightmode="'+mode+'"]').click();await until(()=>visible('#modechange'));await check('camera warning '+mode,q('#mode_degrees').textContent===(mode==='wasp'?'90°':'30°'));q('#mode_confirm').click();await until(()=>!visible('#modechange'))}
async function layout(mode){
 for(const size of [[320,640],[390,844],[820,1180],[1180,820],[1366,657]]){
  f.width=size[0];f.height=size[1];await sleep(150);
  const d=f.contentDocument, blocks=[...d.querySelectorAll('.param-block')].filter(x=>x.getClientRects().length).map(x=>x.getBoundingClientRect());
  const segs=[...d.querySelectorAll('#detection_params .seg, #detection_params .idle-input')].map(x=>x.getBoundingClientRect());
  await check(mode+' section edges '+size,blocks.every(x=>Math.abs(x.left-blocks[0].left)<1&&Math.abs(x.width-blocks[0].width)<1));
  await check(mode+' detection alignment '+size,segs.every(x=>Math.abs(x.left-segs[0].left)<1&&Math.abs(x.width-segs[0].width)<1));
  await check(mode+' no horizontal overflow '+size,d.documentElement.scrollWidth<=size[0]&&q('#sh_speed .body').scrollWidth<=q('#sh_speed .body').clientWidth);
  const buttons=[...d.querySelectorAll('#sh_speed .seg button')].filter(x=>x.getClientRects().length);
  await check(mode+' choice text fits '+size,buttons.every(x=>x.scrollWidth<=x.clientWidth+1));
  await check(mode+' equal choices '+size,[...d.querySelectorAll('#sh_speed .seg')].filter(x=>x.getClientRects().length).every(x=>{const a=[...x.children].map(b=>b.getBoundingClientRect().width);return Math.max(...a)-Math.min(...a)<1}));
  await check('Idle stays below Camera '+size,q('#idle_seconds').getBoundingClientRect().top>=q('#camseg').getBoundingClientRect().bottom);
 }
 f.width=390;f.height=844;await sleep(150);
}
async function flightLayout(){
 for(const size of [[320,640],[390,844],[568,320],[667,375],[844,390],[820,1180],[1180,820],[1366,657]]){
  f.width=size[0];f.height=size[1];await sleep(150);
  const buttons=[...q('#wasp_pad .wasp-profile-stack').querySelectorAll('button')], boxes=buttons.map(x=>x.getBoundingClientRect());
  const abort=q('#b_wasp_abort').getBoundingClientRect(), launch=q('#b_eng').getBoundingClientRect();
  await check('Wasp profiles stack vertically '+size,buttons.every(x=>x.getClientRects().length)&&boxes.every((x,i)=>!i||(x.top>=boxes[i-1].bottom&&Math.abs(x.left-boxes[0].left)<1&&Math.abs(x.width-boxes[0].width)<1)));
  await check('Wasp has aligned circular Abort and Launch only '+size,Math.abs(abort.width-abort.height)<1&&[...q('.acts').querySelectorAll('.rbtn')].filter(x=>x.getClientRects().length).length===2&&Math.abs(abort.top-launch.top)<1&&Math.abs(abort.width-launch.width)<1);
  await check('Wasp hides Seeker controls '+size,!visible('#normal_pad')&&!visible('#b_lock')&&!visible('#dial')&&!visible('#hbox')&&!visible('#tape'));
  await check('Wasp controls stay inside viewport '+size,[q('#wasp_pad'),q('.acts'),q('#vbox')].every(x=>{const b=x.getBoundingClientRect();return b.left>=0&&b.right<=size[0]+1&&b.top>=0&&b.bottom<=size[1]+1}));
  await check('Wasp control labels fit '+size,[...buttons,q('#b_wasp_abort'),q('#b_eng')].every(x=>x.scrollWidth<=x.clientWidth+1));
  const side=size[0]>=700||(size[0]>=560&&size[0]>size[1]);
  const tiles=q('.tiles').getBoundingClientRect(), camera=q('#vbox').getBoundingClientRect(), controls=q('.ctl').getBoundingClientRect();
  await check('Wasp layout matches orientation '+size,side?tiles.bottom<=camera.top&&camera.right<=controls.left:controls.top>=camera.bottom);
  if(side){
   const t=[...f.contentDocument.querySelectorAll('.tile')].map(x=>x.getBoundingClientRect()), brand=q('.brand').getBoundingClientRect();
   await check('Wasp telemetry stays in one header row '+size,t.every((x,i)=>Math.abs(x.top-t[0].top)<1&&(!i||x.left>=t[i-1].right-1))&&tiles.left>=brand.right&&tiles.top<=brand.bottom&&tiles.bottom>=brand.top);
   await check('Wasp image matches control panel height '+size,Math.abs(camera.top-controls.top)<1&&Math.abs(camera.height-controls.height)<1);
  }
  await check('Wasp main timer fits its panel '+size,q('.main-countdown').scrollWidth<=q('.main-countdown').clientWidth+1&&visible('#wasp_launch_main'));
  await check('Wasp summary bar is removed '+size,!q('#wasp_summary'));
  await check('Wasp camera uses the full frame '+size,Math.abs(q('#v').getBoundingClientRect().width-camera.width)<3&&Math.abs(q('#v').getBoundingClientRect().height-camera.height)<3);
  const throttle=q('#wasp_main_throttle').getBoundingClientRect(), extra=q('#wasp_pad [data-wasp-profile="extra"]').getBoundingClientRect();
  await check('Main throttle row sits below Extra and fits '+size,visible('#wasp_main_throttle')&&throttle.top>=extra.bottom&&throttle.bottom<=size[1]+1&&q('#wasp_main_throttle').scrollWidth<=q('#wasp_main_throttle').clientWidth+1);
 }
 f.width=390;f.height=844;await sleep(150);
}
async function main(){
 await load();
 await check('startup has no chooser or blocking popup',!q('#modechooser')&&!visible('#modechange')&&!q('.app').inert);
 await check('startup is read-only',(await req('/commands')).length===0);
 await check('Seeker keeps its original flight controls',visible('#normal_pad')&&visible('#b_lock')&&visible('#b_wasp_abort')&&visible('#dial')&&visible('#hbox')&&visible('#tape')&&!visible('#wasp_pad'));
 await check('TensorRT FPS stays green',q('#c_ai_k').textContent==='TensorRT'&&q('#c_ai').classList.contains('ok'));
 parameters();
 await check('three Seeker parameter sections in order',[...f.contentDocument.querySelectorAll('.param-title')].filter(x=>x.getClientRects().length).map(x=>x.textContent).join('|')==='1Flight mode|2Detection|3Flight parametersSEEKER');
 await check('Seeker ON and Wasp OFF',q('[data-flightmode="normal"]').textContent.startsWith('Seeker drone')&&q('[data-flightmode="normal"]').getAttribute('aria-pressed')==='true'&&q('[data-flightmode="wasp"]').getAttribute('aria-pressed')==='false');
 await check('detection always visible',visible('#skyseg')&&visible('#folseg')&&visible('#camseg'));
 await check('Idle default is 10 seconds and Search removed',q('#idle_seconds').value==='10'&&!q('[data-g="search"]'));
 const field=q('#idle_seconds');field.focus();field.value='12.5';await sleep(400);
 await check('polling preserves typed Idle value',field.value==='12.5');
 field.dispatchEvent(new f.contentWindow.Event('change',{bubbles:true}));field.blur();
 await until(()=>q('#idle_seconds').value==='12.5');
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
 await check('simple steering is visible and advanced controls start collapsed',visible('#wasp_profiles')&&!q('#wasp_advanced').open&&!visible('[data-wp="blend"]')&&!visible('[data-wp="gain"]')&&visible('[data-wp="throttle"]'));
 await check('Motion assist is visible and defaults OFF',visible('[data-wp="motion_assist"]')&&q('[data-wp="motion_assist"] [data-wv="0"]').getAttribute('aria-pressed')==='true');
 q('[data-wp="motion_assist"] [data-wv="1"]').click();await until(()=>q('[data-wp="motion_assist"] [data-wv="1"]').getAttribute('aria-pressed')==='true');
 await check('Motion assist applies to the backend',(await req('/state')).mission.wasp_params.motion_assist===1);
 q('#wasp_profiles [data-wasp-profile="extra"]').click();await until(()=>q('#wasp_profiles [data-wasp-profile="extra"]').classList.contains('on'));
 await check('Motion assist survives profile changes',(await req('/state')).mission.wasp_params.motion_assist===1);
 q('[data-wp="motion_assist"] [data-wv="0"]').click();await until(()=>q('[data-wp="motion_assist"] [data-wv="0"]').getAttribute('aria-pressed')==='true');
 q('#wasp_profiles [data-wasp-profile="standard"]').click();await until(()=>q('#wasp_profiles [data-wasp-profile="standard"]').classList.contains('on'));
 await check('Standard is selected from actual backend settings',q('[data-wasp-profile="standard"]').getAttribute('aria-pressed')==='true'&&q('#wasp_profile_help').textContent.includes('Standard defaults'));
 const profileStart=(await req('/commands')).length;
 q('[data-wasp-profile="smooth"]').click();await until(()=>q('[data-wasp-profile="smooth"]').classList.contains('on'));
 const smooth=(await req('/state')).mission.wasp_params;
 await check('one profile command applies all three steering values',smooth.blend===1&&smooth.rate===45&&smooth.gain===2&&(await req('/commands')).length===profileStart+1);
 q('[data-wasp-profile="responsive"]').click();await until(()=>q('[data-wasp-profile="responsive"]').classList.contains('on'));
 const responsive=(await req('/state')).mission.wasp_params;
 await check('Responsive applies together without changing launch throttle',responsive.blend===0.5&&responsive.rate===90&&responsive.gain===4&&responsive.throttle===smooth.throttle);
 q('#wasp_profiles [data-wasp-profile="extra"]').click();await until(()=>q('#wasp_pad [data-wasp-profile="extra"]').classList.contains('on'));
 const extra=(await req('/state')).mission.wasp_params;
 await check('Extra applies exact steering settings from Parameters',extra.blend===0.25&&extra.rate===120&&extra.gain===6&&extra.lead===0.08&&extra.camera_roll===0);
 await check('Launch countdown is only controlled from the main screen',!q('#wasp_launch_seconds')&&q('#wasp_launch_main').value==='3');
 await countdown(4);
 await check('manual timer keeps selected steering profile',(await req('/state')).mission.wasp_steering_profile==='extra');
 q('[data-wasp-profile="standard"]').click();await until(()=>q('[data-wasp-profile="standard"]').classList.contains('on'));
 await check('launch countdown stays the same when steering changes',(await req('/state')).mission.wasp_params.launch_s===4);
 q('#wasp_profiles [data-wasp-profile="extra"]').click();await until(()=>q('#wasp_profiles [data-wasp-profile="extra"]').classList.contains('on')&&q('#wasp_launch_main').value==='4');
 await check('Extra uses the same shared timer',(await req('/state')).mission.wasp_steering_profile==='extra');
 q('#wasp_profiles [data-wasp-profile="standard"]').click();await until(()=>q('#wasp_profiles [data-wasp-profile="standard"]').classList.contains('on')&&q('#wasp_launch_main').value==='4');
 await check('main screen shows active steering and timer before launch',q('#wasp_pad [data-wasp-profile="standard"]').classList.contains('on')&&q('#wasp_launch_main').value==='4');
 q('#wasp_advanced').open=true;
 await check('Wasp has motion assist plus seven existing parameter groups and no old timers',q('#pp_wasp').querySelectorAll('[data-wp]').length===8&&!q('[data-wp="duration"]')&&!q('[data-wp="no_det"]'));
 await check('Idle is shared across modes',q('#idle_seconds').value==='12.5');
 q('#idle_seconds').focus();q('#idle_seconds').value='7';q('#idle_seconds').dispatchEvent(new f.contentWindow.Event('change',{bubbles:true}));
 q('#idle_seconds').dispatchEvent(new f.contentWindow.KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
 await until(()=>q('#idle_seconds').value==='7'&&q('[data-wp="gain"] [data-wv="2"]').classList.contains('on'));
 await check('Wasp has gain 2 by default',q('[data-wp="gain"] [data-wv="2"]').classList.contains('on'));
 q('[data-wp="rate"] [data-wv="45"]').click();await until(()=>q('[data-wp="rate"] [data-wv="45"]').classList.contains('on'));
 await check('Idle and 45-degree choices apply to backend',(await req('/state')).mission.idle_s===7&&(await req('/state')).mission.wasp_params.rate===45);
 await check('edited steering becomes selected profile default',q('#wasp_profile_help').textContent.includes('Standard defaults')&&q('#wasp_profiles [data-wasp-profile="standard"]').classList.contains('on'));
 await check('Enter ends Idle editing',f.contentDocument.activeElement!==q('#idle_seconds'));
 await layout('wasp');
 q('[data-sky="off"]').click();await until(()=>q('[data-sky="off"]').classList.contains('on'));
 q('[data-fol="off"]').click();await until(()=>q('[data-fol="off"]').classList.contains('on'));
 q('[data-cam="60"]').click();await until(()=>q('[data-cam="60"]').classList.contains('on'));
 await check('detection controls work in Wasp',!(await req('/state')).sky_gate&&!(await req('/state')).follower&&(await req('/state')).cam_fps===60);
 await check('Wasp throttle choices are 75, 85 and 100 percent',[...q('[data-wp="throttle"]').children].map(b=>b.textContent).join(',')==='75%,85%,100%');
 q('[data-wp="throttle"] [data-wv="0.85"]').click();await until(()=>q('[data-wp="throttle"] [data-wv="0.85"]').classList.contains('on'));
 await check('85 percent throttle applies to backend',(await req('/state')).mission.wasp_params.throttle===.85);
 q('[data-wp="throttle"] [data-wv="0.75"]').click();await until(()=>q('[data-wp="throttle"] [data-wv="0.75"]').classList.contains('on'));
 q('[data-wp="straight"] [data-wv="0.5"]').click();await until(()=>q('[data-wp="straight"] [data-wv="0.5"]').classList.contains('on'));
 q('[data-wp="blend"] [data-wv="0.25"]').click();
 await until(()=>q('[data-wp="blend"] [data-wv="0.25"]').classList.contains('on'));
 await countdown(4);
 const count=(await req('/commands')).length;
 await load();
 await check('refresh has no popup and sends no mode command',!visible('#modechange')&&!q('.app').inert&&(await req('/commands')).length===count);
 parameters();
 await check('refresh preserves Wasp mode',q('[data-flightmode="wasp"]').classList.contains('on'));
 await check('refresh preserves throttle',q('[data-wp="throttle"] [data-wv="0.75"]').classList.contains('on'));
 await check('refresh preserves Idle, rate and gain',q('#idle_seconds').value==='7'&&q('[data-wp="rate"] [data-wv="45"]').classList.contains('on')&&q('[data-wp="gain"] [data-wv="2"]').classList.contains('on'));
 await check('refresh preserves edited profile and collapsed advanced view',q('#wasp_pad [data-wasp-profile="standard"]').classList.contains('on')&&!q('#wasp_advanced').open);
 await check('refresh preserves camera confirmation',q('#mountstatus').textContent==='90° ✓');
 await check('refresh preserves manual launch countdown',q('#wasp_launch_main').value==='4');
 await req('/set?armed=1&phase=WASP_READY');await until(()=>q('[data-flightmode="normal"]').disabled);
 await check('Idle locked while armed',q('#idle_seconds').disabled);
 await check('launch countdown editable while armed and ready',!q('#wasp_launch_main').disabled);
 await check('Wasp profiles and parameters editable while ready',[...f.contentDocument.querySelectorAll('[data-wp] button, [data-wasp-profile]')].every(x=>!x.disabled));
 q('[data-flightmode="normal"]').click();await sleep(100);
 await check('armed mode click sends nothing',(await req('/commands')).length===count&&!visible('#modechange'));
 q('#sh_speed [data-close]').click();
 await flightLayout();
 q('#wasp_pad [data-wasp-profile="extra"]').click();await until(()=>q('#wasp_pad [data-wasp-profile="extra"]').classList.contains('on'));
 await check('Extra can be selected while armed and ready',(await req('/state')).mission.wasp_params.rate===120);
 q('#wasp_pad [data-wasp-profile="standard"]').click();await until(()=>q('#wasp_launch_main').value==='4'&&q('#wasp_pad [data-wasp-profile="standard"]').classList.contains('on'));
 await check('switching back restores edited advanced defaults',(await req('/state')).mission.wasp_params.rate===45&&(await req('/state')).mission.wasp_params.blend===0.25);
 await check('Main throttle row offers all three choices',[...q('#wasp_main_throttle').children].map(b=>b.textContent).join(',')==='75%,85%,100%');
 q('#wasp_main_throttle [data-wv="0.85"]').click();await until(()=>q('#pp_wasp [data-wp="throttle"] [data-wv="0.85"]').classList.contains('on'));
 for(const profile of ['smooth','responsive','extra','standard']){
  q('#wasp_pad [data-wasp-profile="'+profile+'"]').click();await until(()=>q('#wasp_pad [data-wasp-profile="'+profile+'"]').classList.contains('on'));
  await check('Throttle and launch countdown stay independent of '+profile,q('#wasp_main_throttle [data-wv="0.85"]').classList.contains('on')&&q('#wasp_launch_main').value==='4'&&(await req('/state')).mission.wasp_params.throttle===.85);
 }
 q('#wasp_main_throttle [data-wv="0.75"]').click();await until(()=>q('#pp_wasp [data-wp="throttle"] [data-wv="0.75"]').classList.contains('on'));
 await check('main timer is a scroll picker without text input',q('#wasp_launch_main').getAttribute('role')==='listbox'&&!q('.main-countdown input'));
 const wheel=q('#wasp_launch_main'), rows=[...wheel.children], index=rows.findIndex(x=>x.getAttribute('data-seconds')==='5');
 wheel.dispatchEvent(new f.contentWindow.WheelEvent('wheel',{deltaY:50,bubbles:true}));wheel.scrollTop=index*rows[0].getBoundingClientRect().height;
 await until(async()=>q('#wasp_launch_main').value==='5'&&(await req('/state')).mission.wasp_params.launch_s===5);
 await check('rolling the main timer saves the countdown',(await req('/state')).mission.wasp_params.launch_s===5);
 await countdown(4);
 await check('Wasp aim and launch controls remain correct',q('#aim').style.top==='50%'&&!visible('#arc')&&q('#b_up').disabled&&!q('#b_eng').disabled);
 await req('/set?armed=1&target=0');await until(()=>q('#c_det').textContent==='NO');
 await check('launch is available without an initial detection',!q('#b_eng').disabled);
 await req('/set?armed=1&attitude=0');await until(()=>q('#b_eng').disabled);
 await check('launch still requires fresh attitude',q('#b_eng').disabled);
 await req('/set?armed=1&attitude=1&gate=wrong-mode');await until(()=>q('#badge').textContent==='wrong-mode');
 await check('launch still requires the controller gate',q('#b_eng').disabled);
 await req('/set?armed=1&gate=');await until(()=>!q('#b_eng').disabled);
 parameters();
 await check('Test mode follows flight parameters',q('#test_mode_params').getBoundingClientRect().top>=q('#flight_params').getBoundingClientRect().bottom);
 await check('Centre timer defaults to 3 and is editable while armed',q('#test_centre_timer').value==='3'&&!q('#test_centre_timer').disabled);
 q('#test_centre_timer').value='1.5';q('#test_centre_timer').dispatchEvent(new f.contentWindow.Event('change',{bubbles:true}));
 await until(async()=>(await req('/state')).mission.wasp_params.centre_timer===1.5);
 await check('Centre timer is applied to backend',(await req('/state')).mission.wasp_params.centre_timer===1.5);
 const centreCommands=(await req('/commands')).length;
 q('#test_centre_timer').value='-1';q('#test_centre_timer').dispatchEvent(new f.contentWindow.Event('change',{bubbles:true}));
 await check('Invalid centre timer sends nothing',q('#test_centre_timer').value==='1.5'&&(await req('/commands')).length===centreCommands);
 q('[data-testmode="on"]').click();await until(()=>!q('#sh_speed').classList.contains('open')&&visible('#test_prompt'));
 await check('Test mode returns to image and requires a target',(await req('/state')).mission.test_mode&&q('#b_eng').disabled);
 function point(x,y){const r=q('#v').getBoundingClientRect();q('#vbox').dispatchEvent(new f.contentWindow.PointerEvent('pointerdown',{bubbles:true,cancelable:true,pointerId:3,clientX:r.left+r.width*x,clientY:r.top+r.height*y}))}
 point(.7,.3);await until(()=>visible('#test_target'));
 const box=q('#test_target').getBoundingClientRect(), frame=q('#vbox');
 await check('Test target has one tenth image dimensions',Math.abs(box.width-frame.clientWidth*.1)<1&&Math.abs(box.height-frame.clientHeight*.1)<1);
 point(.25,.75);await until(()=>q('#test_target').style.left==='25%'&&q('#test_target').style.top==='75%');
 await check('touching again moves the fixed target',(await req('/state')).mission.test_target.x===.25&&(await req('/state')).mission.test_target.y===.75);
 await flightLayout();
 await check('rotating preserves the test target',q('#test_target').style.left==='25%'&&q('#test_target').style.top==='75%');
 tap('#b_eng');await until(()=>q('#b_eng').classList.contains('countdown'));
 await check('launch countdown runs in the backend and blocks repeated launches',(await req('/state')).mission.wasp_progress.stage==='COUNTDOWN'&&q('#b_eng').disabled);
 await check('settings lock during countdown',q('#wasp_launch_main').disabled&&q('#test_centre_timer').disabled&&q('#wasp_pad [data-wasp-profile="extra"]').disabled);
 await check('Motion assist locks during countdown',q('[data-wp="motion_assist"] [data-wv="1"]').disabled);
 const targetBefore=(await req('/state')).mission.test_target;point(.8,.2);await sleep(200);
 await check('Test target stays fixed during countdown',JSON.stringify((await req('/state')).mission.test_target)===JSON.stringify(targetBefore));
 await load();
 await check('refresh preserves launch countdown',q('#b_eng').classList.contains('countdown')&&(await req('/commands')).filter(x=>x[0]==='engage').length===1);
 tap('#b_wasp_abort');await until(()=>q('#engtxt').textContent==='IDLE');
 await check('Abort cancels the launch countdown',(await req('/state')).mission.wasp_progress.stage!=='COUNTDOWN');
 await req('/set?armed=0&phase=PILOT');await sleep(350);parameters();
 await countdown(3);
 q('#sh_speed [data-close]').click();await req('/set?armed=1&phase=WASP_READY');await until(()=>!q('#b_eng').disabled);
 tap('#b_eng');await until(()=>q('#engtxt strong')&&q('#engtxt strong').textContent==='3');
 await until(()=>q('#engtxt strong')&&q('#engtxt strong').textContent==='2');
 await until(()=>q('#engtxt strong')&&q('#engtxt strong').textContent==='1');
 await check('default countdown displays 3, 2, 1',q('#b_eng').disabled);
 await until(()=>q('#engtxt').textContent==='WASP FLYING');
 await check('initial acquisition shows the total countdown',q('#badge').textContent.includes('STRAIGHT')&&q('#badge').textContent.includes(' s'));
 await until(()=>parseFloat(q('#test_target').style.left)>25&&parseFloat(q('#test_target').style.left)<50);
 await check('Test target moves toward the center only after launch',parseFloat(q('#test_target').style.top)<75&&parseFloat(q('#test_target').style.top)>50);
 await load();
 await check('Refresh preserves Centre timer and motion',q('#test_centre_timer').value==='1.5'&&(await req('/state')).mission.test_centre_remaining_s<1.5);
 await until(()=>q('#test_target').style.left==='50%'&&q('#test_target').style.top==='50%');
 await sleep(250);
 await check('Test target stays centered',q('#test_target').style.left==='50%'&&q('#test_target').style.top==='50%');
 await req('/set?armed=1&phase=WASP_FLIGHT&elapsed=2');await until(()=>q('#badge').textContent.includes('STRAIGHT · 4.'));
 await check('no-detection wait remains active after Straight launch',q('#b_eng').disabled);
 await req('/set?armed=1&target=1&acquired=1');await until(()=>q('#badge').textContent.includes('BLENDING'));
 await check('late acquisition shows the steering transition',q('#badge').textContent.includes('BLENDING'));
 await req('/set?armed=1&phase=WASP_FLIGHT&elapsed=1.25');await until(()=>q('#badge').textContent.includes('GUIDING'));
 await check('each launch attempt sent once',(await req('/commands')).filter(x=>x[0]==='engage').length===2);
 await load();
 await check('armed refresh preserves the active flight',!visible('#modechange')&&!q('.app').inert&&(await req('/state')).mission.phase==='WASP_FLIGHT'&&(await req('/commands')).filter(x=>x[0]==='engage').length===2);
 parameters();await check('camera switching blocked during Wasp flight',q('[data-cam="30"]').disabled&&q('[data-cam="60"]').disabled);
 q('#sh_speed [data-close]').click();tap('#b_wasp_abort');await until(()=>q('#engtxt').textContent==='IDLE');
 await check('abort remains available and sends once per tap',(await req('/commands')).filter(x=>x[0]==='abort').length===2&&q('#b_eng').disabled);
 await check('Idle identifies height hold and drift',q('#badge').textContent==='IDLE · HOLD · XY DRIFT');
 await req('/set?armed=0&phase=PILOT');await sleep(350);parameters();
 q('[data-cam="30"]').click();await until(()=>q('[data-cam="30"]').classList.contains('on'));
 q('#sh_speed [data-close]').click();await flightLayout();parameters();
 await switchMode('normal');
 await check('leaving WASP disables Test mode',!(await req('/state')).mission.test_mode&&!visible('#test_target'));
 await check('Abort is available in Seeker too',visible('#b_wasp_abort'));
 await check('returning to Seeker restores flight controls',visible('#normal_pad')&&visible('#b_lock')&&visible('#dial')&&visible('#hbox')&&visible('#tape')&&!visible('#wasp_pad'));
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
                        m.cycle(time.perf_counter(), context())
                    elif k == "abort":
                        m.cycle(time.perf_counter(), context())
                        m.idle.stage = "HOLD"
                return self.send(dict(state(), refused=why))
            if u.path == "/set":
                mapper.armed = q.get("armed", ["0"])[0] == "1"
                if not mapper.armed:
                    m.idle.stop()
                    m.wasp.used = False
                    m.engaged = False
                m.phase = q.get("phase", [m.phase])[0]
                if "mode" in q:
                    m.set_wasp_enabled(q["mode"][0] == "wasp", selection=True)
                if "elapsed" in q:
                    m.wasp.started = time.perf_counter()-float(q["elapsed"][0])
                    m._run_t = m.wasp.started
                    if m.wasp.acquired_t is not None:
                        m.wasp.acquired_t = m.wasp.started
                for key in ("target", "attitude"):
                    if key in q:
                        simulation[key] = q[key][0] == "1"
                if "gate" in q:
                    simulation["gate_why"] = q["gate"][0]
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
