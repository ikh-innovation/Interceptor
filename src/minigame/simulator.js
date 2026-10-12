'use strict';
const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const add = (a,b) => a.map((v,i)=>v+b[i]);
const sub = (a,b) => a.map((v,i)=>v-b[i]);
const mul = (a,s) => a.map(v=>v*s);
const dot = (a,b) => a.reduce((n,v,i)=>n+v*b[i],0);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm = a => Math.hypot(...a);
const unit = a => mul(a,1/(norm(a)||1));
const radians = d => d*Math.PI/180;
const smooth = v => {v=clamp(v,0,1);return v*v*(3-2*v);};
function rotate(v, axis, angle) {
  const c=Math.cos(angle),s=Math.sin(angle);
  return add(add(mul(v,c),mul(cross(axis,v),s)),mul(axis,dot(axis,v)*(1-c)));
}
function basis(elevation,heading,roll) {
  const e=radians(elevation),h=radians(heading);
  const up=[Math.sin(h)*Math.cos(e),Math.cos(h)*Math.cos(e),Math.sin(e)];
  const right=rotate([Math.cos(h),-Math.sin(h),0],up,radians(roll));
  return [right,cross(up,right),up];
}
function transform(R,v) {return add(add(mul(R[0],v[0]),mul(R[1],v[1])),mul(R[2],v[2]));}
class Flight {
  constructor(cfg) {
    this.cfg=cfg;this.t=0;this.p=[0,0,cfg.height];this.v=[0,0,0];this.R=basis(cfg.elevation,cfg.heading,cfg.roll);
    this.launchR=this.R.map(v=>v.slice());this.omega=[0,0,0];this.launched=false;this.done=false;
    this.controller=new ProfileController(cfg.profile,cfg.camera,cfg.launch);
    this.mouseBox=cfg.box.slice();this.mouseTarget=cfg.box.slice();
    if(cfg.testMode&&(cfg.centre>0||cfg.boxMotion==='mouse'))this.controller.update(cfg.box,cfg.launch);
    this.phase='READY';this.distance=0;this.path=[this.p.slice()];this.sampleAt=0;
  }
  step(dt) {
    if(this.done)return;
    this.t+=dt;
    if(this.cfg.boxMotion==='mouse'){
      // Smooth pointer samples across fixed physics steps instead of treating
      // each browser event as a one-step jump followed by a stopped box.
      const delta=sub(this.mouseTarget,this.mouseBox);
      this.mouseBox=norm(delta)<1e-6?this.mouseTarget.slice():add(this.mouseBox,mul(delta,1-Math.exp(-dt/.04)));
    }
    if(this.t<this.cfg.launch){this.phase='COUNTDOWN';return;}
    if(!this.launched){this.launched=true;this.v=mul(this.R[2],12);}
    const elapsed=this.t-this.cfg.launch;
    if(elapsed>=this.cfg.duration){this.done=true;this.phase='COMPLETE';return;}
    const p=this.cfg.profile;
    const blend=smooth((elapsed-p.straight)/p.blend);
    this.phase=elapsed<p.straight?'STRAIGHT':blend<1?'BLENDING':'FLIGHT';
    const box=this.box();
    const centered=box!==null&&Math.hypot(box[0]-.5,box[1]-.5)<=1e-9;
    // The scripted box reaching centre ends the playground's turn. Clear
    // motion history and angular motion so an old turn cannot continue.
    // Keep attitude and linear velocity; thrust/gravity/drag still apply.
    if(centered)this.omega=[0,0,0];
    let command;
    try{command=this.controller.update(centered?null:box,this.t);}
    catch(e){this.done=true;this.phase=e.message;return;}
    if(this.controller.stationary)this.omega=[0,0,0];
    // App body X=display forward, Y=display right, Z=opposite thrust.
    const requested=[command[1],command[0],-command[2]];
    for(let i=0;i<3;i++){
      const desired=requested[i];
      this.omega[i]+=(desired-this.omega[i])*dt/(0.06+dt);
    }
    const worldOmega=transform(this.R,this.omega),rate=norm(worldOmega);
    if(rate>1e-10)this.R=this.R.map(v=>rotate(v,mul(worldOmega,1/rate),rate*dt));
    const thrust=4*9.81*this.cfg.throttle;
    const acceleration=add(add(mul(this.R[2],thrust),[0,0,-9.81]),mul(this.v,-4*9.81/1600*norm(this.v)));
    this.v=add(this.v,mul(acceleration,dt));
    const old=this.p;this.p=add(this.p,mul(this.v,dt));
    if(this.p[2]<=0.18){this.p[2]=0.18;this.v=[0,0,0];this.done=true;this.phase='GROUND CONTACT';}
    this.distance+=norm(sub(this.p,old));
    if(this.t>=this.sampleAt || this.done){this.path.push(this.p.slice());this.sampleAt=this.t+0.05;}
  }
  steeringRates(){
    return this.controller.requested.slice();
  }
  moveBox(position){this.mouseTarget=position.map(v=>clamp(v,.05,.95));}
  box(){
    const cfg=this.cfg;
    if(!cfg.testMode)return null;
    if(cfg.boxMotion==='mouse')return this.mouseBox.slice();
    const f=!this.launched?0:cfg.centre===0?1:clamp((this.t-cfg.launch)/cfg.centre,0,1);
    return cfg.box.map(v=>v+(.5-v)*f);
  }
}
let settings,selected='standard',throttle=1,flight=null,running=false,paused=false,accumulator=0;
let box=[.75,.3],follow=false,orbit=-.75,tilt=.32,zoom=14;
const scene=$('scene'),ctx=scene.getContext('2d'),boxCanvas=$('bbox'),bc=boxCanvas.getContext('2d');
function number(id){const el=$(id),v=Number(el.value);if(!el.value.trim()||!Number.isFinite(v)||v<Number(el.min)||v>Number(el.max))throw Error(`${el.getAttribute('aria-label')||id}: enter ${el.min}–${el.max}.`);return v;}
function config(){return {profile:{...settings.profiles[selected]},camera:{hfov:number('hfov'),vfov:number('vfov')},throttle,launch:number('launch'),duration:number('duration'),elevation:number('elevation'),heading:number('heading'),roll:number('roll'),height:number('height'),centre:number('centre'),testMode:$('testMode').checked,boxMotion:$('boxMotion').value,box:box.slice()};}
function showSelection(){
  document.querySelectorAll('[data-profile]').forEach(b=>{const on=b.dataset.profile===selected;b.classList.toggle('on',on);b.setAttribute('aria-pressed',on);});
  document.querySelectorAll('[data-throttle]').forEach(b=>{const on=Number(b.dataset.throttle)===throttle;b.classList.toggle('on',on);b.setAttribute('aria-pressed',on);});
  const p=settings.profiles[selected];
  $('profileInfo').textContent=`${p.gain} gain · ${p.rate}°/s rate · ${p.blend} s blend · ${p.straight} s straight launch · ${Math.round(p.lead*1000)} ms lead · ${p.camera_roll}° alignment`;
}
const advancedFields=[['straight','Straight launch',' s'],['blend','Steering blend',' s'],['rate','Steering rate','°/s'],['gain','Steering gain',''],['lead','Motion lead',' ms'],['camera_roll','Camera alignment','°']];
function showAdvanced(){
  for(const [key] of advancedFields)$('profile_'+key).value=settings.profiles[selected][key];
}
function buildAdvanced(){
  for(const [key,label,unit] of advancedFields){
    const row=document.createElement('label');row.className='field';row.textContent=label;
    const select=document.createElement('select');select.id='profile_'+key;select.setAttribute('aria-label',label);
    for(const value of settings.profile_choices[key]){
      const option=document.createElement('option');option.value=value;
      option.textContent=(key==='lead'?Math.round(value*1000):value)+unit;select.append(option);
    }
    select.onchange=()=>{settings.profiles[selected][key]=Number(select.value);showSelection();preview();};
    row.append(select);$('advancedFields').append(row);
  }
  showAdvanced();
}
function preview(){if(!settings||running||paused)return;try{flight=new Flight(config());$('notice').textContent='';}catch(e){$('notice').textContent=e.message;}}
function lockSetup(lock){document.querySelectorAll('aside input,aside button,aside select').forEach(e=>e.disabled=lock);}
function reset(){endBoxDrag();running=paused=false;accumulator=0;lockSetup(false);$('play').textContent='Play flight';$('status').textContent='Ready to fly';preview();}
$('play').onclick=()=>{
  if(running){endBoxDrag();running=false;paused=true;$('play').textContent='Resume';$('status').textContent='Paused';return;}
  if(!paused){try{flight=new Flight(config());}catch(e){$('notice').textContent=e.message;return;}}
  running=true;paused=false;accumulator=0;lockSetup(true);$('notice').textContent='';$('play').textContent='Pause';$('status').textContent='Flight running';
};
$('reset').onclick=reset;
function setView(close){
  follow=close;
  for(const [id,on] of [['follow',close],['overview',!close]]){
    $(id).classList.toggle('on',on);$(id).setAttribute('aria-pressed',on);
  }
}
$('follow').onclick=()=>setView(true);
$('overview').onclick=()=>setView(false);
$('orbitReset').onclick=()=>{orbit=-.75;tilt=.32;zoom=14;};
for(const id of ['elevation','heading','roll'])$(id).addEventListener('input',()=>{
  $(id+'Value').textContent=$(id).value+'°'+(id==='heading'?[' N',' E',' S',' W'][Math.round(Number($(id).value)/90)%4]:'');
});
$('centre').addEventListener('input',()=>{if(settings&&$('centre').value!=='')settings.profiles[selected].centre_timer=Number($('centre').value);});
$('boxMotion').onchange=preview;
document.querySelectorAll('aside input').forEach(e=>e.addEventListener('input',preview));
document.addEventListener('visibilitychange',()=>{if(document.hidden&&running)$('play').click();});
function placeBox(x,y){
  if(paused||!$('testMode').checked)return false;
  const position=[clamp(x,.05,.95),clamp(y,.05,.95)];
  if(running){
    if(flight.cfg.boxMotion!=='mouse')return false;
    flight.moveBox(position);
  }else{box=position;preview();}
  return true;
}
let boxPointer=null;
function endBoxDrag(){
  const pointer=boxPointer;boxPointer=null;
  if(pointer!==null&&boxCanvas.hasPointerCapture(pointer))boxCanvas.releasePointerCapture(pointer);
}
function pointerBox(e){
  const r=boxCanvas.getBoundingClientRect();
  return placeBox((e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height);
}
boxCanvas.onpointerdown=e=>{
  if(e.button!==0||boxPointer!==null||!pointerBox(e))return;
  e.preventDefault();boxPointer=e.pointerId;boxCanvas.setPointerCapture(e.pointerId);boxCanvas.focus({preventScroll:true});
};
boxCanvas.onpointermove=e=>{if(e.pointerId===boxPointer)pointerBox(e);};
boxCanvas.onpointerup=boxCanvas.onpointercancel=e=>{if(e.pointerId===boxPointer)endBoxDrag();};
boxCanvas.onlostpointercapture=()=>{boxPointer=null;};
window.addEventListener('blur',endBoxDrag);
boxCanvas.onkeydown=e=>{
  const delta={ArrowLeft:[-.025,0],ArrowRight:[.025,0],ArrowUp:[0,-.025],ArrowDown:[0,.025]}[e.key];
  if(delta){e.preventDefault();const current=running?flight.mouseTarget:box;placeBox(current[0]+delta[0],current[1]+delta[1]);}
};
let drag=null;
scene.onpointerdown=e=>{drag=[e.clientX,e.clientY];scene.setPointerCapture(e.pointerId);};
scene.onpointermove=e=>{if(!drag)return;orbit-=(e.clientX-drag[0])*.008;tilt=clamp(tilt+(e.clientY-drag[1])*.006,.08,1.3);drag=[e.clientX,e.clientY];};
scene.onpointerup=scene.onpointercancel=()=>drag=null;
scene.addEventListener('wheel',e=>{e.preventDefault();zoom=clamp(zoom*Math.exp(e.deltaY*.001),5,180);},{passive:false});
// Procedural scenery is display-only; it never changes flight dynamics.
const environmentCanvas=document.createElement('canvas');
const environmentContext=environmentCanvas.getContext('2d');
let environmentKey='';
function drawEnvironment(w,h,eye,forward,right,up,focal){
  const ew=Math.min(640,Math.round(w)),eh=Math.max(1,Math.round(ew*h/w));
  const key=[w,h,...eye,...forward].join(',');
  if(key!==environmentKey){
    environmentKey=key;environmentCanvas.width=ew;environmentCanvas.height=eh;
    const frame=environmentContext.createImageData(ew,eh),pixels=frame.data;
    const light=unit([-.4,.3,.85]);
    for(let y=0;y<eh;y++)for(let x=0;x<ew;x++){
      const sx=(x/ew*w-w/2)/focal,sy=(h/2-y/eh*h)/focal;
      const dx=forward[0]+right[0]*sx+up[0]*sy;
      const dy=forward[1]+right[1]*sx+up[1]*sy;
      const dz=forward[2]+right[2]*sx+up[2]*sy;
      const length=Math.hypot(dx,dy,dz),alt=dz/length,az=Math.atan2(dy,dx);
      let r,g,b;
      if(dz<0){
        const t=-eye[2]/dz,gx=eye[0]+dx*t,gy=eye[1]+dy*t,range=t*length;
        const field=Math.sin(Math.floor(gx/32)*127.1+Math.floor(gy/32)*311.7)*43758.5453;
        const variation=(field-Math.floor(field))*24;
        const grain=3*Math.sin(gx*3.7+gy*2.1)*Math.sin(gy*4.3-gx*1.4);
        const rows=3*Math.sin(gx*2.3+gy*.2);
        r=79+variation+grain+rows;g=101+variation+grain+rows;b=57+variation*.5+grain;
        const path=Math.abs(gy-12-5*Math.sin(gx/85));
        if(path<1.7){r=153+grain;g=139+grain;b=109+grain;}
        const fog=1-Math.exp(-range/700);
        r+=(185-r)*fog;g+=(204-g)*fog;b+=(210-b)*fog;
      }else{
        const height=clamp(alt*2,0,1);
        r=190-131*height;g=214-77*height;b=225-35*height;
        const ridge=.009+.014*Math.pow(Math.sin(az*5+.6),2)+.009*Math.pow(Math.sin(az*13),2);
        if(alt<ridge){r=130+alt*1400;g=156+alt*1100;b=170+alt*950;}
        else{
          const cloud=Math.sin(az*12+alt*27)+.6*Math.sin(az*25-alt*61)+.3*Math.sin(az*61+alt*75);
          const haze=clamp((cloud-.65)*1.4,0,.87)*clamp((alt-.03)*10,0,1)*clamp((.65-alt)*5,0,1);
          r+=(247-r)*haze;g+=(246-g)*haze;b+=(238-b)*haze;
          const sun=(dx*light[0]+dy*light[1]+dz*light[2])/length;
          const glow=Math.pow(Math.max(0,sun),80)*.6;
          r+=(255-r)*glow;g+=(247-g)*glow;b+=(220-b)*glow;
        }
      }
      const i=(y*ew+x)*4;pixels[i]=r;pixels[i+1]=g;pixels[i+2]=b;pixels[i+3]=255;
    }
    environmentContext.putImageData(frame,0,0);
  }
  ctx.imageSmoothingEnabled=true;ctx.drawImage(environmentCanvas,0,0,w,h);
}
function render(){
  const dpr=Math.min(window.devicePixelRatio||1,2),w=scene.clientWidth,h=scene.clientHeight;
  if(scene.width!==Math.round(w*dpr)||scene.height!==Math.round(h*dpr)){scene.width=Math.round(w*dpr);scene.height=Math.round(h*dpr);}
  ctx.setTransform(dpr,0,0,dpr,0,0);
  if(!flight){ctx.fillStyle='#94b7c9';ctx.fillRect(0,0,w,h);return;}
  const p=flight.p;
  let look=add(p,[0,0,-.3]),distance=zoom;
  if(!follow){
    const low=[0,0,0],high=[0,0,flight.cfg.height];
    for(const point of [...flight.path,p])for(let axis=0;axis<3;axis++){
      low[axis]=Math.min(low[axis],point[axis]);high[axis]=Math.max(high[axis],point[axis]);
    }
    look=mul(add(low,high),.5);
    // Fit the path's bounding sphere, leaving room for telemetry and the drone.
    const radius=norm(sub(high,low))/2+3;
    distance=Math.max(14,radius/Math.sin(Math.atan(1/2.1))*1.2)*(zoom/14);
  }
  const eye=add(look,[Math.cos(orbit)*Math.cos(tilt)*distance,Math.sin(orbit)*Math.cos(tilt)*distance,Math.sin(tilt)*distance]);
  const forward=unit(sub(look,eye)),right=unit(cross(forward,[0,0,1])),up=cross(right,forward),focal=Math.min(w,h)*1.05;
  drawEnvironment(w,h,eye,forward,right,up,focal);
  function project(v){const d=sub(v,eye),z=dot(d,forward);return z>.1?[w/2+dot(d,right)*focal/z,h/2-dot(d,up)*focal/z,z]:null;}
  function line(a,b,color,width=1){const za=dot(sub(a,eye),forward),zb=dot(sub(b,eye),forward);if(za<=.11&&zb<=.11)return;if(za<=.11)a=add(a,mul(sub(b,a),(.11-za)/(zb-za)));else if(zb<=.11)b=add(b,mul(sub(a,b),(.11-zb)/(za-zb)));a=project(a);b=project(b);if(!a||!b)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}
  function polygon(vertices,color){const q=vertices.map(project);if(q.some(v=>!v))return;ctx.fillStyle=color;ctx.beginPath();q.forEach((v,i)=>i?ctx.lineTo(v[0],v[1]):ctx.moveTo(v[0],v[1]));ctx.closePath();ctx.fill();}
  // Sparse distance marks retain a readable world scale over the landscape.
  const step=20,base=p.map(v=>Math.floor(v/step)*step),extent=100;
  for(let i=-extent;i<=extent;i+=step){line([base[0]+i,base[1]-extent,.01],[base[0]+i,base[1]+extent,.01],'#c6d3a71c');line([base[0]-extent,base[1]+i,.01],[base[0]+extent,base[1]+i,.01],'#c6d3a71c');}
  const shadow=[];
  for(let i=0;i<32;i++){const a=i*Math.PI/16;shadow.push([p[0]+Math.cos(a)*1.8,p[1]+Math.sin(a)*.8,.02]);}
  polygon(shadow,`rgba(24,35,27,${.24*Math.exp(-p[2]/60)})`);
  line([0,0,.015],[8,0,.015],'#b36f4d',2);line([0,0,.015],[0,8,.015],'#548d87',2);
  for(const [text,v] of [['E',[9,0,0]],['N',[0,9,0]]]){const q=project(v);if(q){ctx.fillStyle='#304f48';ctx.font='bold 13px system-ui';ctx.fillText(text,q[0],q[1]);}}
  polygon([[-1.2,-1.2,.02],[1.2,-1.2,.02],[1.2,1.2,.02],[-1.2,1.2,.02]],'#455452');
  const platform=[0,0,flight.cfg.height];line([0,0,0],platform,'#394747',8);
  const launchUp=flight.launchR[2];line(sub(platform,mul(launchUp,.8)),add(platform,mul(launchUp,1.3)),'#64736b',12);
  line(platform,add(platform,mul(launchUp,3)),'#e9ff83',2);
  for(let i=1;i<flight.path.length;i++)line(flight.path[i-1],flight.path[i],'#dfff74',3);
  line(flight.path[flight.path.length-1],p,'#dfff74',3);
  // One-second trail markers make the curved trajectory readable in overview.
  for(let i=0;i<flight.path.length;i+=20){const q=project(flight.path[i]);if(q){ctx.fillStyle='#f1ffbd';ctx.beginPath();ctx.arc(q[0],q[1],3,0,2*Math.PI);ctx.fill();}}
  const droneScale=2.5; // Display scale only: no aerodynamic geometry is inferred.
  const local=v=>add(p,transform(flight.R,mul(v,droneScale)));
  const faces=[],sides=16,light=unit([-.4,.3,.85]);
  function face(vertices,color){
    const points=vertices.map(local),center=mul(points.reduce(add,[0,0,0]),1/points.length);
    let normal=unit(cross(sub(points[1],points[0]),sub(points[2],points[0])));
    // Two-sided shading keeps the thin fins visible throughout an orbit.
    const brightness=.4+.6*Math.abs(dot(normal,light));
    faces.push({points,depth:dot(sub(center,eye),forward),color:`rgb(${color.map(c=>Math.round(c*brightness)).join(',')})`});
  }
  const rings=[[-.85,.12],[-.65,.21],[.2,.21],[.3,.21],[.65,.16],[1.15,0]];
  for(let j=0;j<rings.length-1;j++)for(let i=0;i<sides;i++){
    const a=i*2*Math.PI/sides,b=(i+1)*2*Math.PI/sides;
    const [z0,r0]=rings[j],[z1,r1]=rings[j+1];
    const vertices=[[r0*Math.cos(a),r0*Math.sin(a),z0],[r0*Math.cos(b),r0*Math.sin(b),z0],
      [r1*Math.cos(b),r1*Math.sin(b),z1]];
    if(r1>0)vertices.push([r1*Math.cos(a),r1*Math.sin(a),z1]);
    face(vertices,j===2?[220,241,130]:j===4?[69,85,99]:[216,226,229]);
  }
  for(let i=0;i<4;i++){
    const a=i*Math.PI/2+Math.PI/4,c=Math.cos(a),s=Math.sin(a);
    face([[c*.16,s*.16,-.25],[c*.6,s*.6,-.8],[c*.6,s*.6,-.98],[c*.12,s*.12,-.85]],i%2?[193,212,219]:[225,240,156]);
  }
  const tail=[];
  for(let i=0;i<sides;i++){const a=-i*2*Math.PI/sides;tail.push([.12*Math.cos(a),.12*Math.sin(a),-.855]);}
  face(tail,[40,48,53]);
  faces.sort((a,b)=>b.depth-a.depth).forEach(f=>polygon(f.points,f.color));
  const label=project(local([0,0,1.4]));
  if(label){ctx.font='bold 11px system-ui';ctx.lineWidth=3;ctx.strokeStyle='#20364099';ctx.strokeText('WASP',label[0]-16,label[1]);ctx.fillStyle='#f2f5ed';ctx.fillText('WASP',label[0]-16,label[1]);}
  $('phase').textContent=flight.phase;
  $('time').textContent=flight.t.toFixed(1)+' s';$('altitude').textContent=p[2].toFixed(1)+' m';$('speed').textContent=norm(flight.v).toFixed(1)+' m/s';$('travel').textContent=flight.distance.toFixed(1)+' m';
  renderBox();
}
function renderBox(){
  const w=640,h=360;bc.fillStyle='#263b43';bc.fillRect(0,0,w,h);bc.strokeStyle='#91b3b21c';bc.lineWidth=1;
  for(let x=0;x<w;x+=40){bc.beginPath();bc.moveTo(x,0);bc.lineTo(x,h);bc.stroke();}for(let y=0;y<h;y+=40){bc.beginPath();bc.moveTo(0,y);bc.lineTo(w,y);bc.stroke();}
  bc.strokeStyle='#aac0c1';bc.beginPath();bc.moveTo(305,180);bc.lineTo(335,180);bc.moveTo(320,165);bc.lineTo(320,195);bc.stroke();
  const b=flight.box();
  if(b){bc.strokeStyle='#e9ff83';bc.lineWidth=2;bc.strokeRect(b[0]*w-32,b[1]*h-18,64,36);bc.fillStyle='#e9ff83';bc.font='12px system-ui';bc.fillText('TEST BOX',clamp(b[0]*w-32,4,570),Math.max(14,b[1]*h-25));}
  const rates=flight.steeringRates().map(v=>v*180/Math.PI);
  $('steeringInfo').textContent=b?`Box-motion rate request · roll ${rates[0].toFixed(1)}°/s · pitch ${rates[1].toFixed(1)}°/s (before launch blend)`: 'Test mode off · no steering request.';
  $('boxStatus').textContent=!b?'Test mode off':`Steering box · ${Math.round(b[0]*100)}%, ${Math.round(b[1]*100)}%${paused?' · paused':running?(flight.cfg.boxMotion==='mouse'?' · mouse control · drag to move':' · automatic centring'):' · drag to place'}`;
}
let previous=performance.now();
function animate(now){
  const wallDt=Math.min((now-previous)/1000,.1);previous=now;
  if(running&&flight){
    accumulator+=wallDt*Number($('playback').value);
    while(accumulator>=.005&&!flight.done){flight.step(.005);accumulator-=.005;}
    if(flight.done){endBoxDrag();running=false;lockSetup(false);$('play').textContent='Play again';$('status').textContent=flight.phase==='GROUND CONTACT'?'Flight ended · ground contact':flight.phase==='COMPLETE'?'Flight complete':flight.phase;}
    else if(!flight.launched)$('status').textContent=`Launching in ${Math.max(0,flight.cfg.launch-flight.t).toFixed(1)} s`;
    else $('status').textContent='Flight running';
  }
  render();requestAnimationFrame(animate);
}
async function init(){
  try{
    const r=await fetch('settings');if(!r.ok)throw Error('Cannot load application settings');settings=await r.json();
    $('hfov').value=settings.camera.hfov;$('vfov').value=settings.camera.vfov;
    selected=settings.selected;throttle=settings.throttle;$('launch').value=settings.launch_s;$('centre').value=settings.profiles[selected].centre_timer;
    for(const name of Object.keys(settings.profiles)){const b=document.createElement('button');b.dataset.profile=name;b.textContent=name;b.onclick=()=>{selected=name;$('centre').value=settings.profiles[name].centre_timer;showSelection();showAdvanced();preview();};$('profiles').append(b);}
    for(const value of settings.throttle_choices){const b=document.createElement('button');b.dataset.throttle=value;b.textContent=Math.round(value*100)+'%';b.onclick=()=>{throttle=value;showSelection();preview();};$('throttles').append(b);}
    $('source').textContent=settings.source;buildAdvanced();showSelection();$('play').disabled=false;reset();
  }catch(e){$('status').textContent='Unable to load settings';$('notice').textContent=e.message;}
  requestAnimationFrame(animate);
}
init();
