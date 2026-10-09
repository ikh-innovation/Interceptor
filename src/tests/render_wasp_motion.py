#!/usr/bin/env python3
"""Reproducible side-by-side WASP steering simulation, using production code.

This is an angular camera-tracking demonstration, not an airframe/trajectory
prediction. Detector observations and camera vibration are synthetic. Both
sides use the same target, noise, outliers, dropouts and rotational plant.
"""

import argparse
from collections import deque
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from follower import TargetFollower
from mission import Mission
from tracker import DroneTracker, KalmanTracker
from wasp import Wasp, WaspGuidance, camera_ray, rotation, unit, cross, rotate
from wasp_motion import WaspMotionEstimator, project_bearing

W, H = 640, 480
TH, TV = math.tan(math.radians(31.1)), math.tan(math.radians(24.4))
BLUE, GREEN, YELLOW, RED = (255, 194, 85), (134, 232, 122), (74, 206, 255), (97, 92, 244)
WHITE, DIM = (241, 239, 229), (164, 151, 128)
SCENES = ((0, 6, '01  Small target / detector jitter'),
          (6, 11, '02  Camera vibration / moving sky'),
          (11, 16, '03  Single-frame false boxes'),
          (16, 21, '04  Target changes direction'),
          (21, 25, '05  Short detection gaps'),
          (25, 30, '06  Tiny target / flow fallback'))


def scene(t):
    return next((i, name) for i, (a,b,name) in enumerate(SCENES) if a <= t < b)


def target_world(t):
    if t < 16:
        x, y = 52 + 15*math.sin(t*.9), -27 + 12*math.sin(t*.7)
    elif t < 21:
        u=t-16
        x=52+15*math.sin(16*.9)+55*math.sin(u*1.45)
        y=-27+12*math.sin(16*.7)+30*math.sin(u*1.7)
    else:
        x=52+15*math.sin(16*.9)+55*math.sin(5*1.45)+20*math.sin((t-21)*.8)
        y=-27+12*math.sin(16*.7)+30*math.sin(5*1.7)+10*math.sin((t-21)*.7)
    return camera_ray(x,y,W,H,TH,TV)


def sky(seed):
    rng=np.random.default_rng(seed)
    h,w=H+300,W+500
    clouds=np.zeros((h,w),np.float32)
    for scale,weight in ((80,1),(35,.4),(12,.08)):
        f=rng.normal(size=(h//scale+2,w//scale+2)).astype(np.float32)
        clouds += weight*cv2.resize(f,(w,h),interpolation=cv2.INTER_CUBIC)
    clouds=np.maximum(0,clouds-.1)
    clouds=cv2.GaussianBlur(clouds,(0,0),3)
    clouds=np.clip(clouds/(clouds.max()+1e-9),0,1)[...,None]
    gradient=np.linspace(0,1,h)[:,None,None]
    base=np.array([204,143,86])[None,None,:]+gradient*np.array([34,51,65])[None,None,:]
    base=np.broadcast_to(base,(h,w,3))
    return np.uint8(np.clip(base*(1-.65*clouds)+np.array([247,244,236])*.65*clouds,0,255))


def mosquito(frame, x, y, size):
    # Render at 4x for subpixel motion of a tiny dark body and pale wings.
    k=max(20,size*3)
    mask=np.zeros((k*4,k*4),np.uint8)
    c=k*2;s=max(3,size*2)
    cv2.ellipse(mask,(c,c),(max(2,s//5),max(3,s//2)),12,0,360,255,-1)
    cv2.ellipse(mask,(c-s//3,c-s//7),(max(2,s//3),max(2,s//5)),-25,0,360,145,-1)
    cv2.ellipse(mask,(c+s//3,c-s//7),(max(2,s//3),max(2,s//5)),25,0,360,145,-1)
    for sign in (-1,1):
        for offset in (-s//3,0,s//3):
            cv2.line(mask,(c,c+offset),(c+sign*s//2,c+offset+s//4),190,1)
    alpha=cv2.resize(mask,(k,k),interpolation=cv2.INTER_AREA).astype(float)/255
    x0,y0=math.floor(x-k/2),math.floor(y-k/2)
    alpha=cv2.warpAffine(alpha,np.float32([[1,0,x-k/2-x0],[0,1,y-k/2-y0]]),(k,k))
    xa,ya,xb,yb=max(0,x0),max(0,y0),min(W,x0+k),min(H,y0+k)
    if xa < xb and ya < yb:
        a=alpha[ya-y0:yb-y0,xa-x0:xb-x0,None]
        frame[ya:yb,xa:xb]=frame[ya:yb,xa:xb]*(1-a)+np.array([25,30,37])*a


class RatesMapper:
    armed=False
    camera_mount_deg=90

    @staticmethod
    def wasp_pwm(rates, throttle):
        return rates


class CameraSimulation:
    def __init__(self, assist, params, follow):
        self.assist=assist
        self.m=Mission(RatesMapper())
        self.m.set_wasp_enabled(True)
        self.m.wasp.params.update(params,motion_assist=float(assist))
        self.w=self.m.wasp
        self.g=WaspGuidance()
        self.mat=np.eye(3)
        self.rate=np.zeros(3)
        self.command=np.zeros(3)
        self.pending=deque()
        self.observation=None
        self.det_t=None
        self.last_cmd=np.zeros(3)
        self.next_send=3.0
        self.rows=[]
        t=DroneTracker.__new__(DroneTracker)
        t.fw,t.fh,t.full_w,t.full_h=W,H,W,H
        t.ref_display=(W,H);t.MIN_ROI_HALF=50;t.MISS_TO_SEARCH=20;t.MISS_TO_PREDICT=2
        t.kalman=KalmanTracker();t.motion=WaspMotionEstimator()
        t._motion_enabled=assist;t._motion_observation=None
        t.follower=TargetFollower();t.use_follower=follow;t.follow_hits=0
        t.use_sky_mask=False;t.sky_mask=None;t._ground_rejected=False
        t.state,t.missed,t.last_bbox,t.pos='SEARCH',0,None,None
        t.layer,t.conf,t.gate_note='SEARCH',None,''
        t._frame_idx=0;t._t_last_frame=None;t._frame_dt=None;t._ui_fps=30;t.cam_mode=30
        t._gate_rejects,t.gated_total=0,0;t.mode_name=self.m.ai_level_now()
        t.ap=SimpleNamespace(mission=self.m,law=SimpleNamespace(_tan_h=TH,_tan_v=TV),
            H_FOV=math.radians(62.2),V_FOV=math.radians(48.8),
            wasp_motion_context=lambda capture_t:(self.capture_att,TH,TV))
        self.tracker=t
        self.raw=None;self.used=None;self.visible=True
        self.source='SEARCH';self.raw_outlier=False;self.processing_ms=[]

    def physics(self,t,dt):
        while self.pending and self.pending[0][0] <= t+1e-9:
            _, self.observation, self.det_t=self.pending.popleft()
        if t >= 3 and self.w.started is None:
            self.w.begin(3,target_acquired=False)
        if t >= self.next_send-1e-9 and self.w.started is not None:
            self.next_send += .02
            att={'rotation':self.mat}
            ctx=dict(att=att,ai_cmd=self.observation,det_t=self.det_t,det_conf=.8)
            if not self.w.fault(t,ctx,.2):
                if self.w.acquired_t is None:
                    self.w.acquire(t,ctx)
            else:
                self.w.lose_target()
            self.command=np.array(self.w.output(RatesMapper(),t,ctx))
        self.rate += dt/(.08+dt)*(self.command-self.rate)
        angle=float(np.linalg.norm(self.rate))*dt
        if angle > 1e-12:
            axis=self.rate/np.linalg.norm(self.rate)
            k=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
            turn=np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)
            self.mat=self.mat@turn

    def capture(self,t,noise,shake,drop,outlier,size,background,fps):
        self.capture_att={'rotation':self.mat.copy()}
        centre=project_bearing(target_world(t),self.capture_att,W,H,TH,TV)
        if centre is None:
            raise ValueError('Simulated target left guidance cone')
        tx,ty=centre[0]+shake[0],centre[1]+shake[1]
        # Sky follows the camera. This procedural scene is not flight footage.
        down,right,forward=rotate(self.mat,(0,0,-1))
        sx,sy=250+int(right*W/(2*TH))-int(shake[0]),150+int(down*H/(2*TV))-int(shake[1])
        sx,sy=max(0,min(background.shape[1]-W,sx)),max(0,min(background.shape[0]-H,sy))
        frame=background[sy:sy+H,sx:sx+W].copy()
        mosquito(frame,tx,ty,size)
        det_x,det_y=tx+noise[0],ty+noise[1]
        if outlier:
            det_x+=37;det_y-=19
        detection=None if drop else (int(round(det_x)),int(round(det_y)),
            int(round(det_x-size/2)),int(round(det_y-size/2)),size,size,.8)
        self.raw=None if detection is None else detection[:2]+(size,size)
        self.raw_outlier=outlier and not drop
        def detect(_,region,imgsz,enhance):
            if detection is not None and region[0]<=detection[0]<region[2] and region[1]<=detection[1]<region[3]:
                return detection
            return None
        self.tracker._detect_region=detect
        self.tracker._frame_idx+=1;self.tracker._ui_fps=fps;self.tracker.cam_mode=fps
        import time
        start=time.perf_counter()
        with patch('tracker.perf_counter',return_value=t):
            fused,_,_,_,detected=self.tracker._process(frame,frame,capture_t=t)
        self.processing_ms.append((time.perf_counter()-start)*1000)
        if fused is not None and detected:
            if self.assist:
                cmd=self.tracker._motion_observation
            else:
                cmd=self.g.observe(fused[0]-W/2,fused[1]-H/2,W,H,TH,TV,self.capture_att,t)
                cmd['camera_roll']=0
            self.pending.append((t+.060,cmd,t+.060))
        elif fused is None:
            self.g.reset()
            self.pending.append((t+.060,None,self.det_t))
        self.used=None if self.tracker.last_bbox is None else (
            self.tracker.last_bbox[0]+self.tracker.last_bbox[2]/2,
            self.tracker.last_bbox[1]+self.tracker.last_bbox[3]/2,
            self.tracker.last_bbox[2],self.tracker.last_bbox[3])
        self.source=self.tracker.layer
        row=dict(t_s=round(t,4),scene=scene(t)[0],motion_assist=self.assist,
            line_of_sight_error_px=round(math.hypot(centre[0]-W/2,centre[1]-H/2),4),
            roll_rate_dps=math.degrees(self.command[0]),pitch_rate_dps=math.degrees(self.command[1]),
            estimate_source=self.source,outlier=int(self.raw_outlier),detector_gap=int(drop),
            accepted=int(detected),flow_quality=self.tracker.motion.flow_quality if self.assist else 0)
        self.rows.append(row)
        return frame,(tx,ty,size,size),row


def text(frame,message,point,size=.5,color=WHITE,thickness=1):
    cv2.putText(frame,message,point,cv2.FONT_HERSHEY_SIMPLEX,size,color,thickness,cv2.LINE_AA)


def box(frame,b,color,thickness=1):
    if b is None:return
    x,y,w,h=b
    cv2.rectangle(frame,(round(x-w/2),round(y-h/2)),(round(x+w/2),round(y+h/2)),color,thickness)


def chart(canvas,rows,offset,t,color,key,maximum,title):
    x,y=offset;cw,ch=600,48
    text(canvas,title,(x,y-7),.4,DIM)
    cv2.rectangle(canvas,(x,y),(x+cw,y+ch),(63,47,35),1)
    recent=[r for r in rows if t-5<=r['t_s']<=t]
    points=[]
    for r in recent:
        px=x+round((r['t_s']-max(0,t-5))/5*cw)
        py=y+ch-round(min(maximum,abs(r[key]))/maximum*ch)
        points.append((px,py))
    if len(points)>1:cv2.polylines(canvas,[np.array(points)],False,color,1,cv2.LINE_AA)


def panel(frame,truth,sim,label,t):
    f=frame.copy()
    cx,cy=W//2,H//2
    cv2.line(f,(cx-14,cy),(cx+14,cy),WHITE,1);cv2.line(f,(cx,cy-14),(cx,cy+14),WHITE,1)
    box(f,truth,GREEN);box(f,sim.raw,RED if sim.raw_outlier else YELLOW);box(f,sim.used,BLUE,2)
    if sim.raw_outlier and sim.raw:
        x,y=map(round,sim.raw[:2]);text(f,'OUTLIER',(min(W-95,x+10),max(20,y)),.4,RED,1)
    # Equal magnification makes the very small target and bbox jitter visible.
    tx,ty=truth[:2];x=max(0,min(W-80,round(tx)-40));y=max(0,min(H-60,round(ty)-30))
    inset=cv2.resize(f[y:y+60,x:x+80],(240,180),interpolation=cv2.INTER_NEAREST)
    f[15:195,W-255:W-15]=inset
    cv2.rectangle(f,(W-255,15),(W-15,195),BLUE,1)
    text(f,'3x TARGET DETAIL',(W-245,215),.4,(48,53,61),1)
    cv2.rectangle(f,(10,H-57),(W-10,H-10),(31,24,16),-1)
    text(f,sim.source,(20,H-36),.43,BLUE)
    text(f,('COUNTDOWN '+str(max(1,3-int(t)))) if t<3 else
         'ROLL %+.1f / PITCH %+.1f deg/s'%tuple(np.degrees(sim.command[:2])),(20,H-19),.43,WHITE)
    return f


def metrics(sim,fps):
    rows=[r for r in sim.rows if r['t_s']>=4]
    commands=np.array([[r['roll_rate_dps'],r['pitch_rate_dps']] for r in rows])
    result=dict(motion_assist=sim.assist,
        rms_line_of_sight_error_px=float(np.sqrt(np.mean([r['line_of_sight_error_px']**2 for r in rows]))),
        rms_command_change_dps=float(np.sqrt(np.mean(np.square(np.diff(commands,axis=0))))),
        mean_processing_ms=float(np.mean(sim.processing_ms)),
        p95_processing_ms=float(np.percentile(sim.processing_ms,95)),
        flow_frames=sum('FLOW' in r['estimate_source'] for r in rows),
        outlier_frames=sum(r['outlier'] for r in rows),
        accepted_outlier_frames=sum(r['outlier'] and r['accepted'] and
            r['estimate_source'].startswith(('ZOOM ', 'CLOSE ', 'MOTION DETECTOR')) for r in rows),
        fps=fps)
    result['scenes']={name:{
        'rms_line_of_sight_error_px':float(np.sqrt(np.mean([r['line_of_sight_error_px']**2 for r in rows if r['scene']==i]))),
        'accepted_frames':sum(r['accepted'] for r in rows if r['scene']==i)} for i,(_,_,name) in enumerate(SCENES)}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('reports/wasp_motion'))
    ap.add_argument('--fps',type=int,choices=(30,60),default=30)
    ap.add_argument('--no-video',action='store_true')
    ap.add_argument('--follower-off',action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    fps=args.fps;rng=np.random.default_rng(7284);background=sky(814)
    params=dict(Wasp().params,straight=.5,blend=.25,rate=120,gain=6,lead=.08,throttle=1)
    before,after=[CameraSimulation(assist,params,not args.follower_off) for assist in (False,True)]
    cw,ch=1328,840;process=None
    video=args.output/('wasp-motion-side-by-side-%dfps.mp4'%fps)
    if not args.no_video:
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg:
            import imageio_ffmpeg
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        process=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
            '-s',f'{cw}x{ch}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast',
            '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    dt=1/300;next_capture=0;preview=None
    for step in range(30*300):
        t=step*dt
        for sim in (before,after):sim.physics(t,dt)
        if t<next_capture-1e-9:continue
        next_capture+=1/fps
        stage,title=scene(t)
        shake=np.zeros(2)
        if 6<=t<11:
            shake=np.array([3.3*math.sin(t*2*math.pi*9)+1.4*math.sin(t*2*math.pi*13),
                            2.2*math.sin(t*2*math.pi*11)])
        noise=rng.normal(0,1.4 if t<25 else 1.7,2)
        outlier=(11<=t<16 and int(round((t-11)*fps))%round(.6*fps)==0)
        drop=21<=t<25 and (t-21)%1.0<.10
        size=5 if t>=25 else 10 if t<21 else 18
        images=[]
        for sim,label in ((before,'OFF / current'),(after,'ON / motion assist')):
            frame,truth,row=sim.capture(t,noise,shake,drop,outlier,size,background,fps)
            if not args.no_video:images.append(panel(frame,truth,sim,label,t))
        if args.no_video:continue
        canvas=np.full((ch,cw,3),(26,19,13),np.uint8)
        text(canvas,'WASP  /  TINY TARGET MOTION ASSIST',(24,31),.7,WHITE,2)
        text(canvas,'ARTIFICIAL SIMULATION - angular tracking only; not a real-flight prediction',(24,55),.43,DIM)
        text(canvas,title,(24,80),.54,YELLOW)
        text(canvas,f'{t:04.1f}s  /  {fps} FPS  /  same disturbances',(947,79),.43,DIM)
        for i,(sim,img) in enumerate(zip((before,after),images)):
            x=16+i*656
            canvas[115:595,x:x+W]=img
            text(canvas,'BEFORE / OFF' if i==0 else 'AFTER / ON',(x+5,106),.54,YELLOW if i==0 else BLUE,2)
            text(canvas,'LOS error %.1f px'%sim.rows[-1]['line_of_sight_error_px'],(x+5,620),.52,WHITE)
            chart(canvas,sim.rows,(x+8,655),t,BLUE,'line_of_sight_error_px',40,'Camera pointing error  /  last 5 seconds  /  0-40 px')
            chart(canvas,sim.rows,(x+8,735),t,YELLOW,'roll_rate_dps',40,'Roll command magnitude  /  last 5 seconds  /  0-40 deg/s')
        text(canvas,'GREEN: real target    YELLOW / RED: detector box / outlier    BLUE: steering estimate    WHITE: camera centre',(24,810),.43,WHITE)
        text(canvas,'Extra response: 120 deg/s, gain 6, blend 0.25 s  |  60 ms frame delay  |  assumed rate response: 80 ms',(24,830),.40,DIM)
        if preview is None and outlier:preview=canvas.copy()
        if process is not None:process.stdin.write(canvas.tobytes())
    if process is not None:
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('Video encoding failed')
        cv2.imwrite(str(args.output/f'preview-{fps}fps.png'),preview)
    summary=dict(description=__doc__,seed=7284,fps=fps,detector='synthetic observations',
        follower=not args.follower_off,parameters=params,baseline=metrics(before,fps),assisted=metrics(after,fps))
    (args.output/f'metrics-{fps}fps.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name,sim in (('off',before),('on',after)):
        with (args.output/f'{name}-{fps}fps.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(sim.rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(sim.rows)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
