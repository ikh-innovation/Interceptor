#!/usr/bin/env python3
"""Five offline 30 FPS stress comparisons with genuine TensorRT inference.

A photographed drone and sky are projected through a translating, rotating
camera. Movement is synthetic. Both production trackers see identical pixels
and the same simulated capture-time attitude. Truth is used only for scoring
and drawing; it never seeds or influences a tracker. No FC is connected.
Only five MP4 files and a Markdown report are produced, without JSON/CSV dumps.
"""

import argparse
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from follower import TargetFollower
from mission import Mission
from tracker import DroneTracker, KalmanTracker, SkyDetector
from wasp_motion import WaspMotionEstimator

W, H, FPS = 640, 480, 30
NATIVE = (1280, 960)
TH, TV = math.tan(math.radians(31.1)), math.tan(math.radians(24.4))
FX, FY = W/(2*TH), H/(2*TV)
BLUE, GREEN, YELLOW, RED = (255, 194, 85), (134, 232, 122), (74, 206, 255), (97, 92, 244)
WHITE, DIM = (241, 239, 229), (164, 151, 128)
# Camera CV axes (right, down, forward) expressed in the guidance body basis.
BASIS = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
SCENARIOS = (
    ('01-zigzag', 'FAST ZIGZAGS / RAPID SIZE CHANGES', 27091),
    ('02-hairpins', 'HAIRPIN REVERSALS / WHIPPING CAMERA', 38214),
    ('03-spirals', 'ACCELERATING SPIRALS / ROLL + VIBRATION', 65128),
    ('04-crossings', 'HIGH-SPEED CROSSINGS / LEAVE + REACQUIRE', 70439),
    ('05-chaos', 'CHAOTIC BURSTS / CAMERA SNAPS + DISTRACTORS', 91572),
)


def text(frame, message, point, size=.5, color=WHITE, thickness=1):
    cv2.putText(frame, message, point, cv2.FONT_HERSHEY_SIMPLEX, size,
                color, thickness, cv2.LINE_AA)


def box(frame, bbox, color, thickness=1):
    if bbox is None:
        return
    x, y, w, h = bbox
    cv2.rectangle(frame, (round(x-w/2), round(y-h/2)),
                  (round(x+w/2), round(y+h/2)), color, thickness)


def matches(candidate, truth):
    if candidate is None or truth is None:
        return False
    x, y, w, h = candidate
    tx, ty, tw, th = truth
    intersection = max(0., min(x+w/2, tx+tw/2)-max(x-w/2, tx-tw/2)) * max(
        0., min(y+h/2, ty+th/2)-max(y-h/2, ty-th/2))
    iou = intersection/max(1e-9, w*h+tw*th-intersection)
    return math.hypot(x-tx, y-ty) <= max(3., tw/2) and iou >= .10


class DetectorAudit:
    """Record genuine neural boxes without modifying model inputs or outputs."""
    def __init__(self, model, tracker):
        self.model, self.tracker = model, tracker
        self.region, self.boxes, self.calls = None, [], 0

    def __call__(self, image, **kwargs):
        result = self.model(image, **kwargs)
        self.calls += 1
        x0, y0 = self.region[:2]
        sx, sy = self.tracker.fw/self.tracker.full_w, self.tracker.fh/self.tracker.full_h
        if result and result[0].boxes:
            for coords, confidence in zip(result[0].boxes.xyxy.cpu().numpy(),
                                          result[0].boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(float, coords)
                assert .25 <= confidence <= 1
                self.boxes.append(((x0+(x1+x2)/2)*sx, (y0+(y1+y2)/2)*sy,
                                   (x2-x1)*sx, (y2-y1)*sy, float(confidence)))
        return result


class Mapper:
    armed = False
    camera_mount_deg = 90

    @staticmethod
    def wasp_pwm(*args):
        raise RuntimeError('This offline benchmark must never produce flight output')


class Evaluation:
    def __init__(self, assist, model):
        self.assist = assist
        self.mission = Mission(Mapper(), wasp_path='/dev/null')
        why = self.mission.set_wasp_enabled(True)
        if why:
            raise RuntimeError('Cannot configure offline WASP fixture: '+why)
        self.mission.wasp.params.update(motion_assist=float(assist), camera_roll=0.)
        self.att = None
        self.rows, self.raw, self.used = [], [], None
        tracker = DroneTracker.__new__(DroneTracker)
        tracker.fw, tracker.fh, tracker.full_w, tracker.full_h = W, H, *NATIVE
        tracker.ref_display = (W, H)
        tracker.MIN_ROI_HALF, tracker.MISS_TO_SEARCH, tracker.MISS_TO_PREDICT = 50, 20, 2
        tracker.IMGSZ_SEARCH = tracker.IMGSZ_TRACK = tracker.IMGSZ_CLOSE = 640
        tracker.target_class, tracker.conf_thresh, tracker.using_engine = 0, .01, True
        tracker.kalman, tracker.motion = KalmanTracker(), WaspMotionEstimator()
        tracker._motion_enabled, tracker._motion_observation, tracker._motion_hint = assist, None, None
        tracker.follower, tracker.use_follower, tracker.follow_hits = TargetFollower(), True, 0
        tracker.use_sky_mask, tracker.sky, tracker.sky_mask = True, SkyDetector(scale=.5), None
        tracker._ground_rejected = False
        tracker.state, tracker.missed, tracker.last_bbox, tracker.pos = 'SEARCH', 0, None, None
        tracker.layer, tracker.conf, tracker.gate_note = 'SEARCH', None, ''
        tracker._frame_idx, tracker._t_last_frame, tracker._frame_dt = 0, None, None
        tracker._ui_fps, tracker.cam_mode, tracker._gate_rejects, tracker.gated_total = FPS, FPS, 0, 0
        tracker.mode_name = self.mission.ai_level_now()
        tracker.ap = SimpleNamespace(mission=self.mission,
            law=SimpleNamespace(_tan_h=TH, _tan_v=TV),
            H_FOV=math.radians(62.2), V_FOV=math.radians(48.8),
            backend=SimpleNamespace(attitude=lambda: self.att),
            wasp_motion_context=lambda capture_t: (self.att, TH, TV))
        self.tracker = tracker
        self.audit = DetectorAudit(model, tracker)
        tracker.model = self.audit
        detector = tracker._detect_region

        def audited_region(frame, region, imgsz, enhance):
            self.audit.region = (max(0, region[0]), max(0, region[1]),
                                 min(tracker.full_w, region[2]), min(tracker.full_h, region[3]))
            return detector(frame, region, imgsz, enhance)

        tracker._detect_region = audited_region

    def evaluate(self, native, display, truth, visible, capture_t, attitude):
        self.att, self.audit.boxes = attitude, []
        self.tracker._frame_idx += 1
        start = time.perf_counter()
        with patch('tracker.perf_counter', return_value=capture_t):
            _, _, _, _, measured = self.tracker._process(native, display, capture_t=capture_t)
        assert self.tracker._motion_enabled == self.assist
        if self.assist:
            assert self.tracker.layer.startswith('MOTION')
        elapsed = (time.perf_counter()-start)*1000
        self.raw = self.audit.boxes
        bbox = self.tracker.last_bbox
        self.used = None if bbox is None else (bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, bbox[2], bbox[3])
        model_hit = visible and any(matches(d[:4], truth) for d in self.raw)
        track_hit = visible and measured and matches(self.used, truth)
        error = None if not visible or self.used is None else math.hypot(
            self.used[0]-truth[0], self.used[1]-truth[1])
        row = dict(t=capture_t-100, visible=visible, model_hit=bool(model_hit),
            track_hit=bool(track_hit), measured=bool(measured),
            error=error, error_plot=200 if error is None else error,
            processing_ms=elapsed, source=self.tracker.layer)
        self.rows.append(row)
        return row


def camera(t, scenario):
    envelope = 1-math.exp(-max(0., t-.2)/.3)
    bank = envelope*(24*math.sin(2*math.pi*.73*t)+6*math.sin(2*math.pi*2.3*t)
                     +1.2*math.sin(2*math.pi*11*t))
    pitch = envelope*(7*math.sin(2*math.pi*.61*t)+4*math.sin(2*math.pi*1.7*t)
                      +.65*math.sin(2*math.pi*13*t))
    yaw = envelope*(10*math.sin(2*math.pi*.49*t)+6*math.sin(2*math.pi*1.23*t)
                    +.75*math.sin(2*math.pi*9*t))
    if scenario in (1, 4):
        yaw += envelope*8*math.tanh(3*math.sin(2*math.pi*.87*t))
    r, p, y = map(math.radians, (bank, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rz = np.array([[cr, -sr, 0.], [sr, cr, 0.], [0., 0., 1.]])
    rx = np.array([[1., 0., 0.], [0., cp, -sp], [0., sp, cp]])
    ry = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
    return ry@rx@rz, (bank, pitch, yaw)


def trajectory(index, duration):
    rng = np.random.default_rng(SCENARIOS[index][2])
    phase = rng.uniform(-math.pi, math.pi, 3)
    oscillators = phase.copy()
    frequencies = np.array([.9, 1.2, 2.9])
    next_frequency_change = 0.
    position, velocity, next_turn = np.zeros(2), np.zeros(2), 0.
    samples = []
    triangle = lambda angle: 2/math.pi*math.asin(math.sin(angle))
    previous = None
    for i in range(round(duration*FPS)):
        t = i/FPS
        if t >= next_frequency_change:
            frequencies = np.array([rng.uniform(.55, 1.4), rng.uniform(.75, 1.7), rng.uniform(2.2, 3.8)])
            next_frequency_change = t+rng.uniform(.18, .70)
        oscillators += 2*math.pi*frequencies/FPS
        envelope = 1-math.exp(-max(0., t-.35)/.16)
        if index == 0:
            x = 155*triangle(oscillators[0])+30*math.sin(oscillators[2])
            y = 80*triangle(oscillators[1])+25*math.sin(1.4*oscillators[2])
        elif index == 1:
            x = 180*math.tanh(2.5*math.sin(oscillators[0]+.3*math.sin(3*t)))
            y = 90*math.sin(oscillators[1])+22*math.sin(oscillators[2])
        elif index == 2:
            theta = oscillators[0]+2*math.pi*.025*t*t
            x = 155*math.sin(theta)+35*math.sin(oscillators[2])
            y = 100*math.cos(theta+.4*math.sin(oscillators[1]))+22*math.sin(1.3*oscillators[2])
        elif index == 3:
            x = 310*math.sin(oscillators[0]+.4*math.sin(2.3*t))
            y = 140*math.sin(oscillators[1])
        else:
            if t >= next_turn:
                angle, speed = rng.uniform(-math.pi, math.pi), rng.uniform(350, 1250)
                velocity = speed*np.array([math.cos(angle), math.sin(angle)])
                next_turn = t+rng.uniform(.10, .32)
            position += velocity/FPS
            for axis, bound in enumerate((230, 140)):
                if abs(position[axis]) > bound:
                    position[axis] = math.copysign(2*bound-abs(position[axis]), position[axis])
                    velocity[axis] *= -1
            x, y = position
        x, y = envelope*x, envelope*y
        zoom = .5+.39*math.sin(2*math.pi*.61*t+phase[2])+.10*math.sin(2*math.pi*1.37*t)
        width = math.exp(math.log(3.2)+(math.log(46)-math.log(3.2))*zoom)
        width = (1-envelope)*20+envelope*width
        distance = FX*.6/width  # Assumed 0.6m target; changing range, fixed lens.
        rotation, angles = camera(t, index)
        body = BASIS@rotation@BASIS.T
        omega = np.zeros(3) if previous is None else cv2.Rodrigues(previous['body'].T@body)[0].ravel()*FPS
        attitude = dict(rotation=body, rollspeed=float(omega[0]),
                        pitchspeed=float(omega[1]), yawspeed=float(omega[2]))
        camera_position = np.array([0., 0., 25*t])
        relative = distance*np.array([x/FX, y/FY, 1.])
        world_target = camera_position+relative
        sample = dict(t=t, x=x, y=y, desired_width=width, distance=distance,
            rotation=rotation, body=body, attitude=attitude, angles=angles,
            camera_position=camera_position, relative=relative, world_target=world_target,
            camera_rate=math.degrees(float(np.linalg.norm(omega))),
            occluded=index == 4 and t>3 and t%2.3<.10,
            distractor=index == 4 and t>4 and .2<t%3.7<.75)
        samples.append(sample)
        previous = sample
    return samples


def project_patch(sample, sprite, intrinsic, relative=None, width=.6):
    centre = sample['relative'] if relative is None else relative
    height = width*sprite.shape[0]/sprite.shape[1]
    corners = centre+np.array([[-width/2, -height/2, 0.], [width/2, -height/2, 0.],
                              [width/2, height/2, 0.], [-width/2, height/2, 0.]])
    camera_points = corners@sample['rotation']
    if min(camera_points[:, 2]) <= .1:
        return None
    pixels = camera_points@intrinsic.T
    return (pixels[:, :2]/pixels[:, 2:3]).astype(np.float32)


def paste(frame, sprite, corners):
    if corners is None:
        return
    nw, nh = NATIVE
    x1, y1 = np.floor(corners.min(axis=0)).astype(int)-2
    x2, y2 = np.ceil(corners.max(axis=0)).astype(int)+2
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(nw, x2), min(nh, y2)
    if x2 <= x1 or y2 <= y1:
        return
    sh, sw = sprite.shape[:2]
    source = np.float32([[0, 0], [sw-1, 0], [sw-1, sh-1], [0, sh-1]])
    transform = cv2.getPerspectiveTransform(source, corners-np.float32([x1, y1]))
    alpha = sprite[..., 3:4].astype(np.float32)/255
    premultiplied = np.concatenate((sprite[..., :3]*alpha, alpha), axis=2)
    warped = cv2.warpPerspective(premultiplied, transform, (x2-x1, y2-y1))
    frame[y1:y2, x1:x2] = warped[..., :3]+frame[y1:y2, x1:x2]*(1-warped[..., 3:4])


def make_image(background, sprite, sample, previous):
    nw, nh = NATIVE
    intrinsic = np.array([[2*FX, 0., nw/2], [0., 2*FY, nh/2], [0., 0., 1.]])
    source_k = intrinsic.copy()
    source_k[:2, 2] = (background.shape[1]/2, background.shape[0]/2)
    # A 900m cloud plane has translation parallax; the lens follows the body
    # rotation directly. This is a kinematic stress model, without a gimbal.
    translation = np.diag([1., 1., 1-sample['camera_position'][2]/900])
    transform = intrinsic@sample['rotation'].T@translation@np.linalg.inv(source_k)
    frame = cv2.warpPerspective(background, transform, NATIVE, borderMode=cv2.BORDER_REFLECT_101)
    corners = project_patch(sample, sprite, intrinsic)
    if not sample['occluded']:
        paste(frame, sprite, corners)
    if sample['distractor']:
        distractor = sample['distance']*np.array([-.18, .08, 1.])
        paste(frame, sprite, project_patch(sample, sprite, intrinsic, distractor, .9))
    # Approximate exposure blur, based on actual inter-frame camera movement.
    # Target blur adds its relative screen displacement rather than fake boxes.
    blur = 0
    if previous is not None and corners is not None:
        centre = corners.mean(axis=0)
        old = project_patch(previous, sprite, intrinsic)
        delta = np.zeros(2) if old is None else centre-old.mean(axis=0)
        exposure = .003 if sample['t']%2.4<1.8 else .006
        blur = min(21, max(1, round(np.linalg.norm(delta)*FPS*exposure)))
        if blur > 1:
            k = blur|1
            kernel = np.zeros((k, k), np.float32)
            direction = delta/max(1e-9, np.linalg.norm(delta))
            a, b = np.round((k-1)/2-direction*(k-1)/2).astype(int), np.round((k-1)/2+direction*(k-1)/2).astype(int)
            cv2.line(kernel, tuple(a), tuple(b), 1., 1)
            kernel /= kernel.sum()
            frame = cv2.filter2D(frame, -1, kernel)
    if corners is None:
        truth, visible = None, False
    else:
        low, high = corners.min(axis=0)/2, corners.max(axis=0)/2
        truth = (*(low+high)/2, *(high-low))
        visible = bool(np.all(low >= 0) and np.all(high < (W, H)) and not sample['occluded'])
    return frame, cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA), truth, visible, blur


def chart(canvas, rows, offset, key, maximum, title, color):
    x, y = offset
    text(canvas, title, (x, y-7), .39, DIM)
    cv2.rectangle(canvas, (x, y), (x+600, y+42), (63, 47, 35), 1)
    t = rows[-1]['t']
    recent = [r for r in rows if t-5 <= r['t'] <= t]
    points = [(x+round((r['t']-max(0, t-5))/5*600),
               y+42-round(min(maximum, max(0., float(r[key])))/maximum*42)) for r in recent]
    if len(points)>1:
        cv2.polylines(canvas, [np.array(points)], False, color, 1, cv2.LINE_AA)


def metrics(evaluation):
    rows = evaluation.rows
    eligible = [r for r in rows if r['visible']]
    errors = [r['error'] for r in eligible if r['measured'] and r['error'] is not None]
    return dict(frames=len(rows), visible=len(eligible),
        model_hits=sum(r['model_hit'] for r in rows), tracking_hits=sum(r['track_hit'] for r in rows),
        model_pct=100*sum(r['model_hit'] for r in rows)/max(1, len(eligible)),
        tracking_pct=100*sum(r['track_hit'] for r in rows)/max(1, len(eligible)),
        error_rms=None if not errors else float(np.sqrt(np.mean(np.square(errors)))),
        processing_ms=float(np.mean([r['processing_ms'] for r in rows])),
        flow_frames=sum('FLOW' in r['source'] for r in rows), calls=evaluation.audit.calls)


def panel(frame, truth, visible, evaluation):
    f = frame.copy()
    cv2.drawMarker(f, (W//2, H//2), WHITE, cv2.MARKER_CROSS, 20, 1)
    box(f, truth, GREEN)
    for detection in evaluation.raw:
        box(f, detection[:4], YELLOW if matches(detection[:4], truth) else RED)
    box(f, evaluation.used, BLUE, 2)
    if truth is not None and 0<=truth[0]<W and 0<=truth[1]<H:
        x, y = max(0, min(W-80, round(truth[0])-40)), max(0, min(H-60, round(truth[1])-30))
        inset = cv2.resize(f[y:y+60, x:x+80], (240, 180), interpolation=cv2.INTER_NEAREST)
        f[12:192, W-252:W-12] = inset
        cv2.rectangle(f, (W-252, 12), (W-12, 192), BLUE, 1)
        text(f, '3x TARGET DETAIL', (W-242, 211), .4, (40, 45, 50))
    else:
        text(f, 'TARGET OUTSIDE VIEW', (W-242, 37), .43, RED)
    cv2.rectangle(f, (10, H-59), (W-10, H-10), (31, 24, 16), -1)
    text(f, evaluation.tracker.layer, (20, H-37), .44, BLUE)
    row = evaluation.rows[-1]
    status = 'MODEL HIT' if row['model_hit'] else 'MODEL MISS' if visible else 'TARGET NOT FULLY VISIBLE'
    text(f, status+' | '+('TRACKED' if row['track_hit'] else 'NO VERIFIED TRACK'), (20, H-19), .38)
    return f


def canvas_frame(frame, truth, visible, sample, evaluations, title, speed, blur):
    canvas = np.full((900, 1328, 3), (26, 19, 13), np.uint8)
    text(canvas, 'WASP / HARD MOTION / REAL TENSORRT / 30 FPS', (24, 31), .69, WHITE, 2)
    text(canvas, 'Synthetic 3D motion / genuine best.engine detections / identical frames + attitude / no gimbal',
         (24, 55), .43, DIM)
    text(canvas, title, (24, 81), .51, BLUE)
    size = '--' if truth is None else '%.1f'%truth[2]
    text(canvas, f'{sample["t"]:04.1f}s | {speed:5.0f} px/s | {size} px wide', (947, 81), .4, DIM)
    for col, evaluation in enumerate(evaluations):
        x = 16+col*656
        canvas[115:595, x:x+W] = panel(frame, truth, visible, evaluation)
        text(canvas, 'MOTION ASSIST OFF / CURRENT' if col == 0 else 'MOTION ASSIST ON',
             (x+5, 106), .51, YELLOW if col == 0 else BLUE, 2)
        results = metrics(evaluation)
        text(canvas, f'Fresh model: {results["model_pct"]:5.1f}% | Tracked: {results["tracking_pct"]:5.1f}%',
             (x+5, 620), .49)
        error = evaluation.rows[-1]['error']
        text(canvas, 'BBox centre error: '+('--' if error is None else f'{error:.1f} px'), (x+5, 645), .43, DIM)
        chart(canvas, evaluation.rows, (x+8, 676), 'error_plot', 200,
              'BBox centre error / last 5s / 0-200 px; missing estimate at top', BLUE)
        chart(canvas, evaluation.rows, (x+8, 757), 'track_hit', 1,
              'Correct measured tracking / last 5s / top = tracked, bottom = missed', GREEN)
    bank, pitch, yaw = sample['angles']
    text(canvas, f'CAMERA: 25 m/s | bank {bank:+.0f} / pitch {pitch:+.0f} / yaw {yaw:+.0f} deg | '
         f'rotation {sample["camera_rate"]:.0f} deg/s | image blur {blur} px', (24, 829), .43, YELLOW)
    text(canvas, 'GREEN: placed target | YELLOW: matching neural box | RED: other neural box | BLUE: tracking estimate',
         (24, 855), .43)
    text(canvas, 'Coverage counts fully visible target frames. Holds are excluded. No fabricated boxes or live flight commands.',
         (24, 877), .4, DIM)
    text(canvas, '25 m/s and body motion are assumed kinematics; this is a stress test, not calibrated flight prediction.',
         (24, 895), .38, DIM)
    return canvas


def run(model, args, sprite, sky, index):
    name, title, seed = SCENARIOS[index]
    samples = trajectory(index, args.duration)
    background = cv2.resize(sky, (NATIVE[0]*2, NATIVE[1]*2), interpolation=cv2.INTER_LINEAR)
    evaluations = [Evaluation(on, model) for on in (False, True)]
    output = args.output/(name+'-30fps.mp4')
    ffmpeg = args.ffmpeg or shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.Popen([ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', '1328x900', '-r', str(FPS), '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '21', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output)], stdin=subprocess.PIPE)
    previous, last_truth, speeds, widths, camera_rates = None, None, [], [], []
    try:
        for i, sample in enumerate(samples):
            native, frame, truth, visible, blur = make_image(background, sprite, sample, previous)
            speed = 0 if truth is None or last_truth is None else math.hypot(
                truth[0]-last_truth[0], truth[1]-last_truth[1])*FPS
            speeds.append(speed)
            if truth is not None:
                widths.append(truth[2])
            camera_rates.append(sample['camera_rate'])
            for evaluation in evaluations:
                evaluation.evaluate(native, frame, truth, visible, 100+sample['t'], sample['attitude'])
            canvas = canvas_frame(frame, truth, visible, sample, evaluations, title, speed, blur)
            process.stdin.write(canvas.tobytes())
            if i == min(len(samples)-1, FPS*2):
                cv2.imwrite('/tmp/wasp-stress-'+name+'-preview.png', canvas)
            if i%(FPS*2) == 0:
                print(name, f'{sample["t"]:.1f}/{args.duration:.1f}s',
                      [(r['model_hits'], r['tracking_hits'], r['visible']) for r in map(metrics, evaluations)], flush=True)
            previous, last_truth = sample, truth
        results = [metrics(e) for e in evaluations]
        for col, result in enumerate(results):
            x = 16+col*656
            cv2.rectangle(canvas, (x, 603), (x+640, 805), (26, 19, 13), -1)
            text(canvas, 'FINAL RESULTS / VISIBLE TARGET FRAMES', (x+8, 631), .51, WHITE, 2)
            text(canvas, f'TensorRT target hits: {result["model_hits"]}/{result["visible"]} = {result["model_pct"]:.1f}%',
                 (x+8, 667), .52)
            text(canvas, f'Correct tracking: {result["tracking_hits"]}/{result["visible"]} = {result["tracking_pct"]:.1f}%',
                 (x+8, 700), .52, GREEN)
            rms = '--' if result['error_rms'] is None else '%.2f px'%result['error_rms']
            text(canvas, 'Measured bbox centre RMS error: '+rms, (x+8, 733), .48, BLUE)
            text(canvas, f'Target visible: {result["visible"]}/{result["frames"]} frames | Neural calls: {result["calls"]}',
                 (x+8, 765), .42, DIM)
        for _ in range(FPS*2):
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('Video encoding failed')
    summary = dict(name=name, title=title, seed=seed, video=output.name,
        speed_p95=float(np.percentile(speeds, 95)), speed_max=max(speeds),
        width_min=min(widths), width_max=max(widths), camera_rate_max=max(camera_rates),
        baseline=results[0], assisted=results[1])
    print('READY', str(output.resolve()), summary, flush=True)
    return summary


def write_report(args, model, results):
    digest = hashlib.sha256(args.engine.read_bytes()).hexdigest()
    lines = ['# WASP: five hard 30 FPS TensorRT comparisons', '',
        'Motion assist OFF is on the left; ON is on the right. These are offline vision stress tests.', '',
        'The target and sky are photographs from frame 2430 of '
        '`/home/iknowhow/Documents/Interceptor/data/ClipExport.mp4`. The original target '
        'box was `[455, 348, 508, 378]`, with TensorRT confidence 0.875. '
        'Movement, size changes and camera motion are synthetic. Neural detections are genuine.', '',
        'The suite uses one photographed target pose. It is a stress comparison of the current '
        'pipelines, not a model-accuracy benchmark on unseen flight footage.', '',
        f'Engine: `{args.engine}`; SHA-256 `{digest}`. '
        f'Backend: `{type(model.predictor.model.backend).__name__}` on `{model.predictor.model.device}`. '
        f'Model confidence threshold: {model.predictor.args.conf}.', '',
        'Every video uses 30 FPS, native 1280×960 pixels and 640×480 tracking coordinates. '
        f'Each contains {args.duration:g} seconds of motion plus a 2-second results card. '
        'Playback FPS does not establish real-time inference throughput.', '',
        'Both production trackers receive identical frames and capture-time camera attitude. '
        'OFF retains the existing Kalman/follower path; ON uses the current motion-assist estimator. '
        'Existing sky gating and detector settings remain enabled. No tracker tuning or model '
        'weight changes were made for these tests. Ground truth never seeds a detector or tracker.', '',
        'The camera follows the same prescribed path in both modes, independently of tracking '
        'outputs. These videos isolate vision behavior rather than simulate closed-loop steering.', '',
        'The camera translates forward at an assumed 25 m/s and rotates with the simulated body, '
        'without a gimbal. A cloud plane at 900m supplies translation parallax; bank, pitch, yaw '
        'and 9–13Hz vibration supply rigid-camera motion. The photographed target is projected '
        'as a 0.6m planar object whose range changes repeatedly, producing apparent zoom without '
        'changing the fixed lens calibration. Exposure blur is an approximate directional image '
        'filter, capped at 21 native pixels. This kinematic model is not calibrated against the '
        'physical drone, and camera angles/accelerations are deliberately aggressive.', '',
        'Coverage denominators include only frames where the entire target is in view and '
        'not deliberately occluded. Each video also shows visible-frame counts, so leaving '
        'the camera view cannot masquerade as a detector failure or improvement. '
        'A correct box requires centre distance <= max(3px, half target width) and bbox IoU >= 0.10. '
        'Measured tracking can come from a detector, follower or validated flow. Held/predicted '
        'boxes are excluded. The centre-error RMS includes all measured estimates while the '
        'target is visible, including wrong associations; it does not penalize missing estimates.', '',
        '| Scenario | Fresh model OFF / ON | Correct tracking OFF / ON | Centre RMS OFF / ON | Target visible |',
        '| --- | ---: | ---: | ---: | ---: |']
    for s in results:
        a, b = s['baseline'], s['assisted']
        rms = lambda r: '--' if r['error_rms'] is None else '%.2f'%r['error_rms']
        lines.append(f'| [{s["name"]}]({s["video"]}) | {a["model_pct"]:.1f}% / {b["model_pct"]:.1f}% | '
            f'{a["tracking_pct"]:.1f}% / {b["tracking_pct"]:.1f}% | {rms(a)} / {rms(b)} px | '
            f'{a["visible"]}/{a["frames"]} |')
    lines += ['', '| Scenario | Apparent speed p95 / max | Apparent width min–max | Max body rotation | Seed |',
              '| --- | ---: | ---: | ---: | ---: |']
    for s in results:
        lines.append(f'| {s["name"]} | {s["speed_p95"]:.0f} / {s["speed_max"]:.0f} px/s | '
            f'{s["width_min"]:.1f}–{s["width_max"]:.1f} px | {s["camera_rate_max"]:.0f} deg/s | {s["seed"]} |')
    lines += ['', 'The earlier videos, previews and JSON/CSV files were removed. Only these five '
        'MP4 comparisons, this report and the two real photographic assets remain.', '',
        'Reproduce from the repository root:', '', '```bash',
        '.env/yolo/bin/python src/tests/render_wasp_motion_stress.py', '```', '',
        'Use `--ffmpeg /path/to/ffmpeg` if ffmpeg is not on PATH. The script accepts '
        '`--scenario 1` through `--scenario 5` for a single scenario and requires the genuine '
        'TensorRT backend. It opens no FC connection and sends no flight commands.', '']
    if len(results) == 5 and all(s['baseline']['tracking_pct'] > s['assisted']['tracking_pct'] for s in results):
        lines += ['', 'Verdict: OFF has higher correct-tracking coverage in all five scenarios. '
            'The current ON estimator accepts far fewer correct tracks. Motion assist remains '
            'OFF by default; these results do not support treating ON as a flight improvement.', '']
    (args.output/'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, default=Path('runs/detect/v0/weights/best.engine'))
    parser.add_argument('--output', type=Path, default=Path('reports/wasp_motion'))
    parser.add_argument('--assets', type=Path, default=Path('reports/wasp_motion/assets'))
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--ffmpeg')
    parser.add_argument('--scenario', type=int, choices=range(1, 6))
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 0 < args.duration <= 30:
        parser.error('--duration must be positive and at most 30 seconds')
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.engine), task='detect')
    model(np.zeros((960, 1280, 3), np.uint8), imgsz=640, conf=.25, verbose=False)
    if type(model.predictor.model.backend).__name__ != 'TensorRTBackend':
        raise RuntimeError('Genuine TensorRT inference is required')
    sprite = cv2.imread(str(args.assets/'real-mosquito-target.png'), cv2.IMREAD_UNCHANGED)
    sky = cv2.imread(str(args.assets/'real-sky.jpg'))
    if sprite is None or sprite.shape[2] != 4 or sky is None:
        raise RuntimeError('Missing real target/sky photographic assets')
    results = []
    for index in ([args.scenario-1] if args.scenario else range(5)):
        results.append(run(model, args, sprite, sky, index))
        write_report(args, model, results)


if __name__ == '__main__':
    main()
