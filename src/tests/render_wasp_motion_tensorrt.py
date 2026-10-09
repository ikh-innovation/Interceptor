#!/usr/bin/env python3
"""Offline, reproducible motion-assist comparison using genuine TensorRT inference.

The photographed drone and sky are composited into a synthetic random trajectory.
Both production trackers receive identical camera frames. Ground truth is used
only for drawing/scoring, never for detector crops, tracking or guidance.
No flight-controller connection is opened and no flight commands are sent.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import patch

import cv2
import numpy as np
from ultralytics import YOLO

from render_wasp_motion import (CameraSimulation, W, H, TH, TV, BLUE, GREEN,
                                YELLOW, RED, WHITE, DIM, text, box, chart)
from tracker import DroneTracker, SkyDetector
from wasp import Wasp, WaspGuidance


def matches(candidate, truth):
    """Small-object hit: centre within half target width and IoU >= 0.10."""
    if candidate is None:
        return False
    x, y, w, h = candidate
    tx, ty, tw, th = truth
    intersection = max(0, min(x+w/2, tx+tw/2)-max(x-w/2, tx-tw/2)) * max(
        0, min(y+h/2, ty+th/2)-max(y-h/2, ty-th/2))
    iou = intersection / max(1e-9, w*h+tw*th-intersection)
    return math.hypot(x-tx, y-ty) <= max(3, tw/2) and iou >= .10


class DetectorAudit:
    """Forward unmodified model results; record their native frame coordinates."""

    def __init__(self, model, tracker):
        self.model, self.tracker = model, tracker
        self.region = None
        self.boxes = []
        self.calls = 0

    def __call__(self, image, **kwargs):
        result = self.model(image, **kwargs)
        self.calls += 1
        x0, y0 = self.region[:2]
        sx = self.tracker.fw / self.tracker.full_w
        sy = self.tracker.fh / self.tracker.full_h
        if result and result[0].boxes:
            for coordinates, confidence in zip(result[0].boxes.xyxy.cpu().numpy(),
                                                result[0].boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(float, coordinates)
                self.boxes.append(dict(box=[(x0+(x1+x2)/2)*sx,
                    (y0+(y1+y2)/2)*sy, (x2-x1)*sx, (y2-y1)*sy],
                    confidence=float(confidence), region=list(self.region)))
        return result


class Evaluation(CameraSimulation):
    def __init__(self, assist, model, fps, native):
        params = dict(Wasp().params, blend=.25, rate=120, gain=6, lead=.08)
        super().__init__(assist, params, follow=True)
        tracker = self.tracker
        tracker.full_w, tracker.full_h = native
        tracker.IMGSZ_SEARCH = tracker.IMGSZ_TRACK = tracker.IMGSZ_CLOSE = 640
        tracker.target_class, tracker.conf_thresh = 0, .01
        tracker.using_engine = True
        tracker.use_sky_mask, tracker.sky = True, SkyDetector(scale=.5)
        tracker._ui_fps = tracker.cam_mode = fps
        self.audit = DetectorAudit(model, tracker)
        tracker.model = self.audit
        self.raw_boxes, self.events = [], []
        detector = tracker._detect_region

        def audited_region(frame, region, imgsz, enhance):
            self.audit.region = (max(0, region[0]), max(0, region[1]),
                                 min(tracker.full_w, region[2]), min(tracker.full_h, region[3]))
            return detector(frame, region, imgsz, enhance)

        tracker._detect_region = audited_region

    def evaluate(self, native, frame, truth, t):
        self.capture_att = {'rotation': np.eye(3)}
        self.audit.boxes = []
        self.tracker._frame_idx += 1
        start = time.perf_counter()
        # Only the clock is simulated. The detector and trackers are production code.
        with patch('tracker.perf_counter', return_value=100+t):
            fused, _, _, _, measured = self.tracker._process(native, frame, capture_t=100+t)
        elapsed = (time.perf_counter()-start)*1000
        self.raw_boxes = self.audit.boxes
        self.used = None if self.tracker.last_bbox is None else (
            self.tracker.last_bbox[0]+self.tracker.last_bbox[2]/2,
            self.tracker.last_bbox[1]+self.tracker.last_bbox[3]/2,
            self.tracker.last_bbox[2], self.tracker.last_bbox[3])
        self.source = self.tracker.layer
        hit = any(matches(d['box'], truth) for d in self.raw_boxes)
        tracking_hit = matches(self.used, truth) and measured
        held_hit = matches(self.used, truth) and not measured
        false_boxes = sum(not matches(d['box'], truth) for d in self.raw_boxes)
        cmd = None
        if fused is not None and measured:
            if self.assist:
                cmd = self.tracker._motion_observation
            else:
                cmd = self.g.observe(fused[0]-W/2, fused[1]-H/2, W, H,
                                     TH, TV, self.capture_att, 100+t)
                cmd['camera_roll'] = 0
        self.command = np.zeros(3) if cmd is None else np.array(WaspGuidance.rates(
            cmd, self.capture_att, 100+t, gain=6, lead_s=.08, max_rate_dps=120))
        error = None if self.used is None else math.hypot(
            self.used[0]-truth[0], self.used[1]-truth[1])
        row = dict(t_s=round(t, 5), motion_assist=self.assist,
            fresh_detector_hit=int(hit), measured_tracking_hit=int(tracking_hit),
            held_box_hit=int(held_hit), false_model_boxes=false_boxes,
            model_boxes=len(self.raw_boxes), estimate_source=self.source,
            estimate_error_px=error, measured=int(measured),
            roll_rate_dps=math.degrees(self.command[0]),
            pitch_rate_dps=math.degrees(self.command[1]), processing_ms=elapsed,
            target_width_px=truth[2], target_height_px=truth[3])
        self.rows.append(row)
        self.events.append(dict(t_s=round(t, 5), truth=list(truth),
            detections=self.raw_boxes, estimate=None if self.used is None else list(self.used),
            **{k: v for k, v in row.items() if k != 't_s'}))
        return row


def motion(seed, fps, duration, bursts):
    rng = np.random.default_rng(seed)
    position, velocity, wanted = np.array([W/2+40., H/2-24.]), np.zeros(2), np.zeros(2)
    next_turn = 1.0
    rows = []
    for i in range(round(duration*fps)):
        t = i/fps
        if t >= next_turn:
            heading = rng.uniform(-math.pi, math.pi)
            speed = rng.uniform(20, 155) if bursts else rng.uniform(8, 90)
            wanted = speed*np.array([math.cos(heading), math.sin(heading)])
            next_turn = t+rng.uniform(.25, .9) if bursts else t+rng.uniform(.45, 1.5)
        velocity += (wanted-velocity)/(1+.12*fps)
        position += velocity/fps
        for axis, limit in enumerate((W, H)):
            if position[axis] < 80 or position[axis] > limit-80:
                position[axis] = np.clip(position[axis], 80, limit-80)
                wanted[axis], velocity[axis] = -wanted[axis], -velocity[axis]
        # A known visible target becomes increasingly tiny, then grows again.
        sizes = (20, 12, 8, 5, 16) if bursts else (20, 16, 12, 8, 18)
        size = sizes[min(4, int(t/(duration/5)))]
        amplitude = 2.6 if bursts else 1.4
        shake = amplitude*np.array([math.sin(2*math.pi*9*t)+.3*math.sin(2*math.pi*13*t),
                                   .7*math.sin(2*math.pi*11*t)])
        blur = bursts and (t % 3.2 > 2.9)
        # Brief genuine image occlusion: no fake detector gap is injected.
        occluded = 7 < t < 15 and t % 2.7 < .09
        distractor = 5 < t < 18 and .25 < t % 4.1 < .60
        rows.append(dict(t=t, x=float(position[0]+shake[0]), y=float(position[1]+shake[1]),
            speed=float(np.linalg.norm(velocity)), size=size, shake=shake,
            blur=blur, occluded=occluded, distractor=distractor))
    return rows


def composite(frame, sprite, x, y, width, opacity=1.):
    h = max(2, round(width*sprite.shape[0]/sprite.shape[1]))
    resized = cv2.resize(sprite, (max(2, round(width)), h), interpolation=cv2.INTER_AREA)
    sw = resized.shape[1]
    x0, y0 = round(x-sw/2), round(y-h/2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(frame.shape[1], x0+sw), min(frame.shape[0], y0+h)
    if xb <= xa or yb <= ya:
        return
    source = resized[ya-y0:yb-y0, xa-x0:xb-x0]
    alpha = source[..., 3:4].astype(np.float32)/255*opacity
    frame[ya:yb, xa:xb] = source[..., :3]*alpha+frame[ya:yb, xa:xb]*(1-alpha)


def image(sky, sprite, movement, native):
    nw, nh = native
    scale = nw/W
    # The same vibration displaces background and target in both panels.
    transform = np.float32([[1, 0, movement['shake'][0]*scale],
                            [0, 1, movement['shake'][1]*scale]])
    frame = cv2.warpAffine(sky, transform, (nw, nh), borderMode=cv2.BORDER_REFLECT_101)
    if not movement['occluded']:
        composite(frame, sprite, movement['x']*scale, movement['y']*scale,
                  movement['size']*scale)
    if movement['distractor']:
        composite(frame, sprite, (W-movement['x'])*scale,
                  (H-movement['y'])*scale, 23*scale)
    if movement['blur']:
        kernel = np.zeros((5, 5), np.float32)
        kernel[2, :] = .2
        frame = cv2.filter2D(frame, -1, kernel)
    truth = (movement['x'], movement['y'], movement['size'],
             movement['size']*sprite.shape[0]/sprite.shape[1])
    return frame, cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA), truth


def render_panel(frame, truth, evaluation, movement):
    f = frame.copy()
    cv2.drawMarker(f, (W//2, H//2), WHITE, cv2.MARKER_CROSS, 20, 1)
    box(f, truth, GREEN)
    for detection in evaluation.raw_boxes:
        box(f, detection['box'], YELLOW if matches(detection['box'], truth) else RED)
    box(f, evaluation.used, BLUE, 2)
    x = max(0, min(W-80, round(truth[0])-40))
    y = max(0, min(H-60, round(truth[1])-30))
    inset = cv2.resize(f[y:y+60, x:x+80], (240, 180), interpolation=cv2.INTER_NEAREST)
    f[12:192, W-252:W-12] = inset
    cv2.rectangle(f, (W-252, 12), (W-12, 192), BLUE, 1)
    text(f, '3x TARGET DETAIL', (W-242, 211), .4, (40, 45, 50))
    cv2.rectangle(f, (10, H-59), (W-10, H-10), (31, 24, 16), -1)
    text(f, evaluation.source, (20, H-37), .44, BLUE)
    row = evaluation.rows[-1]
    status = 'MODEL HIT' if row['fresh_detector_hit'] else 'MODEL MISS'
    text(f, status+'  |  '+('TRACKED' if row['measured_tracking_hit'] else 'NO VERIFIED TRACK'),
         (20, H-19), .41, WHITE)
    return f


def metrics(evaluation):
    rows = evaluation.rows
    errors = [r['estimate_error_px'] for r in rows if r['measured'] and r['estimate_error_px'] is not None]
    n = len(rows)
    return dict(frames=n, detector_calls=evaluation.audit.calls,
        fresh_detector_hits=sum(r['fresh_detector_hit'] for r in rows),
        fresh_detector_coverage_pct=100*sum(r['fresh_detector_hit'] for r in rows)/n,
        measured_tracking_hits=sum(r['measured_tracking_hit'] for r in rows),
        measured_tracking_coverage_pct=100*sum(r['measured_tracking_hit'] for r in rows)/n,
        held_box_hits=sum(r['held_box_hit'] for r in rows),
        false_model_boxes=sum(r['false_model_boxes'] for r in rows),
        rms_estimate_error_px=None if not errors else float(np.sqrt(np.mean(np.square(errors)))),
        mean_processing_ms=float(np.mean([r['processing_ms'] for r in rows])),
        flow_frames=sum('FLOW' in r['estimate_source'] for r in rows),
        by_target_width={str(w):dict(frames=sum(r['target_width_px']==w for r in rows),
            fresh_detector_hits=sum(r['target_width_px']==w and r['fresh_detector_hit'] for r in rows),
            measured_tracking_hits=sum(r['target_width_px']==w and r['measured_tracking_hit'] for r in rows))
            for w in sorted({r['target_width_px'] for r in rows})})


def run(model, args, sprite, sky_source, name, fps, seed, bursts):
    # Hold camera geometry fixed to isolate sampling and random motion. The
    # 60fps clip evaluates 60Hz sampling, not the sensor's separate 60fps crop.
    native = (1280, 960)
    background = cv2.resize(sky_source, native, interpolation=cv2.INTER_LINEAR)
    evaluations = [Evaluation(on, model, fps, native) for on in (False, True)]
    output = args.output / (name+'.mp4')
    ffmpeg = args.ffmpeg or shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.Popen([ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', '1328x880', '-r', str(fps), '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output)], stdin=subprocess.PIPE)
    try:
        for i, movement in enumerate(motion(seed, fps, args.duration, bursts)):
            t = movement['t']
            full, frame, truth = image(background, sprite, movement, native)
            panels = []
            for evaluation in evaluations:
                row = evaluation.evaluate(full, frame, truth, t)
                details = dict(target_visible=int(not movement['occluded']),
                    motion_speed_px_s=movement['speed'], image_blur=int(movement['blur']),
                    distractor_visible=int(movement['distractor']))
                row.update(details)
                evaluation.events[-1].update(details)
                row['estimate_error_plot_px'] = row['estimate_error_px'] if row['estimate_error_px'] is not None else 40.
                panels.append(render_panel(frame, truth, evaluation, movement))
            canvas = np.full((880, 1328, 3), (26, 19, 13), np.uint8)
            text(canvas, 'WASP  /  REAL TENSORRT  /  RANDOM TARGET MOTION', (24, 31), .69, WHITE, 2)
            text(canvas, 'Photographed target + sky / artificial motion / genuine best.engine inference / identical frames',
                 (24, 55), .43, DIM)
            text(canvas, 'RANDOM BURSTS + TINY TARGET' if bursts else 'RANDOM DIRECTIONS + VARIABLE SPEED',
                 (24, 81), .51, BLUE)
            text(canvas, f'{t:04.1f}s  |  {fps} FPS  |  {movement["speed"]:5.1f} px/s  |  {movement["size"]} px wide',
                 (860, 81), .41, DIM)
            for col, (evaluation, panel) in enumerate(zip(evaluations, panels)):
                x = 16+col*656
                canvas[115:595, x:x+W] = panel
                text(canvas, 'MOTION ASSIST OFF / CURRENT' if col == 0 else 'MOTION ASSIST ON',
                     (x+5, 106), .51, YELLOW if col == 0 else BLUE, 2)
                n = len(evaluation.rows)
                detector_pct = 100*sum(r['fresh_detector_hit'] for r in evaluation.rows)/n
                tracking_pct = 100*sum(r['measured_tracking_hit'] for r in evaluation.rows)/n
                text(canvas, f'Fresh model hits: {detector_pct:5.1f}%   |   Tracked: {tracking_pct:5.1f}%',
                     (x+5, 620), .48, WHITE)
                error = evaluation.rows[-1]['estimate_error_px']
                text(canvas, 'BBox centre error: '+('--' if error is None else f'{error:.2f} px'),
                     (x+5, 645), .43, DIM)
                chart(canvas, evaluation.rows, (x+8, 675), t, BLUE, 'estimate_error_plot_px', 40,
                      'BBox centre error / last 5 seconds / 0-40 px; missing estimate at top')
                chart(canvas, evaluation.rows, (x+8, 757), t, YELLOW, 'roll_rate_dps', 90,
                      'Computed roll command / offline only / 0-90 deg/s')
            flags = ' | '.join(name for key, name in (('blur', 'IMAGE BLUR'), ('occluded', 'TARGET OCCLUDED'),
                ('distractor', 'SECOND DRONE DISTRACTOR')) if movement[key]) or 'VISIBLE TARGET'
            text(canvas, flags, (24, 834), .46, YELLOW)
            text(canvas, 'GREEN: placed target  |  YELLOW: matching neural box  |  RED: other neural box  |  BLUE: tracker estimate',
                 (24, 856), .42, WHITE)
            text(canvas, 'No fabricated model detections. Flow / follower estimates are counted separately. No live flight commands.',
                 (24, 875), .39, DIM)
            process.stdin.write(canvas.tobytes())
            if i == fps*6:
                cv2.imwrite(str(args.output / (name+'-preview.png')), canvas)
            if i % (fps*2) == 0:
                print(name, f'{t:.1f}/{args.duration:.1f}s',
                      [(sum(r['fresh_detector_hit'] for r in e.rows),
                        sum(r['measured_tracking_hit'] for r in e.rows)) for e in evaluations], flush=True)
    finally:
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('Video encoding failed')
    summary = dict(description=__doc__, video=str(output.resolve()), fps=fps,
        duration_s=args.duration, seed=seed, native_frame=list(native), display_frame=[W, H],
        engine=str(args.engine.resolve()), engine_sha256=hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        backend=type(model.predictor.model.backend).__name__, device=str(model.predictor.model.device),
        model_confidence_threshold=float(model.predictor.args.conf),
        original_follower_enabled=True, sky_gate_enabled=True,
        hit_rule='Centre distance <= max(3 px, half placed width), bbox IoU >= 0.10',
        coverage_denominator='All video frames, including deliberately occluded target frames',
        ground_truth_usage='Drawing and scoring only; never passed to detector/tracker/guidance',
        source=json.loads((args.assets/'source.json').read_text()),
        baseline=metrics(evaluations[0]), assisted=metrics(evaluations[1]))
    (args.output/(name+'-metrics.json')).write_text(json.dumps(summary, indent=2)+'\n')
    for side, evaluation in zip(('off', 'on'), evaluations):
        with (args.output/(name+'-'+side+'.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(evaluation.rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(evaluation.rows)
        with (args.output/(name+'-'+side+'-detections.jsonl')).open('w') as handle:
            for event in evaluation.events:
                handle.write(json.dumps(event)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, default=Path('runs/detect/v0/weights/best.engine'))
    parser.add_argument('--output', type=Path, default=Path('reports/wasp_motion'))
    parser.add_argument('--assets', type=Path, default=Path('reports/wasp_motion/assets'))
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--ffmpeg')
    parser.add_argument('--only', choices=('cruise', 'bursts'))
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('--duration must be positive and finite')
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.engine), task='detect')
    model(np.zeros((960, 1280, 3), np.uint8), imgsz=640, verbose=False)
    backend = type(model.predictor.model.backend).__name__
    if backend != 'TensorRTBackend':
        raise RuntimeError('Genuine TensorRT inference required; found '+backend)
    sprite = cv2.imread(str(args.assets/'real-mosquito-target.png'), cv2.IMREAD_UNCHANGED)
    background = cv2.imread(str(args.assets/'real-sky.jpg'))
    if sprite is None or sprite.shape[2] != 4 or background is None:
        raise RuntimeError('Missing real target/sky assets')
    if args.only != 'bursts':
        run(model, args, sprite, background, 'wasp-tensorrt-random-cruise-30fps', 30, 91027, False)
    if args.only != 'cruise':
        run(model, args, sprite, background, 'wasp-tensorrt-random-bursts-60fps', 60, 58041, True)


if __name__ == '__main__':
    main()
