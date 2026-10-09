import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wasp import Wasp, WaspGuidance, rotate, rotation, unit
from wasp_motion import WaspMotionEstimator, project_bearing
from mission import Mission, phone_command
from tracker import DroneTracker, KalmanTracker
from test_wasp import ATT, TH, TV, Mapper, backend
from fc_control import AcroCommander

W, H = 640, 480


def image(x=320, y=240, size=16, texture=True):
    frame = np.full((H, W, 3), 205, np.uint8)
    if texture:
        s = max(3, size // 2)
        cv2.line(frame, (round(x-s), round(y-s//2)), (round(x+s), round(y+s//2)), (30, 30, 30), 2)
        cv2.line(frame, (round(x-s), round(y+s//2)), (round(x+s), round(y-s//2)), (30, 30, 30), 2)
        for dx, dy in ((-s, -s//2), (s, -s//2), (-s, s//2), (s, s//2)):
            cv2.circle(frame, (round(x+dx), round(y+dy)), 2, (40, 40, 40), -1)
    return frame


def sample(e, x=320, y=240, t=100, size=8, conf=.9, att=ATT, frame=None, synthetic=False):
    return e.update(image(texture=False) if frame is None else frame,
                    (x, y, size, size, conf), att, t, TH, TV, synthetic=synthetic)


class EstimatorTests(unittest.TestCase):
    def test_temporal_filter_reduces_center_and_command_noise_at_both_fps(self):
        for fps in (30, 60):
            e, g, rng = WaspMotionEstimator(), WaspGuidance(), np.random.default_rng(45)
            raw, filtered, boxes = [], [], []
            for i in range(300):
                t = 100 + i/fps
                dx, dy = rng.normal(0, 1.0, 2)
                c = g.observe(dx, dy, W, H, TH, TV, ATT, t)
                o = sample(e, 320+dx, 240+dy, t)
                if o and i > fps:
                    raw.append(WaspGuidance.rates(c, ATT, t))
                    filtered.append(WaspGuidance.rates(o, ATT, t))
                    boxes.append(np.array(o['box'][:2]) - (320,240))
            self.assertGreater(len(filtered), 200)
            self.assertLess(np.sqrt(np.mean(np.square(filtered))), .6*np.sqrt(np.mean(np.square(raw))))
            self.assertLess(np.sqrt(np.mean(np.square(boxes))), .8)

    def test_rotating_aircraft_is_not_target_motion(self):
        world = unit((.12, .08, -1))
        for camera_roll in (0, 90, 180, 270):
            e = WaspMotionEstimator()
            for i in range(70):
                att = dict(roll=.2*math.sin(i*.1), pitch=.1*math.cos(i*.09), yaw=.3*math.sin(i*.08))
                x, y = project_bearing(world, att, W, H, TH, TV, camera_roll)
                o = e.update(image(texture=False), (x, y, 6, 4, .8), att,
                             100+i/60, TH, TV, camera_roll)
                self.assertIsNotNone(o)
                np.testing.assert_allclose(o['bearing'], world, atol=1e-10)
                self.assertLess(np.linalg.norm(o['bearing_rate']), 1e-9)

    def test_isolated_outlier_does_not_enter_motion_history(self):
        e = WaspMotionEstimator()
        for i in range(10):
            sample(e, t=100+i/30)
        self.assertIsNone(sample(e, x=365, y=265, t=100+10/30))
        o = sample(e, t=100+11/30)
        self.assertLess(np.linalg.norm(o['bearing_rate']), 1e-9)
        self.assertAlmostEqual(o['box'][0], 320)
        self.assertEqual(e.rejected, 1)

    def test_real_consistent_movement_passes_gate(self):
        e = WaspMotionEstimator()
        for i in range(20):
            sample(e, t=100+i/60)
        sample(e, x=338, t=100+20/60)
        o = sample(e, x=339, t=100+21/60)
        self.assertIsNotNone(o)
        self.assertGreater(o['box'][0], 326)

    def test_flow_bridges_short_detector_gap_and_rejects_distractor(self):
        e = WaspMotionEstimator()
        for i in range(3):
            x=320+2*i
            o = sample(e, x=x, size=24, t=100+i/30, frame=image(x, size=20))
            self.assertIsNotNone(o)
        o = e.update(image(326, size=20), None, ATT, 100+.1, TH, TV)
        self.assertIsNotNone(o)
        self.assertEqual(o['source'], 'FLOW')
        self.assertAlmostEqual(e.box[0], 326, delta=1)
        o = sample(e, x=370, size=24, t=100+4/30, frame=image(328, size=20))
        self.assertIsNotNone(o)
        self.assertEqual(o['source'], 'FLOW')
        self.assertLess(o['box'][0], 330)
        self.assertEqual(e.rejected, 1)

    def test_flow_cannot_extend_detector_confirmation_forever(self):
        e = WaspMotionEstimator()
        sample(e, size=24, frame=image(size=20))
        for i in range(1, 15):
            o = e.update(image(320+i, size=20), None, ATT, 100+i/60, TH, TV)
            if i/60 > e.FLOW_TTL:
                self.assertIsNone(o)

    def test_tiny_featureless_target_does_not_invent_flow(self):
        e = WaspMotionEstimator()
        sample(e, size=2)
        self.assertIsNone(e.points)
        self.assertIsNone(e.update(image(texture=False), None, ATT, 100.03, TH, TV))

    def test_blur_or_target_disappearance_invalidates_flow(self):
        e = WaspMotionEstimator()
        sample(e, size=24, frame=image(size=20))
        self.assertIsNone(e.update(image(texture=False), None, ATT, 100.03, TH, TV))

    def test_gap_geometry_and_timestamp_reset_history(self):
        e = WaspMotionEstimator()
        for i in range(10): sample(e, x=320+i, t=100+i/60)
        o = sample(e, x=400, t=101)
        self.assertIsNone(o)
        o = sample(e, x=400, t=101.03)
        self.assertEqual(o['box'][0], 400)
        self.assertEqual(o['motion_confidence'], 0)
        o = e.update(image(texture=False), (330, 240, 8, 8, .9), ATT, 101.02, TH, TV, 90)
        self.assertEqual(o['motion_confidence'], 0)
        o = sample(e, t=99)
        self.assertEqual(o['motion_confidence'], 0)

    def test_low_confidence_invalid_values_and_attitude_do_not_steer(self):
        for detection in ((320,240,8,8,.1), (float('nan'),240,8,8,.9),
                          (-1,240,8,8,.9), (320,240,0,8,.9)):
            self.assertIsNone(WaspMotionEstimator().update(image(texture=False), detection, ATT, 100, TH, TV))
        self.assertIsNone(sample(WaspMotionEstimator(), att=None))
        self.assertIsNone(sample(WaspMotionEstimator(), att={'rotation':np.zeros((3,3))}))
        self.assertIsNone(sample(WaspMotionEstimator(), att=dict(roll=float('nan'),pitch=0,yaw=0)))

    def test_repeated_false_boxes_fail_appearance_check_without_flow(self):
        e = WaspMotionEstimator()
        sample(e, size=24, frame=image(size=20))
        e.points=None
        for i in (1,2,3):
            self.assertIsNone(sample(e,x=360,size=24,t=100+i/30,frame=image(size=20)))
        self.assertAlmostEqual(e.box[0],320)

    def test_synthetic_box_uses_filtering_without_sampling_image_texture(self):
        e = WaspMotionEstimator()
        for i in range(12):
            o=sample(e, x=320+i*4, t=100+i/30, frame=image(), synthetic=True)
            self.assertIsNotNone(o)
            self.assertIsNone(e.points)
            self.assertEqual(o['source'],'DETECTOR')

    def test_time_based_filter_consistent_at_30_and_60_fps(self):
        results=[]
        for fps in (30,60):
            e=WaspMotionEstimator()
            for i in range(fps+1):
                o=sample(e, x=320+40*i/fps, t=100+i/fps)
            results.append(o['box'][0])
        self.assertLess(abs(results[0]-results[1]),1)
        self.assertGreater(min(results),357)


class IntegrationTests(unittest.TestCase):
    def test_assisted_detector_selects_target_over_higher_score_distractor(self):
        class Tensor:
            def __init__(self, values): self.values=np.array(values)
            def cpu(self): return self
            def numpy(self): return self.values
            def int(self): return self
            def tolist(self): return self.values.tolist()
            def __getitem__(self,index): return Tensor(self.values[index])
        t=DroneTracker.__new__(DroneTracker)
        t.full_w,t.full_h,t.fw,t.fh=W,H,W,H
        t.sky_mask=None;t.conf_thresh=.2;t.target_class=0
        t._motion_hint=(330,240);t.last_bbox=(326,236,8,8)
        boxes=SimpleNamespace(conf=Tensor([.95,.7]),xyxy=Tensor([[376,206,384,214],[326,236,334,244]]))
        t.model=Mock(return_value=[SimpleNamespace(boxes=boxes)])
        t._motion_enabled=True
        self.assertEqual(t._detect_region(image(),(0,0,W,H),640,False)[:2],(330,240))
        self.assertEqual(t.model.call_args.kwargs['max_det'],6)
        t._motion_enabled=False
        self.assertEqual(t._detect_region(image(),(0,0,W,H),640,False)[:2],(380,210))
        self.assertEqual(t.model.call_args.kwargs['max_det'],1)

    def test_switch_saved_independent_of_response_and_locked_during_launch(self):
        with tempfile.TemporaryDirectory() as td:
            path=str(Path(td)/'wasp.json')
            mapper=Mapper()
            m=Mission(mapper,wasp_path=path)
            self.assertEqual(m.wasp.params['motion_assist'],0)
            self.assertEqual(phone_command(m,'waspp','motion_assist:1',100),'')
            for p in Wasp.STEERING_PROFILES:
                self.assertEqual(m.set_wasp_param('profile',p),'')
                self.assertEqual(m.wasp.params['motion_assist'],1)
            self.assertEqual(Mission(Mapper(),wasp_path=path).wasp.params['motion_assist'],1)
            self.assertEqual(json.loads(Path(path).read_text())['shared']['motion_assist'],1)
            self.assertTrue(m.set_wasp_param('motion_assist',2))
            m.wasp.launch_t=103
            self.assertTrue(m.set_wasp_param('motion_assist',0))
            m.wasp.launch_t=None
            m.set_wasp_enabled(True)
            m.phase='WASP_FLIGHT'
            mapper.armed=True
            self.assertTrue(m.set_wasp_param('motion_assist',0))

    def test_detector_anchor_is_required_by_flight_gate(self):
        o=sample(WaspMotionEstimator())
        w=Wasp()
        w.params['motion_assist']=1
        ctx=dict(att=ATT,ai_cmd=o,det_t=100.24,det_conf=.9)
        self.assertEqual(w.fault(100.24,ctx,.2),'')
        o['capture_t']=100.26
        self.assertTrue(w.fault(100.26,ctx,.2))

    def test_enabled_tracker_skips_kalman_and_legacy_follower(self):
        t=DroneTracker.__new__(DroneTracker)
        mission=Mission(Mapper());mission.set_wasp_enabled(True);mission.wasp.params['motion_assist']=1
        t.ap=SimpleNamespace(mission=mission,wasp_motion_context=lambda now:(ATT,TH,TV))
        t.fw,t.fh,t.full_w,t.full_h=W,H,W,H
        t.kalman=Mock();t.follower=Mock();t.motion=WaspMotionEstimator();t._motion_enabled=True
        t.use_sky_mask=False;t.sky_mask=None;t._frame_dt=1/30
        t._t_last_frame=None;t.mode_name=mission.ai_level_now()
        t._detect_region=Mock(return_value=(320,240,316,236,8,8,.9))
        result=t._process(image(texture=False),image(texture=False),100)
        self.assertTrue(result[4]);self.assertEqual(result[0],(320,240))
        t.kalman.predict.assert_not_called();t.kalman.update.assert_not_called()
        t.follower.track.assert_not_called()
        result=t._process(image(texture=False),image(texture=False),100.03,run_detector=False)
        self.assertFalse(result[4]);self.assertEqual(t._detect_region.call_count,1)

    def test_test_mode_assistance_bypasses_models_and_flow(self):
        t=DroneTracker.__new__(DroneTracker)
        m=Mission(Mapper());m.set_wasp_enabled(True);m.set_test_mode(True);m.set_test_target('.6:.4')
        m.wasp.params['motion_assist']=1
        t.ap=SimpleNamespace(mission=m,wasp_motion_context=lambda now:(ATT,TH,TV))
        t.fw,t.fh=W,H;t.kalman=Mock();t.follower=Mock();t.sky=Mock()
        t._t_last_frame=None
        t._detect_region=Mock()
        with (patch.object(KalmanTracker, 'predict', side_effect=AssertionError('Kalman used')),
              patch.object(KalmanTracker, 'update', side_effect=AssertionError('Kalman used'))):
            self.assertTrue(t._process(image(),image(),100)[4])
        t._detect_region.assert_not_called()
        self.assertIsNone(t.motion.points)

    def test_default_off_leaves_guidance_output_unchanged(self):
        b=backend()
        ap=AcroCommander.__new__(AcroCommander)
        ap.backend=b;ap.mission=Mission(Mapper());ap.mission.set_wasp_enabled(True)
        ap.law=SimpleNamespace(neutral=lambda:{},_tan_h=TH,_tan_v=TV)
        ap.wasp_guidance=WaspGuidance()
        b.wasp_attitude=lambda at_t=None: ATT
        expected=WaspGuidance().observe(20,-15,W,H,TH,TV,ATT,100)
        actual=ap.compute(20,-15,'LOCK',W,H,capture_t=100,observation={'bogus':True})
        expected['camera_roll']=0
        self.assertEqual(actual,expected)
        ap.mission.wasp.params['motion_assist']=1
        obs=sample(WaspMotionEstimator(),x=340,y=225)
        self.assertEqual(ap.compute(20,-15,'LOCK',W,H,capture_t=100,observation=obs),obs)
        self.assertFalse(ap.compute(20,-15,'LOCK',W,H,capture_t=100).get('wasp'))


if __name__=='__main__':
    unittest.main()
