"""Run with python3 -m unittest discover -s src/minigame -p 'test_*.py'.
Browser integration needs Selenium and Chrome installed.
"""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import urlopen

from wasp_minigame import Handler, ThreadingHTTPServer, settings
from wasp import Wasp


class SettingsTests(unittest.TestCase):
    def test_saved_profiles_are_read_without_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'settings.json'
            original = json.dumps(dict(selected='smooth', profiles=dict(smooth=dict(rate=30)),
                                       shared=dict(throttle=.85, launch_s=2)))
            path.write_text(original)
            with patch.dict(os.environ, WASP_DEFAULTS=str(path)):
                result = settings()
            self.assertEqual(result['selected'], 'smooth')
            self.assertEqual(result['profiles']['smooth']['rate'], 30)
            self.assertEqual(result['throttle'], .85)
            self.assertEqual(path.read_text(), original)

    def test_all_saved_advanced_values_and_choices_match_app(self):
        profiles = {name: dict(straight=2, blend=.25, rate=30, gain=6,
                               lead=.15, camera_roll=270, centre_timer=4.5)
                    for name in Wasp.STEERING_PROFILES}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'settings.json'
            original = json.dumps(dict(selected='extra', profiles=profiles))
            path.write_text(original)
            with patch.dict(os.environ, WASP_DEFAULTS=str(path), CAM_HFOV_DEG='70', CAM_VFOV_DEG='nan'):
                result = settings()
            self.assertEqual(result['profiles'], profiles)
            self.assertEqual(result['profile_choices'], {k: v for k, v in Wasp.SETTINGS.items() if k != 'throttle'})
            self.assertEqual(result['camera'], dict(hfov=70, vfov=48.8))
            self.assertEqual(path.read_text(), original)

    def test_invalid_file_falls_back_to_presets(self):
        with patch.dict(os.environ, WASP_DEFAULTS='/nonexistent/settings.json'):
            self.assertEqual(settings()['throttle_choices'], (.75, .85, 1))


class BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from selenium import webdriver
        except ImportError:
            raise unittest.SkipTest('Selenium is not installed')
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/'
        options = webdriver.ChromeOptions()
        for arg in ('--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--window-size=1440,1100'):
            options.add_argument(arg)
        try:
            cls.driver = webdriver.Chrome(options=options)
        except Exception:
            cls.server.shutdown()
            cls.server.server_close()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        from selenium.webdriver.support.ui import WebDriverWait
        self.driver.get(self.url)
        WebDriverWait(self.driver, 10).until(lambda d: d.find_element('id', 'play').is_enabled())

    def js(self, script):
        return self.driver.execute_script(script)

    def test_motion_is_position_independent_and_stops_when_stationary(self):
        result = self.js("""
          const profile={...settings.profiles.standard,straight:0,lead:0};
          const run=(start,speed,dt=.005)=>{
            const c=new ProfileController(profile,config().camera,0);
            c.update(start,0);
            for(let i=1;i<=Math.round(2/dt);i++)c.update(add(start,mul(speed,i*dt)),i*dt);
            return c;
          };
          const slow=run([.8,.2],[-.02,.01]),fast=run([.8,.2],[-.04,.02]);
          const shifted=run([.3,.6],[-.02,.01]),coarse=run([.8,.2],[-.02,.01],.01);
          const reverse=run([.8,.2],[.02,-.01]);
          const request=slow.requested.slice();
          slow.update(slow.previous.box,2.005);
          const stopped={request:slow.requested,command:slow.command};
          const fixed=new Flight({...config(),launch:0,duration:5,profile,box:[.8,.2]});
          fixed.box=()=>[.8,.2];
          for(let i=0;i<400;i++)fixed.step(.005);
          return {request,fast:fast.requested,shifted:shifted.requested,coarse:coarse.requested,
            reverse:reverse.requested,stopped,fixed:fixed.omega};
        """)
        self.assertGreater(result['request'][0], 0)
        self.assertGreater(result['request'][1], 0)
        for axis in range(3):
            self.assertAlmostEqual(result['fast'][axis], 2*result['request'][axis], places=8)
            self.assertAlmostEqual(result['shifted'][axis], result['request'][axis], places=8)
            self.assertAlmostEqual(result['coarse'][axis], result['request'][axis], places=8)
            self.assertAlmostEqual(result['reverse'][axis], -result['request'][axis], places=8)
        self.assertEqual(result['stopped']['request'], [0, 0, 0])
        self.assertEqual(result['stopped']['command'], [0, 0, 0])
        self.assertEqual(result['fixed'], [0, 0, 0])

    def test_motion_commands_keep_app_blend_and_slew_for_all_advanced_values(self):
        class Mapper:
            @staticmethod
            def wasp_pwm(rates, throttle):
                return rates

        cases = []
        for name, profile in Wasp().profiles.items():
            variants = [('default', profile)]
            for key, choices in Wasp.SETTINGS.items():
                if key != 'throttle':
                    variants.extend((f'{key}={value}', dict(profile, **{key: value})) for value in choices)
            cases.extend(dict(name=f'{name}/{label}', profile=params) for label, params in variants)
        results = self.driver.execute_script("""
          return arguments[0].map(c=>{
            const controller=new ProfileController(c.profile,config().camera,0),samples=[];
            controller.update([.9,.2],0);
            for(let i=1;i<=81;i++){
              const t=i*.05;
              const command=controller.update([.9-.05*t,.2+.025*t],t);
              samples.push({t,request:controller.requested,command});
            }
            return {...c,samples};
          });
        """, cases)
        # Compare only the retained output stage against the actual app. Its
        # position-based guidance is deliberately replaced by image motion.
        with patch('wasp.WaspGuidance.rates') as rates:
            for case in results:
                controller = Wasp()
                controller.params.update(case['profile'])
                controller.begin(0)
                for sample in case['samples']:
                    rates.return_value = sample['request']
                    expected = controller.output(Mapper(), sample['t'], dict(ai_cmd={}, att={}))
                    for actual, value in zip(sample['command'], expected):
                        self.assertAlmostEqual(actual, value, places=10, msg=case['name'])

    def test_advanced_settings_are_per_profile_and_locked_in_flight(self):
        result = self.js("""
          document.querySelector('[data-profile="smooth"]').click();
          $('profile_lead').value=.15;$('profile_lead').dispatchEvent(new Event('change'));
          $('profile_camera_roll').value=90;$('profile_camera_roll').dispatchEvent(new Event('change'));
          $('centre').value=4.5;$('centre').dispatchEvent(new Event('input'));
          document.querySelector('[data-profile="extra"]').click();
          const extra={...config().profile};
          document.querySelector('[data-profile="smooth"]').click();
          const smooth={...config().profile},centre=config().centre;
          $('play').click();
          const locked=Array.from(document.querySelectorAll('#advancedFields select')).every(e=>e.disabled);
          $('play').click();$('reset').click();
          return {extra,smooth,centre,locked,unlocked:!$('profile_lead').disabled};
        """)
        self.assertEqual(result['smooth']['lead'], .15)
        self.assertEqual(result['smooth']['camera_roll'], 90)
        self.assertEqual(result['centre'], 4.5)
        self.assertEqual(result['extra']['lead'], self.js('return settings.profiles.extra.lead'))
        self.assertTrue(result['locked'])
        self.assertTrue(result['unlocked'])

    def test_physics_and_linear_box(self):
        result = self.js("""
          const base={...config(),launch:0,duration:8,elevation:90,box:[.5,.5],centre:2};
          const run=(changes,seconds=2)=>{const f=new Flight({...base,...changes});for(let i=0;i<seconds/.005;i++)f.step(.005);return f;};
          const low=run({throttle:.75}),high=run({throttle:1});
          const north=run({elevation:45,heading:0}),east=run({elevation:45,heading:90});
          const countdown=run({launch:3,box:[.8,.2]});
          const quarter=run({box:[.8,.2]},.5),mid=run({box:[.8,.2]},1);
          const centered=run({box:[.8,.2]}),instant=run({box:[.8,.2],centre:0},.005);
          const ground=run({elevation:0},4);
          return {low:low.p[2],high:high.p[2],north:north.p,east:east.p,
            countdown:countdown,waitingBox:countdown.box(),quarter:quarter.box(),mid:mid.box(),
            box:centered.box(),instant:instant.box(),ground:ground.phase};
        """)
        self.assertGreater(result['high'], result['low'])
        self.assertGreater(result['north'][1], 0)
        self.assertAlmostEqual(result['north'][1], result['east'][0], places=7)
        self.assertFalse(result['countdown']['launched'])
        self.assertEqual(result['countdown']['p'][0:2], [0, 0])
        self.assertEqual(result['waitingBox'], [.8, .2])
        for key, expected in [('quarter', [.725, .275]), ('mid', [.65, .35]),
                              ('box', [.5, .5]), ('instant', [.5, .5])]:
            for actual, value in zip(result[key], expected):
                self.assertAlmostEqual(actual, value, places=7)
        self.assertEqual(result['ground'], 'GROUND CONTACT')

    def test_box_direction_rate_limit_and_acro_attitude(self):
        result = self.js("""
          const base={...config(),launch:0,duration:10,elevation:90,roll:0,heading:0,
            testMode:true,centre:2,profile:{...settings.profiles.standard,straight:0}};
          const run=(box,changes={})=>{const f=new Flight({...base,...changes,box});for(let i=0;i<400;i++)f.step(.005);return f;};
          const right=run([.8,.5]),left=run([.2,.5]),up=run([.5,.2]),down=run([.5,.8]);
          const center=run([.5,.5]),off=run([.8,.2],{testMode:false});
          const held=run([.8,.5]);held.cfg.testMode=false;for(let i=0;i<400;i++)held.step(.005);
          const attitude=held.R.map(v=>v.slice());for(let i=0;i<200;i++)held.step(.005);
          const limited=new Flight({...base,box:[.95,.05],profile:{...base.profile,gain:6}});
          limited.step(.005);
          const straight=new Flight({...base,box:[.8,.2],profile:{...base.profile,straight:1}});
          for(let i=0;i<100;i++)straight.step(.005);
          return {right:right.p,left:left.p,up:up.p,down:down.p,center:center.p,off:off.p,
            limit:norm(limited.steeringRates()),rate:radians(base.profile.rate),
            straight:straight.omega,request:held.steeringRates(),omega:held.omega,
            tilt:held.R[2],drift:norm(sub(attitude[2],held.R[2]))};
        """)
        self.assertGreater(result['right'][0], 1)
        self.assertAlmostEqual(result['right'][0], -result['left'][0], places=7)
        # Image up is -body X in the app's thrust-axis camera frame.
        self.assertLess(result['up'][1], -1)
        self.assertGreater(result['down'][1], 1)
        for key in ['center', 'off']:
            self.assertAlmostEqual(result[key][0], 0)
            self.assertAlmostEqual(result[key][1], 0)
        self.assertLessEqual(result['limit'], result['rate'] + 1e-10)
        self.assertEqual(result['straight'], [0, 0, 0])
        self.assertEqual(result['request'], [0, 0, 0])
        self.assertLess(max(abs(v) for v in result['omega']), 1e-6)
        self.assertGreater(result['tilt'][0], .1)
        self.assertLess(result['drift'], 1e-6)

    def test_centered_box_stops_turning_for_every_profile(self):
        results = self.js("""
          return Object.entries(settings.profiles).flatMap(([name,profile])=>
            [0,90,180,270].map(camera_roll=>{
              const f=new Flight({...config(),launch:0,duration:8,height:10,elevation:90,
                centre:2,box:[.75,.3],profile:{...profile,lead:.15,camera_roll}});
              for(let i=0;i<399;i++)f.step(.005);
              const before=f.R.map(v=>v.slice()),position=f.p.slice(),speed=norm(f.v);
              const wasTurning=norm(f.omega);
              f.step(.005);
              const boundaryDrift=Math.max(...f.R.map((v,i)=>norm(sub(v,before[i]))));
              for(let i=0;i<100;i++)f.step(.005);
              return {name,camera_roll,wasTurning,boundaryDrift,done:f.done,
                attitudeDrift:Math.max(...f.R.map((v,i)=>norm(sub(v,before[i])))),
                request:f.steeringRates(),command:f.controller.command,omega:f.omega,
                history:f.controller.previous,movement:norm(sub(f.p,position)),speed};
            }));
        """)
        for result in results:
            with self.subTest(profile=result['name'], alignment=result['camera_roll']):
                self.assertFalse(result['done'])
                self.assertGreater(result['wasTurning'], .001)
                self.assertEqual(result['boundaryDrift'], 0)
                self.assertEqual(result['attitudeDrift'], 0)
                for key in ('request', 'command', 'omega'):
                    self.assertEqual(result[key], [0, 0, 0])
                self.assertIsNone(result['history'])
                self.assertGreater(result['movement'], 1)
                self.assertGreater(result['speed'], 1)

    def test_default_flight_turns_and_profiles_change_the_curve(self):
        from selenium.webdriver.support.ui import WebDriverWait
        self.assertEqual(self.js("return document.querySelectorAll('#steering,#bank,#neutral,[data-key]').length"), 0)
        self.assertEqual(self.js("return document.querySelectorAll('[data-profile]').length"), 4)
        self.js("$('launch').value=0;$('playback').value=2;$('play').click()")
        WebDriverWait(self.driver, 8).until(lambda d: self.js('return flight.t>3'))
        self.js("$('play').click()")
        self.assertGreater(self.js('return Math.abs(flight.p[0])'), 1)
        self.assertFalse(self.js('return follow'))
        result = self.js("""
          return Object.entries(settings.profiles).map(([name,profile])=>{
            const f=new Flight({...config(),launch:0,duration:10,centre:3,box:[.75,.3],profile});
            for(let i=0;i<700;i++)f.step(.005);
            const a=sub(f.path[10],f.path[0]),b=sub(f.p,f.path[f.path.length-10]);
            return {name,side:f.p[0],bend:Math.acos(clamp(dot(unit(a),unit(b)),-1,1))};
          });
        """)
        for trajectory in result:
            self.assertGreater(trajectory['bend'], .1, trajectory['name'])
        by_name = {r['name']: r for r in result}
        self.assertGreater(by_name['extra']['side'], by_name['smooth']['side'] + 1)
        self.assertEqual(len({round(r['side'], 5) for r in result}), 4)
        self.js("$('follow').click()")
        self.assertTrue(self.js('return follow'))

    def test_mouse_mode_moves_in_flight_and_holds_without_timer(self):
        result = self.js("""
          $('boxMotion').value='mouse';
          const cfg={...config(),launch:0,duration:10,centre:0,box:[.7,.3],
            profile:{...settings.profiles.standard,straight:0,blend:.25}};
          flight=new Flight(cfg);running=true;
          for(let i=0;i<100;i++)flight.step(.005);
          const initial=flight.box(),attitude=flight.R.map(v=>v.slice());
          placeBox(.3,.7);
          let peak=0;
          for(let i=0;i<160;i++){flight.step(.005);peak=Math.max(peak,norm(flight.steeringRates()));}
          const held=flight.box(),heldAttitude=flight.R.map(v=>v.slice());
          for(let i=0;i<100;i++)flight.step(.005);
          const drift=norm(sub(flight.R[2],heldAttitude[2]));
          paused=true;running=false;const blocked=placeBox(.9,.9);
          const pausedBox=flight.box();paused=false;running=true;
          placeBox(-2,3);const clamped=flight.mouseTarget.slice();
          flight=new Flight({...cfg,boxMotion:'timed'});
          const automaticBlocked=placeBox(.2,.8);
          running=false;reset();
          return {initial,held,peak,drift,blocked,pausedBox,clamped,automaticBlocked,
            turned:norm(sub(heldAttitude[2],attitude[2])),resetMouse:flight.mouseTarget};
        """)
        self.assertEqual(result['initial'], [.7, .3])
        self.assertEqual(result['held'], [.3, .7])
        self.assertGreater(result['peak'], .01)
        self.assertGreater(result['turned'], .01)
        self.assertEqual(result['drift'], 0)
        self.assertFalse(result['blocked'])
        self.assertEqual(result['pausedBox'], result['held'])
        self.assertEqual(result['clamped'], [.05, .95])
        self.assertFalse(result['automaticBlocked'])
        self.assertEqual(result['resetMouse'], [.75, .3])

    def test_mouse_drag_pointer_capture_and_release(self):
        from selenium.webdriver.common.action_chains import ActionChains
        from selenium.webdriver.support.ui import WebDriverWait
        self.js("$('boxMotion').value='mouse';$('launch').value=0;$('duration').value=30;$('play').click()")
        canvas = self.driver.find_element('id', 'bbox')
        self.js("""
          window.dragStart=null;
          boxCanvas.addEventListener('pointerdown',()=>{
            window.dragStart={captured:boxCanvas.hasPointerCapture(boxPointer),target:flight.mouseTarget.slice()};
          },{once:true});
        """)
        ActionChains(self.driver).move_to_element_with_offset(canvas, 30, 0).click_and_hold().pause(.1).move_by_offset(60, -25).release().perform()
        self.assertTrue(self.js('return dragStart.captured'))
        first = self.js('return dragStart.target')
        last = self.js('return flight.mouseTarget.slice()')
        self.assertGreater(last[0], first[0])
        self.assertLess(last[1], first[1])
        self.assertTrue(self.js('return boxPointer===null'))
        WebDriverWait(self.driver, 3).until(lambda d: self.js('return norm(sub(flight.box(),flight.mouseTarget))<1e-6'))
        self.assertEqual(self.js('return flight.mouseTarget'), last)
        self.js("$('play').click()")
        self.assertTrue(self.js('return paused'))

    def test_play_pause_reset_and_invalid_settings(self):
        from selenium.webdriver.support.ui import WebDriverWait
        self.js("$('launch').value=0;$('play').click()")
        WebDriverWait(self.driver, 5).until(lambda d: self.js('return flight.t>.15'))
        self.js("$('play').click()")
        self.assertTrue(self.js('return paused&&!running'))
        self.assertTrue(self.js("return $('height').disabled"))
        self.js("$('reset').click()")
        self.assertEqual(self.js('return flight.t'), 0)
        self.assertFalse(self.js("return $('height').disabled"))
        self.js("$('duration').value='';$('play').click()")
        self.assertFalse(self.js('return running'))
        self.assertIn('enter', self.js("return $('notice').textContent"))

    def test_assets_mobile_and_console(self):
        self.assertEqual(urlopen(self.url+'simulator.js').status, 200)
        self.assertEqual(urlopen(self.url+'guidance.js').status, 200)
        self.driver.set_window_size(390, 844)
        self.assertTrue(self.js('return document.documentElement.scrollWidth <= innerWidth'))
        self.assertEqual([e for e in self.driver.get_log('browser') if e['level']=='SEVERE'], [])
        self.driver.set_window_size(1440, 1100)


if __name__ == '__main__':
    unittest.main()
