# Drone playground

In the main app, open **Menu → Minigame** (below Messages). The playground opens
inside the app and is served on the same port; no separate minigame server is
needed. Closing the panel unloads the simulation.

To run it separately from the repository root with Python 3 (standard library only):

```sh
python3 src/minigame/wasp_minigame.py
```

Open <http://127.0.0.1:8091/>. Use `--port 8092` for a different port or
`--host 0.0.0.0` to make the page available on your local network.
Everything is served locally; there are no CDN or model downloads.

Choose the app's 75%, 85% or 100% throttle and Smooth, Standard, Responsive or
Extra response. Set platform elevation, compass orientation, drone roll, platform
height, countdown and flight duration, then press **Play flight**. Watch the
third-person animation and telemetry; pause/resume or reset to try another setup.
Overview automatically fits the whole flight path. Switch to Follow for a close
view of the drone. Drag to orbit and scroll to zoom. The lime trail and its
one-second markers show the actual simulated trajectory.
Playback speed changes presentation time, preserving the fixed physics step.
The scene uses procedural sky/clouds, textured ground and a shaded pointed 3D
body. Its geometry, fins, shadow and display scale are visual only; they do not
change the simulated forces or steering.

Choose **Box movement** before pressing Play:

- **Automatic centring** (default): after the countdown, the box moves linearly
  to the centre over the centre timer. A zero timer centres immediately.
- **Mouse control**: drag inside the preview during flight to move the box.
  Release to hold its position. Arrow keys also work when the preview is focused.
  This mode ignores the centre timer. Pointer movement is smoothed over 40 ms so
  the fixed physics steps receive continuous movement between mouse events.

Drag the preview before flight to choose the initial box position. The box stays
inside the image, including when dragging outside the preview. Pause freezes box
input and flight together; reset restores the initial position. Setup controls
lock during playback and pause. With test mode off, steering requests are zero.

Each response has an **Advanced settings** panel with the same allowed values as
`Wasp.SETTINGS`: straight launch, steering blend, maximum rate, gain, target lead,
and camera alignment. Values load from the app's saved profiles. Edits and centre
timers stay separate for each profile within the current browser session; reloading
restores saved app values. Setup is locked while running or paused.

Steering uses **frame-to-frame X/Y movement divided by simulation time**. A
stationary box produces no turn, even away from the centre. Identical movement
produces identical steering at any image position. A leftward image shift requests
a rightward turn (and vice versa), so the box moving inward corresponds to the
drone turning toward its starting direction. Both axes are combined.

The movement estimate uses an 80 ms filter. Gain scales motion, and motion lead
anticipates changes in filtered image velocity. FOV converts normalized movement
to approximate angular velocity, and camera alignment rotates the input axes.
The app's combined rate cap, straight-launch interval, smoothstep steering blend,
and command slew limit (`2 * rate / blend`) shape the result. A separate, assumed
60 ms angular response follows the command. Stationary samples clear the motion
estimate and stop rotation, preserving current attitude and linear velocity.

This motion-driven controller intentionally differs from `WaspGuidance` in the
live app, which uses absolute bearing error and world-bearing feed-forward.
The app's profile values and blend/slew behavior are retained, but the full
controller is no longer a numerical replica. Application body X corresponds to
the display's forward axis, body Y to right, and body Z opposite thrust; image up
is -body X at zero camera alignment. Body-rate telemetry uses app roll/pitch order.

Camera field-of-view controls default to `CAM_HFOV_DEG` / `CAM_VFOV_DEG` in the
server environment, or the app defaults of 62.2° / 48.8°. Enter effective FOVs
including crop if the live app uses a cropped camera mode; live camera state is
not read by this standalone simulator.

Acro integrates rotation without automatic attitude levelling. When the box reaches
the centre, the playground clears steering commands, bearing history and angular
velocity to end the turn immediately. It holds the current attitude and preserves
linear velocity; thrust, gravity and drag continue, so the spatial trajectory can
still curve. This idealized centre hold is a simulator-specific override: the app
controller itself can still request rotation from bearing-motion feed-forward
with a centred box. The box is scripted in screen coordinates, not a
world-space target projected from the changing drone pose. Its **linear** centring
is intentionally retained; the app's test box instead uses smoothstep easing.
The simulation ends at ground contact or the chosen duration. To compare profiles, use the same initial box, centre timer and throttle.

The server reads `~/.config/wasp/wasp.json`, falling back to the application's
presets. Set `WASP_DEFAULTS=/path/to/wasp.json` to preview a different settings
file. Simulator changes stay in the browser and never overwrite application
settings. There is no detector, enemy drone, interception scoring, Mission
execution, camera access or flight-controller connection.

Physics is illustrative, not an airframe calibration: thrust-to-weight 4,
drag-limited speed 40 m/s, initial launch speed 12 m/s, first-order angular response
of 60 ms. The drone is drawn larger for visibility. Guidance observations and
commands use ideal, zero-latency 5 ms steps. Real camera/control timing, detection
noise, RC calibration, FC ACRO rate/expo/PWM limits, battery, and motor dynamics
are not reproduced. The app supplies steering commands, not measured thrust or
aerodynamic coefficients: matching its profiles does not calibrate the trajectory
to a real airframe.

## Verification

```sh
python3 -m unittest discover -s src/minigame -p 'test_*.py' -v
```

The settings tests use the standard library. Browser tests require Selenium and
Chrome; they exercise the actual JavaScript physics, controls and responsive page.
Tests check stationary off-centre boxes, movement direction, speed scaling,
position independence, step-size independence and the centre hold. Differential
tests compare the retained command blend/slew stage against `Wasp.output` for all
four profiles and every allowed advanced-setting value.
