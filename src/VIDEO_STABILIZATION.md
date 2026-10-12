Video Stabilization is in Parameters → Detection, directly below Camera's
30 FPS / 60 FPS buttons. It starts OFF and can be changed while the CSI camera
is running, including during Wasp flight. The existing control lease applies.

ON stabilizes the web stream and local OpenCV preview. OFF restores the original
annotated feed on the next camera frame. Neither command reopens the camera.
Detection, guidance, camera settings, and recorded video keep their existing
coordinates and behavior. Wasp motion assist has been removed; old saved entries
are ignored and cannot enable it.

Motion is estimated on a grayscale image up to 320 pixels wide with at most 160
corners, sparse pyramidal Lucas–Kanade flow, forward/backward validation, and a
RANSAC rigid-motion fit. Scale estimates are discarded. A causal exponential
filter smooths translation and rotation without waiting for future frames.
The preview worker retains only the newest pending frame, so slow processing
cannot build a frame queue or hold up detection. OpenCV's existing thread setting
is preserved. Optical-flow details: [OpenCV tracking documentation](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html).

The output canvas has fixed black padding for the current frame dimensions.
Its size accommodates the maximum correction (6% translation per axis and 5°
rotation), including all four corners. Source pixels are neither cropped nor
zoomed. Padding keeps the same aspect ratio to rounding precision. Test-target
taps are mapped through the inverse preview transform; taps in black padding are
ignored. Rendering uses [OpenCV affine warping](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).

Fresh existing ATTITUDE angular rates, in radians per second, seed optical flow
using the configured camera mounting angle and camera alignment. Gyro assistance
requires the normal camera flip setting (CAM_FLIP_METHOD=0); other flips use
optical flow. Missing or stale telemetry also falls back to optical flow. After
a reliable visual estimate, gyro can bridge feature loss for up to 200 ms.
Loss of features, camera/mount changes, or gaps over 150 ms reset the filter.
This is an approximate rigid-motion correction; it does not remove rolling
shutter distortion or motion blur.

The existing `/state` endpoint includes `video_stabilization`,
`stabilization.source`, and `stabilization.ms` (analysis plus rendering time).
The existing `/cmd` endpoint accepts `stabilization=on` or `stabilization=off`.
The configured STREAM_FPS limit remains unchanged (15 FPS by default); camera
processing remains at the selected camera rate independently of the preview
worker. To display 30 FPS over MJPEG, use the existing STREAM_FPS=30 setting.

Run the stabilization checks from the project root:

```sh
.env/yolo/bin/python src/tests/test_video_stabilization.py
```

They exercise shake reduction at 30/60 FPS, preservation of every frame corner,
gyro fallback, recovery, live toggles, stale worker results, replacement of queued
frames, the HTTP command endpoint, and unchanged detection/recording frames.
