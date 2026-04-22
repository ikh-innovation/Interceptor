import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# PID CONTROLLER
# =========================
class PID:
    def __init__(self, kp=0.1, ki=0.002, kd=0.03):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.ix = 0
        self.iy = 0
        self.px = 0
        self.py = 0

    def reset(self):
        self.ix = self.iy = 0
        self.px = self.py = 0

    def compute(self, ex, ey):
        self.ix += ex
        self.iy += ey

        dx = ex - self.px
        dy = ey - self.py

        out_x = self.kp * ex + self.ki * self.ix + self.kd * dx
        out_y = self.kp * ey + self.ki * self.iy + self.kd * dy

        self.px = ex
        self.py = ey

        return out_x, out_y


# =========================
# AUTO-LOCK TRACKER
# =========================
class AutoLockTracker:
    def __init__(
        self,
        model_path,
        video_path,
        use_kalman=True,
        use_pid=True
    ):

        self.model = YOLO(model_path)
        self.video_path = video_path

        # FLAGS
        self.use_kalman = use_kalman
        self.use_pid = use_pid

        # DISPLAY SIZE
        self.display_size = (640, 640)

        # PID
        self.pid = PID()

        # KALMAN FILTER (optional)
        if self.use_kalman:
            self.kf = cv2.KalmanFilter(4, 2)
            self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
            self.kf.transitionMatrix = np.array(
                [[1,0,1,0],
                 [0,1,0,1],
                 [0,0,1,0],
                 [0,0,0,1]], np.float32
            )
            self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # STATE MACHINE
        self.state = "SEARCH"
        self.initialized = False
        self.missed = 0
        self.lock_counter = 0

        # PARAMETERS
        self.lost_thresh = 3
        self.lock_radius = 3
        self.lock_frames = 3

    # =========================
    # DETECTION
    # =========================
    def detect(self, frame):
        res = self.model(frame, verbose=False)

        if res and res[0].boxes:
            box = res[0].boxes.xyxy.int().cpu().tolist()[0]
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            return (cx, cy)

        return None

    # =========================
    # STATE MACHINE
    # =========================
    def update_state(self, error_mag, detected):

        if detected:
            self.missed = 0
        else:
            self.missed += 1

        if self.missed > self.lost_thresh:
            self.state = "SEARCH"
            self.initialized = False
            self.lock_counter = 0
            self.pid.reset()
            return

        if detected and error_mag < self.lock_radius:
            self.lock_counter += 1
        else:
            self.lock_counter = 0

        if self.lock_counter > self.lock_frames:
            self.state = "LOCK"
        elif detected:
            self.state = "TRACK"
        else:
            self.state = "SEARCH"

    # =========================
    # MAIN LOOP
    # =========================
    def run(self):

        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # resize INPUT FOR VISUALIZATION ONLY
            frame = cv2.resize(frame, self.display_size)

            h, w = frame.shape[:2]
            cx, cy = w / 2, h / 2

            target = self.detect(frame)
            detected = target is not None

            # =========================
            # KALMAN / POSITION
            # =========================
            if self.use_kalman:

                pred = self.kf.predict()

                if detected:
                    if not self.initialized:
                        self.kf.statePre = np.array(
                            [[target[0]], [target[1]], [0], [0]],
                            np.float32
                        )
                        self.initialized = True

                    self.kf.correct(
                        np.array([[np.float32(target[0])],
                                  [np.float32(target[1])]])
                    )

                    pos = target
                else:
                    pos = (int(pred[0]), int(pred[1]))

            else:
                pos = target if detected else (int(cx), int(cy))

            # =========================
            # ERROR
            # =========================
            err_x = pos[0] - cx
            err_y = pos[1] - cy
            error_mag = abs(err_x) + abs(err_y)

            self.update_state(error_mag, detected)

            # =========================
            # CONTROL LOGIC
            # =========================
            if self.state == "SEARCH":
                cmd_x, cmd_y = 0, 0
                color = (0, 0, 255)

            elif self.state == "TRACK":
                if self.use_pid:
                    cmd_x, cmd_y = self.pid.compute(err_x, err_y)
                else:
                    cmd_x, cmd_y = err_x * 0.05, err_y * 0.05
                color = (0, 165, 255)

            elif self.state == "LOCK":
                if self.use_pid:
                    cmd_x, cmd_y = self.pid.compute(err_x * 0.3, err_y * 0.3)
                else:
                    cmd_x, cmd_y = err_x * 0.02, err_y * 0.02
                color = (0, 255, 0)

            else:
                cmd_x, cmd_y = 0, 0
                color = (255, 255, 255)

            # clamp
            cmd_x = float(np.clip(cmd_x, -25, 25))
            cmd_y = float(np.clip(cmd_y, -25, 25))

            # =========================
            # OUTPUT
            # =========================
            print(f"[{self.state}] YAW:{cmd_x:+.2f} | PITCH:{cmd_y:+.2f}")

            # =========================
            # VISUALS
            # =========================
            cv2.drawMarker(frame, pos, color, cv2.MARKER_CROSS, 18, 2)

            cv2.putText(
                frame,
                self.state,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            cv2.imshow("AUTO-LOCK 640x640", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app = AutoLockTracker(
        model_path="./runs/detect/v0/weights/best.pt",
        video_path="./data/videos_drones/10.mp4",
        use_kalman=True,   # toggle
        use_pid=True       # toggle
    )

    app.run()