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
    def __init__(self, model_path, video_path):

        self.model = YOLO(model_path)
        self.video_path = video_path

        # Kalman (x, y, vx, vy)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1,0,1,0],
             [0,1,0,1],
             [0,0,1,0],
             [0,0,0,1]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        self.pid = PID()

        # State machine
        self.state = "SEARCH"

        self.initialized = False
        self.missed = 0

        # thresholds
        self.lost_thresh = 11
        self.lock_radius = 5   # pixels
        self.lock_frames = 3   # stability requirement

        self.lock_counter = 0

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

        if not detected:
            self.missed += 1
        else:
            self.missed = 0

        # LOST → SEARCH
        if self.missed > self.lost_thresh:
            self.state = "SEARCH"
            self.initialized = False
            self.lock_counter = 0
            self.pid.reset()
            return

        # LOCK CONDITION
        if error_mag < self.lock_radius and detected:
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

            h, w = frame.shape[:2]
            cx, cy = w / 2, h / 2

            target = self.detect(frame)
            pred = self.kf.predict()

            detected = target is not None

            # choose position
            if detected:
                pos = target

                if not self.initialized:
                    self.kf.statePre = np.array([[pos[0]],[pos[1]],[0],[0]], np.float32)
                    self.initialized = True

                self.kf.correct(np.array([[np.float32(pos[0])],[np.float32(pos[1])]]))

            else:
                pos = (int(pred[0]), int(pred[1]))

            # error
            err_x = pos[0] - cx
            err_y = pos[1] - cy
            error_mag = abs(err_x) + abs(err_y)

            # update state machine
            self.update_state(error_mag, detected)

            # =========================
            # CONTROL LOGIC
            # =========================
            if self.state == "SEARCH":
                cmd_x, cmd_y = 0, 0
                color = (0, 0, 255)

            elif self.state == "TRACK":
                cmd_x, cmd_y = self.pid.compute(err_x, err_y)
                color = (0, 165, 255)

            elif self.state == "LOCK":
                cmd_x, cmd_y = self.pid.compute(err_x * 0.3, err_y * 0.3)
                color = (0, 255, 0)

            else:
                cmd_x, cmd_y = 0, 0
                color = (255, 255, 255)

            # clamp output
            cmd_x = float(np.clip(cmd_x, -25, 25))
            cmd_y = float(np.clip(cmd_y, -25, 25))

            # =========================
            # OUTPUT (simulate PX4)
            # =========================
            print(f"[{self.state}] YAW:{cmd_x:+.2f} | PITCH:{cmd_y:+.2f}")

            # =========================
            # VISUALS
            # =========================
            cv2.drawMarker(frame, pos, color, cv2.MARKER_CROSS, 20, 2)
            cv2.putText(frame, self.state, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("AUTO LOCK TRACKER", frame)

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
        video_path="./data/videos_drones/2.mp4"
    )
    app.run()