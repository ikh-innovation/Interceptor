import cv2
import numpy as np
from ultralytics import YOLO


class AerialInterceptorTracker:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.video_path = video_path

        self.display_size = (512, 512)

        # Confidence thresholds
        self.wide_conf = 0.1
        self.narrow_conf = 0.05

        # FIXED narrow camera size (NxN)
        self.narrow_size = 200

        # ---------------- Kalman Filter ----------------
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )

        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Tracking state
        self.initialized = False
        self.missed_count = 0
        self.max_reserve = 30

    # ---------------- WIDE CAMERA (FULL FRAME) ----------------
    def _detect_wide(self, frame):
        results = self.model(frame, conf=self.wide_conf, verbose=False)

        if results and results[0].boxes:
            box = results[0].boxes.xyxy.cpu().numpy()[0]

            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)

            return (cx, cy), box

        return None, None

    # ---------------- NARROW CAMERA (CENTER NxN + SECOND INFERENCE) ----------------
    def _detect_narrow(self, frame):
        h, w = frame.shape[:2]

        cx, cy = w // 2, h // 2
        half = self.narrow_size // 2

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        # FIXED CENTER CROP (real narrow camera)
        narrow_frame = frame[y1:y2, x1:x2]

        # SECOND YOLO INFERENCE (IMPORTANT)
        results = self.model(narrow_frame, conf=self.narrow_conf, verbose=False)

        if results and results[0].boxes:
            box = results[0].boxes.xyxy.cpu().numpy()[0]

            # map narrow coords → full frame coords
            box = [
                box[0] + x1,
                box[1] + y1,
                box[2] + x1,
                box[3] + y1
            ]

            cx_obj = int((box[0] + box[2]) / 2)
            cy_obj = int((box[1] + box[3]) / 2)

            return (cx_obj, cy_obj), box, (x1, y1, x2, y2)

        return None, None, (x1, y1, x2, y2)

    # ---------------- MAIN LOOP ----------------
    def process(self):
        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ---------------- KALMAN PREDICTION ----------------
            prediction = self.kf.predict()
            pred_pos = (int(prediction[0]), int(prediction[1]))

            # ---------------- TWO YOLO INFERENCES ----------------
            wide_center, wide_box = self._detect_wide(frame)
            narrow_center, narrow_box, roi_box = self._detect_narrow(frame)

            # ---------------- FUSION LOGIC ----------------
            center = None
            box = None
            source = "NONE"

            # Narrow has priority (high precision center lock)
            if narrow_center:
                center = narrow_center
                box = narrow_box
                source = "NARROW"
            elif wide_center:
                center = wide_center
                box = wide_box
                source = "WIDE"

            # ---------------- STATUS ----------------
            status_color = (0, 0, 255)
            label = "LOST"

            if center:
                status_color = (0, 255, 0)
                label = "TRACKING"
                self.missed_count = 0

                if not self.initialized:
                    self.kf.statePre = np.array(
                        [[center[0]], [center[1]], [0], [0]],
                        dtype=np.float32
                    )
                    self.initialized = True

                self.kf.correct(
                    np.array([[np.float32(center[0])],
                              [np.float32(center[1])]])
                )

                target_pos = center

            else:
                self.missed_count += 1
                target_pos = pred_pos

                if self.missed_count < self.max_reserve:
                    label = "PREDICTING"
                    status_color = (0, 165, 255)
                else:
                    self.initialized = False

            label = f"{label} | {source}"

            # ---------------- RENDER ----------------
            if self.initialized:
                self._render(frame, target_pos, status_color, label)

            # Draw FIXED narrow NxN box
            x1, y1, x2, y2 = roi_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # ---------------- DISPLAY ----------------
            output = cv2.resize(frame, self.display_size)
            cv2.imshow("Dual YOLO (Wide + Narrow NxN)", cv2.resize(output, (1024, 1024)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # ---------------- VISUALIZATION ----------------
    def _render(self, frame, pos, color, label):
        cv2.drawMarker(frame, pos, color, cv2.MARKER_CROSS, 20, 2)

        vx = self.kf.statePost[2].item()
        vy = self.kf.statePost[3].item()

        future_x = int(pos[0] + vx * 20)
        future_y = int(pos[1] + vy * 20)

        cv2.arrowedLine(frame, pos, (future_x, future_y),
                        (255, 255, 0), 2, tipLength=0.2)

        speed = abs(vx) + abs(vy)

        cv2.putText(
            frame,
            f"{label} | V:{speed:.1f}",
            (pos[0] + 15, pos[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )


if __name__ == "__main__":
    app = AerialInterceptorTracker(
        model_path="./runs/detect/v0/weights/best.pt",
        video_path="./data/videos_drones/10.mp4"
    )
    app.process()