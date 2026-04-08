import cv2
import numpy as np
from ultralytics import YOLO

class AerialInterceptorTrackerPX4:
    def __init__(self, model_path, video_path):
        # Load YOLO model
        self.model = YOLO(model_path)
        self.video_path = video_path

        # Display and Camera Settings
        self.display_size = (1024, 1024)
        self.cam_fov_horizontal = 80.0  # Degrees (adjust to your camera)
        self.cam_fov_vertical = 60.0    # Degrees (adjust to your camera)

        # Kalman Filter setup (x, y, vx, vy)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Tracking state
        self.initialized = False
        self.missed_count = 0
        self.max_reserve = 30

    def _get_yolo_center(self, frame):
        results = self.model.predict(frame, conf=0.1, verbose=False)
        if results and results[0].boxes:
            box = results[0].boxes.xyxy.int().cpu().tolist()[0]
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            return (center_x, center_y)
        return None

    def _calculate_px4_guidance(self, pos, frame_w, frame_h):
        """
        Calculates the Pitch and Yaw angles needed to center the target.
        Positive Yaw = Target is Right (Turn Right)
        Positive Pitch = Target is Below (Pitch Down)
        """
        # 1. Calculate pixel error from image center
        err_x = pos[0] - (frame_w / 2)
        err_y = pos[1] - (frame_h / 2)

        # 2. Convert pixels to degrees using FOV
        yaw_angle = (err_x / frame_w) * self.cam_fov_horizontal
        pitch_angle = (err_y / frame_h) * self.cam_fov_vertical

        return round(yaw_angle, 2), round(pitch_angle, 2)

    def process(self):
        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            h, w = frame.shape[:2]
            center = self._get_yolo_center(frame)
            prediction = self.kf.predict()

            status_color = (0, 0, 255)
            label = "SEARCHING"
            target_pos = None

            if center:
                status_color = (0, 255, 0)
                label = "LOCK"
                self.missed_count = 0

                if not self.initialized:
                    self.kf.statePre = np.array([[center[0]], [center[1]], [0], [0]], dtype=np.float32)
                    self.initialized = True

                self.kf.correct(np.array([[np.float32(center[0])], [np.float32(center[1])]]))
                target_pos = center
            else:
                self.missed_count += 1
                if self.initialized and self.missed_count < self.max_reserve:
                    label = "PREDICTING"
                    status_color = (0, 165, 255)
                    target_pos = (int(prediction[0].item()), int(prediction[1].item()))
                else:
                    self.initialized = False

            # --- PX4 GUIDANCE LOGIC ---
            if target_pos:
                # Use the Kalman position (or the arrow end) to guide the drone
                # Using target_pos here represents aiming at the PREDICTED future spot
                yaw_err, pitch_err = self._calculate_px4_guidance(target_pos, w, h)
                
                # PRINT TO TERMINAL AS REQUESTED
                print(f"[PX4 CMD] -> YAW: {yaw_err:+.2f}° | PITCH: {pitch_err:+.2f}° | MODE: {label}")
                
                self._render(frame, target_pos, status_color, label)
            else:
                print("[PX4 CMD] -> Target Lost: Holding Position")

            output = cv2.resize(frame, self.display_size)
            cv2.imshow("PX4 Interceptor Tracker", output)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

    def _render(self, frame, pos, color, label):
        cv2.drawMarker(frame, pos, color, cv2.MARKER_CROSS, 30, 2)
        vx, vy = self.kf.statePost[2].item(), self.kf.statePost[3].item()
        
        # Draw the Arrow (Future Trajectory)
        future_x, future_y = int(pos[0] + vx * 20), int(pos[1] + vy * 20)
        cv2.arrowedLine(frame, pos, (future_x, future_y), (255, 255, 0), 2)
        
        cv2.putText(frame, f"{label} V:{abs(vx)+abs(vy):.1f}", (pos[0]+15, pos[1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

if __name__ == "__main__":
    app = AerialInterceptorTrackerPX4(
        model_path="./runs/detect/v0/weights/best.pt",
        video_path="./data/videos_drones/1.mp4"
    )
    app.process()