import cv2
import numpy as np
from ultralytics import YOLO


class AerialInterceptorTracker:
    def __init__(self, model_path, video_path):
        # Load YOLO model
        self.model = YOLO(model_path)
        self.video_path = video_path

        # Display settings
        self.display_size = (512, 512)

        # Kalman Filter setup (x, y, vx, vy)
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

    def _get_yolo_center(self, frame):
        """Detect object and return its center + bounding box."""
        results = self.model(frame, verbose=False)

        if results and results[0].boxes:
            box = results[0].boxes.xyxy.int().cpu().tolist()[0]
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            return (center_x, center_y), box

        return None, None

    def process(self):
        """Main loop for video processing."""
        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            center, box = self._get_yolo_center(frame)
            prediction = self.kf.predict()

            status_color = (0, 0, 255)  # Red
            label = "LOST"

            if center:
                # Detection available
                status_color = (0, 255, 0)  # Green
                label = "TRACKING"
                self.missed_count = 0

                if not self.initialized:
                    # Initialize Kalman filter
                    self.kf.statePre = np.array(
                        [[center[0]], [center[1]], [0], [0]],
                        dtype=np.float32
                    )
                    self.initialized = True

                # Correct with measurement
                self.kf.correct(
                    np.array([[np.float32(center[0])],
                              [np.float32(center[1])]])
                )

                target_pos = center

            else:
                # No detection → rely on prediction
                self.missed_count += 1
                target_pos = (
                    int(prediction[0].item()),
                    int(prediction[1].item())
                )

                if self.missed_count < self.max_reserve:
                    label = "PREDICTING"
                    status_color = (0, 165, 255)  # Orange
                else:
                    # Reset tracking if too many misses
                    self.initialized = False

            # Render only if tracker is active
            if self.initialized:
                self._render(frame, target_pos, status_color, label)

            # Display
            output = cv2.resize(frame, self.display_size)
            cv2.imshow("Production Tracker", cv2.resize(output, (1024, 1024)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _render(self, frame, pos, color, label):
        """Draw tracking visuals."""
        # Draw position marker
        cv2.drawMarker(frame, pos, color, cv2.MARKER_CROSS, 20, 2)

        # Velocity
        vx = self.kf.statePost[2].item()
        vy = self.kf.statePost[3].item()

        # भविष्य position (visual prediction)
        future_x = int(pos[0] + vx * 20)
        future_y = int(pos[1] + vy * 20)

        # Draw direction arrow
        cv2.arrowedLine(frame, pos, (future_x, future_y),
                        (255, 255, 0), 2, tipLength=0.2)

        # Label
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
        video_path="./data/videos_shaheed/1.mp4"
    )
    app.process()