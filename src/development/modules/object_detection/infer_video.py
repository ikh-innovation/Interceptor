import os
import sys
import cv2
from ultralytics import YOLO

sys.path.append('./')

from src.development.modules.object_detection.config import inference_config

class YoloVideoInferencer:
    def __init__(self, path_weights: str, path_video: str, conf: float, imgsz:int):
        self.weights = path_weights
        self.model = YOLO(self.weights)
        self.video_path = path_video
        self.imgsz = imgsz
        self.conf = conf

    def run_video_inference(self):
        if not os.path.isfile(self.video_path):
            print(f"Error: Video not found: {self.video_path}")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video: {self.video_path}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(
                source=frame,
                conf=self.conf,
                imgsz=self.imgsz,
                iou=0.5,
                max_det=1,
                verbose=False
            )

            annotated = results[0].plot()
            cv2.imshow("YOLO Video Inference", cv2.resize(annotated, (1024, 1024)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    path_video = "./data/videos_shaheed/1.mp4"

    yolo_inferencer = YoloVideoInferencer(
        path_weights=inference_config['weights'],
        path_video=path_video,
        conf=inference_config['conf'],
        imgsz=inference_config['imgsz']
    )

    yolo_inferencer.run_video_inference()