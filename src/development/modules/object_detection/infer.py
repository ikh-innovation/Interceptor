import os
import sys
import cv2
import random
from ultralytics import YOLO

sys.path.append('./')

from src.development.modules.object_detection.config import inference_config


class YoloInferencer:
    def __init__(self, path_weights: str, path_images: str, conf: float, imgsz:int):
        self.weights = path_weights
        self.model = YOLO(self.weights)
        self.images_path = path_images
        self.conf = conf
        self.imgsz = imgsz

    def run_image_folder_inference(self):

        if not os.path.isdir(self.images_path):
            print(f"Error: Folder not found: {self.images_path}")
            return

        exts = (".jpg", ".jpeg", ".png", ".bmp")
        image_files = [
            os.path.join(self.images_path, f)
            for f in os.listdir(self.images_path)
            if f.lower().endswith(exts)
        ]

        if not image_files:
            print("No images found in folder:", self.images_path)
            return

        random.shuffle(image_files)
        print("Press 'q' to quit.")

        for img_path in image_files:

            img = cv2.imread(img_path)
            if img is None:
                print("Could not read:", img_path)
                continue

            results = self.model.predict(
                source=img,
                conf=self.conf,
                iou=0.1,
                max_det=1,
                imgsz = self.imgsz,
                verbose=False
            )

            annotated = results[0].plot()

            cv2.imshow("YOLO Image Inference", cv2.resize(annotated, (512, 512)))

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":

    path_images = "./data/dataset/test/images"

    yolo_inferencer = YoloInferencer(
        path_weights=inference_config['weights'],
        path_images=path_images,
        conf=inference_config['conf'],
        imgsz=inference_config['imgsz']
    )

    yolo_inferencer.run_image_folder_inference()