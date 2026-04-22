from ultralytics import YOLO

model = YOLO("yolo26s.pt")

model.export(format="onnx", half=True)