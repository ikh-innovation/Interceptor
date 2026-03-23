# ---------------- EXPERIMENT CONFIG ----------------
project_config = {
    'name': 'v0',  # change on new experiment
    'path': './data/dataset',
    'runs': './runs',
    'device': 'cuda'
}

# ---------------- TRAINING CONFIG ----------------
training_config = {
    'model': 'yolov8n.pt',
    'epochs': 200,
    'batch': 32,
    'size': 640
}

# ---------------- INFERENCE CONFIG ----------------
inference_config = {
    'weights': f"./runs/{project_config['name']}/weights/best.pt",
    'confidence': 0.03,
    'test_images': './data/dataset/test/images',
    'size': 640
}