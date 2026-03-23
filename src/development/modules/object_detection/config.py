'''
Training config 
'''
training_config = {
    
    'model': 'yolo26s.pt',
    'device': 0,
    'epochs': 300,
    'imgsz': 640,
    'batch': 32,
    'yaml': './data/dataset/data.yaml',
    'name': 'yolov26s_v0',
}

'''
Inference config
'''
inference_config = {
    'weights': f"./runs/detect/{training_config['name']}/weights/best.pt",
    'conf': 0.5,
    'imgsz': 640,
}