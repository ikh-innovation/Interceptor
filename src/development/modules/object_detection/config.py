'''
Training config 
'''
training_config = {
    
    'model': 'yolo26n.pt',
    'device': 0,
    'epochs': 200,
    'imgsz': 640,
    'batch': 32,
    'yaml': './data/dataset/data.yaml',
    'name': 'testing',
}

'''
Inference config
'''
inference_config = {
    'weights': f"path",
    'confidence': 0.03,
    'size': 640
}