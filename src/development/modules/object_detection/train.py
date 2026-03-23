import sys
sys.path.append('./')

from ultralytics import YOLO

from src.development.modules.object_detection.config import training_config

class YOLOTrainer:
    """
    YOLO trainer
    """
    def __init__(self, model: str, device: int):
        """
        Initialize model and dataset.
        """
        self.device = device
        self.model = YOLO(model)
        self.run_dir = None

    def train(self, 
              epochs: int,
              imgsz: int, 
              batch: int,
              yaml: str,
              name: str):
        """
        Train model
        """
        self.model.train(
            data=yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=name,
            device=self.device
        )

    def validate(self):
        """
        Validate the trained YOLO model and return metrics.
        """
        return self.model.val()

if __name__ == "__main__":

    params = training_config

    trainer = YOLOTrainer(params['model'], device=params['device'])

    trainer.train(
        epochs=params['epochs'],
        imgsz=params['imgsz'],
        batch=params['batch'],
        yaml=params['yaml'],
        name=params['name']
        )
    
    trainer.validate()