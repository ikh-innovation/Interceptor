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

    def train(self, 
              epochs: int,
              imgsz: int, 
              batch: int,
              yaml: str,
              name: str,
              lr0: float = 0.01,
              lrf: float = 0.01):
        """
        Train model using fixed lr0 and lrf.
        """
        print(f"Training with lr0={lr0}, lrf={lrf}")
        self.model.train(
            data=yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=name,
            device=self.device,
            lr0=lr0,
            lrf=lrf,
            augment=True,
            fliplr=0.5,    
            flipud=0.1,    
            mosaic=1.0,    
            mixup=0.1,     
            copy_paste=0.1 
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
        name=params['name'],
        lr0=params.get('lr0', 0.01),
        lrf=params.get('lrf', 0.01)
    )

    metrics = trainer.validate()
    print(metrics)