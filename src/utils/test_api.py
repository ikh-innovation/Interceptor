import sys
import cv2
import random
import tempfile
from inference_sdk import InferenceHTTPClient

sys.path.append('./')

class APITester:
    
    def __init__(self, url, key, input, thresh):
        '''
        Roboflow api test on input image
        '''
        self.url = url
        self.key = key
        self.input = input
        self.thresh = thresh
        pass
    
    def _draw_preds(self, frame, result):
        '''
        Draw bbox on image based on api result
        '''
        for pred in result.get("predictions", []):
            conf = pred["confidence"]
            if conf < self.thresh:
                continue 

            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return frame
        
    def run(self):
        '''
        Actual inference & drawing of prediction on 
        randomly picked image from input video
        '''
        client = InferenceHTTPClient(api_url=self.url, 
                                     api_key=self.key)
        
        cap = cv2.VideoCapture(self.input)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            frame_idx = random.randint(0, frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue  

            with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
                cv2.imwrite(tmp.name, frame)
                result = client.infer(tmp.name, model_id="lrddv2-fhowg-cegrh/4")
                
            frame = self._draw_preds(frame, result)
            
            cv2.imshow("Random LRDDv2 Detection", cv2.resize(frame, (512,1024)))

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    
    url="https://serverless.roboflow.com"
    key="E4P1asOlS3HJ2EwCHUOy"
    
    path = "./data/videos_shaheed/1.mp4"
    
    tester = APITester(url, key, path, 0.5)
    tester.run()
    