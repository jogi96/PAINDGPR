from ultralytics import YOLO
import numpy as np
import os

class Predictor():
    def __init__(self, model_path:str, prediction_folder:str):
        self.model_path = model_path
        self.prediction_folder = prediction_folder
        self.model = YOLO(model_path)
        pass

    def predict_location_hyperbolas(self, save:bool = False, conf:float = 0.5):
        return self.model.predict(source=self.prediction_folder, save = save, conf=conf)
    
    def extract_boxes(self, results)->dict:
        all_boxes = {}

        for result in results:
            img_name = os.path.basename(result.path)
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                all_boxes[img_name] = boxes# Bounding boxes in (x1, y1, x2, y2) format
            else:
                all_boxes[img_name] = np.empty((0,4))       
        
        return all_boxes


        
       

    
    
    
    
   
    


        