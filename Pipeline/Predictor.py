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
    
    def extract_boxes_and_polygons(self, results)-> dict:
        all_results = {}

        for result in results:
            img_name = os.path.basename(result.path)
            data = {}
            if result.boxes is not None:
                data["boxes"] = result.boxes.xyxy.cpu().numpy()
                data["conf"] = result.boxes.conf.cpu().numpy()
            
            else:
                data["boxes"] = np.empty((0,4)) 
                data["conf"] = np.empty((0,))      
        
            if hasattr(result, "masks") and result.masks is not None:
                print(f"found polys in :{img_name}")
                polygons = [mask.xy[0].tolist() for mask in result.masks]
                data["polygons"] = polygons
            
        
            else:
                data["polygons"] = []

            all_results[img_name] = data
        return all_results


        
       

    
    
    
    
   
    


        