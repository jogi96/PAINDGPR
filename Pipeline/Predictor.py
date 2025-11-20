from ultralytics import YOLO
import pandas as pd
import numpy as np
import os
import re
import json
from sklearn.linear_model import LinearRegression

class Predictor():
    def __init__(self):
        pass

    def extract_boxes_object_detection(self, results)->dict:
        all_boxes = {}

        for result in results:
            img_name = os.path.basename(result.path)
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                all_boxes[img_name] = boxes# Bounding boxes in (x1, y1, x2, y2) format
            else:
                all_boxes[img_name] = np.empty((0,4))       
        
        return all_boxes

    def match_detections(results, dist_trheshhold, out_path, export:bool = False)-> dict:
        global_boxes = []
        global_id = 0
        records = []

        def center_bbox(box):
            x1,y1,x2,y2 = box
            return ((x1+x2)/2, (y1+y2)/2)
    
        def euclidan_distance(c1, c2):
            return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
    
        def parse_filename(filename):
            base_name = os.path.basename(filename)
        
            #getting everything until SGY
            match_main = re.match(r"(.+?\.SGY)", base_name, re.IGNORECASE)
            filename_part = match_main.group(1) if match_main else base_name

            #getting cut_type and cut_number
            match_suffix = re.search(r"_(\w+)[_-](\d+)", base_name)
            cut_type = match_suffix.group(1) if match_suffix else "unknown"
            cut_number = int(match_suffix.group(2)) if match_suffix else -1

            return filename_part, cut_type, cut_number



        for result in results:
            img = os.path.basename(result.path)
            filename, cut_type, cut_number = parse_filename(img)

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
           
            else:
                boxes = np.empty((0,4)) 
                  
        
            if hasattr(result, "masks") and result.masks is not None:
                polygons = [mask.tolist() for mask in result.masks.xy]
            
            else:
                polygons = []

            for i, box in enumerate(boxes):
                center = center_bbox(box)
                existing_id = None

                for gid, gcenter in global_boxes:
                    if euclidan_distance(center, gcenter)< dist_trheshhold:
                        existing_id = gid
                        break
            
                if existing_id is None:
                    existing_id = global_id
                    global_boxes.append((global_id,center))
                    global_id +=1

                polygon_data = json.dumps(polygons[i] if i < len(polygons) else [])
                bbox_data = json.dumps(box.tolist())
            

                records.append({
                    "filename": filename,
                    "cut_type": cut_type,
                    "cut_number": cut_number,
                    "bbox_number": existing_id,
                    "bbox_data": bbox_data,
                    "polygon_data": polygon_data
                })

        df = pd.DataFrame(records)
        if export:
            df.to_csv(f"{out_path}/results.csv")
       
        return df
    
    from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import json

def fit_hyperbolas(df, csv:bool=False, csv_path:str = None):
    if csv:
        df = pd.read_csv(csv_path)
    else:
        df = df

    fits = []
    for i, row in df.iterrows():
        #Fit Hyperbolas with multiplie Regression
        polygon =  json.loads(row["polygon_data"])
        bbox = row["bbox_number"]
        polygon = np.array(polygon)
        x = polygon[:,0]
        t = polygon[:,1]
        t = t

        y = t**2
        X = np.column_stack([np.ones_like(x), x, x**2])
        model = LinearRegression(fit_intercept=False)
        model.fit(X,y)

        print(f"polygon:{i} in bbox:{bbox} , coefficients{model.coef_}")
        b0, b1, b2 = model.coef_

        #Convert Coefficients to Hyperbel context
        v = np.sqrt((4/b2))
        x0 = (-b1) / (2*b2)
        t0 = np.sqrt(b0 -b1**2 / ( 4*b2))

        fits.append({
            "filename": row["filename"],
            "cut_number": row["cut_number"],
            "bbox_number":row["bbox_number"],
            "b0_Model": b0,
            "b1_Model": b1,
            "b2_Model": b2,
            "v":v,
            "x0":x0,
            "t0":t0,
            "X_Data":x,
            "t_data":t
        })
    
    fits_df = pd.DataFrame(fits)
    return fits_df




    
    

    
    


        
       

    
    
    
    
   
    


        