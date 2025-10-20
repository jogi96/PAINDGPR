from ultralytics import YOLO
import numpy as np
import cv2

class Predictor():
    def __init__(self, model_path='yolov8n.pt', conf=0.25, x_max=None, y_max=None, img_size=640):
        self.model = YOLO(model_path)
        self.conf = conf
        self.img_size = img_size
        self.x_max = x_max
        self.y_max = y_max

    def extract_bboxes(self, image)-> list:
        results = self.model.predict(source=image, conf=self.conf)
        bboxes = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in (x1, y1, x2, y2) format
            bboxes.extend(boxes)
        return bboxes
    
    def get_vertex(self, bboxes: np.ndarray) -> np.ndarray:
        vertices =[]
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            vertex = np.array([(x1+x2)/2,y1])
            vertices.append(vertex)
        print(vertices)
        return vertices
    
    def predict_and_scale(self,image :str):
        bboxes = self.extract_bboxes(image)
        verticels = self.get_vertex(bboxes)

        scaled_verteces = []
        for vertex in verticels:
            x,y = vertex
            x_scaled = (self.x_max - 0) /640
            y_scaled = (self.y_max - 0) /640
            x = x * x_scaled
            y = y * y_scaled
            scaled_verteces.append((x,y))
        
        return scaled_verteces ,verticels, bboxes
    
    
    def plot_detected_points(self,image:str, verticels: list):
        img = cv2.imread(str(image))
    
        for vertex in verticels:
            x,y = vertex
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.imshow('Scaled Points', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    


        