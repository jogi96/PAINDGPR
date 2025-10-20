import numpy as np
from .Predictor import Predictor

class Hyperbolas():
    def __init__(self, model_path:str, image:str,  conf:float, x_max:int, y_max:int):
        self.predictor = Predictor(model_path=model_path, conf=conf, x_max=x_max, y_max=y_max)
        self.image = image
    
    def fit_hyperbola2d(self, v:float, z= 0):
        scaled_verteces ,verticels, bboxes = self.predictor.predict_and_scale(image=self.image)
        hyperbolas = []
        for vertex in verticels:
            x0, y0 = vertex
            z = z  # Assuming z=0 for 2D points
            for bbox in bboxes:
                x1 ,y1, x2, y2 = bbox
                x_vals = np.linspace(x1,x2,50)
                T_pred = (2.0 / v) * np.sqrt(((x1+x2) - x0)**2 + z**2)
                hyperbolas.append(T_pred)
        return T_pred


    def fit_hyperbola3d(self):
        pass

    def plot_hyperbola2d(self):
        pass
        