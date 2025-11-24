#Setting Working directory
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

#Import the config file so that only the Filename needs to be changed in the _read_segy function
from config import *

from ultralytics import YOLO

#load pretrained model
model = YOLO("yolo11n-seg.pt")

#train on custom dataset
results = model.train(data=YOLO_MODEL_SEG_H_YAML,epochs = 200,batch=4 ,project="YOLO_Model_seg", name="Hyperbola_seg")


        

