#Setting Working directory
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

#Import the config file so that only the Filename needs to be changed in the _read_segy function
from config import *

from ultralytics import YOLO

# Load pretrained Model
model = YOLO("yolo11n.pt")

# Train on Custom dataset
results = model.train(data=YOLO_MODEL_OD_YAML, epochs = 200, batch= 4, project="YOLO_Model_od", name="Hyperbola_od_v1")
