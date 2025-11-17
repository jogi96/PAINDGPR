from ultralytics import YOLO

# Load pretrained Model
model = YOLO("yolov11n.pt")

# Train on Custom dataset
model.train(data=r"C:\pythonad\PAINDHS25\PAINDGPR\Annotated_Data\YOLO_Object_Detection\data.yaml", epochs = 100, batch= 4, project="YOLO_Model_od", name="Hyperbola_od_v1")
