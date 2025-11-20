from ultralytics import YOLO

# Load pretrained Model
model = YOLO("yolo11n.pt")

# Train on Custom dataset
model.train(data=r"C:\vscode\PAINDGPR\Annotated_Data\YOLO_Object_Detection\data.yaml", epochs = 200, batch= 4, project="YOLO_Model_od", name="Hyperbola_od_v1")
