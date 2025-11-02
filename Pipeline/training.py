from ultralytics import YOLO


model = YOLO("yolo11n-seg.pt")
results = model.train(data=r"C:\pythonad\PAINDHS25\PAINDGPR\YOLO_Model\data.yaml",epochs = 100, project="YOLO_Model_seg", name="Hyperbola_seg_v2")


        

