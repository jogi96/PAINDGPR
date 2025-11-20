from ultralytics import YOLO


model = YOLO("yolo11n-seg.pt")
results = model.train(data=r"C:\vscode\PAINDGPR\Annotated_Data\YOLO_Segmentation_Hyperbola\data.yaml",epochs = 200,batch=4 ,project="YOLO_Model_seg", name="Hyperbola_seg")


        

