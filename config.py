from pathlib import Path

BASE_DIR = Path(__file__).parent

TEST_PIC_SEG_H_DIR = BASE_DIR / "Data/Testdata/Segmentation_Hyperbola/pictures"
TEST_PIC_SEG_RB_DIR = BASE_DIR / "Data/Testdata/Segementation_Reinforcement_Bars/pictures"
TEST_PIC_OD_H_DIR = BASE_DIR / "Data/Testdata/Object_Detection_Hyperbola/pictures"
TEST_FILE_DIR = BASE_DIR / "DATA/Testdata/Files"
YOLO_MODEL_SEG_H_DIR = BASE_DIR / "YOLO_Model_seg/Hyperbola_seg/weights/best.pt"
YOLO_MODEL_OD_DIR = BASE_DIR / "YOLO_Model_od/Hyperbola_od_v13/weights/best.pt"
YOLO_MODEL_SEG_RB_DIR = BASE_DIR /"YOLO_Model_rb_seg/Reinforcement_Bars_seg/weights/best.pt"
YOLO_MODEL_OD_YAML = BASE_DIR / "Annotated_Data/YOLO_Object_Detection/data.yaml"
YOLO_MODEL_SEG_H_YAML = BASE_DIR / "Annotated_Data/YOLO_Segmentation_Hyperbola/data.yaml"
YOLO_MODEL_SEG_RB_YAML = BASE_DIR / "Annotated_Data/YOLO_Segmentation_reinforcing_bars/data.yaml"
PICTURES_FOR_ANNOTATION = BASE_DIR / "Data/Data_for_Models/Pictures"
EDGE_DETECT_DIR = BASE_DIR / "Data/edgedetections"
PREDICT_DIR = BASE_DIR / "Data/images_for_prediction"
ANNOTATION_SEG_H_DIR = BASE_DIR / "Annotated_Data/YOLO_Segmentation_Hyperbola"
ANNOTATION_SEG_RB_DIR = BASE_DIR / "Annotated_Data/YOLO_Segmentation_reinforcing_bars"
ANNOTATION_RAW_SEG_RB_DIR = BASE_DIR / "Annotated_Data/SEG_RAW/YOLO_Segmentation_reinforcing_bars"
ANNOTATION_RAW_SEG_H_DIR = BASE_DIR / "Annotated_Data/SEG_RAW/YOLO_Segmentation_Hyperbola"
OUT_PATH_Detections = BASE_DIR / "Data/Detections_combined"