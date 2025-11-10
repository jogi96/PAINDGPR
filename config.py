from pathlib import Path

BASE_DIR = Path(__file__).parent

TEST_PIC_DIR = BASE_DIR / "Data/Testdata/pictures"
TEST_FILE_DIR = BASE_DIR / "DATA/Testdata/Files"
YOLO_MODEL_SEG_DIR = BASE_DIR / "YOLO_Model_seg/Hyperbola_seg_v2/weights/best.pt"
YOLO_MODEL_OD_DIR = BASE_DIR / "YOLO_Model/best.pt"
YOLO_DATA_DIR = BASE_DIR / "YOLO_Model"
TRAIN_DIR = BASE_DIR / "Data/train/Pictures"
EDGE_DETECT_DIR = BASE_DIR / "Data/edgedetections"
PREDICT_DIR = BASE_DIR / "Data/images_for_prediction"
ANNOTATION_DIR = BASE_DIR / "Data/train/Annotations"
OUT_PATH_Detections = BASE_DIR / "Data/Detections_combined"