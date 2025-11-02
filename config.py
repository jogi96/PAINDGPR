from pathlib import Path

BASE_DIR = Path(__file__).parent

TEST_PIC_DIR = BASE_DIR / "Data/Testdata/pictures"
TEST_FILE_DIR = BASE_DIR / "DATA/Testdata/Files"
YOLO_MODEL_DIR = BASE_DIR / "YOLO_Model_seg/Hyperbola_seg_v1/weights/best.pt"
YOLO_DATA_DIR = BASE_DIR / "YOLO_Model"
TRAIN_DIR = BASE_DIR / "Data/train/Pictures"
EDGE_DETECT_DIR = BASE_DIR / "Data/edgedetections"
PREDICT_DIR = BASE_DIR / "Data/images_for_prediction"
ANNOTATION_DIR = BASE_DIR / "Data/train/Annotations"