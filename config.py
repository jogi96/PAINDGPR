from pathlib import Path

BASE_DIR = Path(__file__).parent

TEST_PIC_DIR = BASE_DIR / "Data/Testdata/pictures"
TEST_FILE_DIR = BASE_DIR / "DATA/Testdata/Files"
YOLO_MODEL_DIR = BASE_DIR / "YoloModel/runs/detect/yolo11n-painD-test/weights/best.pt"