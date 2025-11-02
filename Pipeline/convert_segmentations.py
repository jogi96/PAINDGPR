from ultralytics.data.converter import convert_coco
import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *


train_ann = os.path.join(ANNOTATION_DIR, "train")
val_ann = os.path.join(ANNOTATION_DIR, "valid")

convert_coco(
    labels_dir=train_ann,
    save_dir=os.path.join(YOLO_DATA_DIR, "train"),
    use_segments=True
)

convert_coco(
    labels_dir=val_ann,
    save_dir=os.path.join(YOLO_DATA_DIR, "valid"),
    use_segments=True
)


