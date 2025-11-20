import os
import shutil
from ultralytics.data.converter import convert_coco


class CocoConverter:
    def __init__(self, raw_segmentation_path, save_segmentation_path):
        self.train_ann = os.path.join(raw_segmentation_path, "train")
        self.val_ann = os.path.join(raw_segmentation_path, "valid")

        self.save_segmentation_path = save_segmentation_path

    def convert_split(self, labels_dir, save_subdir):
        # Annotationen konvertieren
        convert_coco(
            labels_dir=labels_dir,
            save_dir=os.path.join(self.save_segmentation_path, save_subdir),
            use_segments=True
        )

        # Bilder kopieren
        self.copy_images(labels_dir, save_subdir)

    def copy_images(self, split_path, save_subdir):
        images_dst = os.path.join(self.save_segmentation_path, save_subdir, "images")
        os.makedirs(images_dst, exist_ok=True)

        for filename in os.listdir(split_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                shutil.copy2(
                    os.path.join(split_path, filename),
                    os.path.join(images_dst, filename)
                )

    def run(self):
        self.convert_split(self.train_ann, "train")
        self.convert_split(self.val_ann, "valid")
