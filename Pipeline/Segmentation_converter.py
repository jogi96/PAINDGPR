import os
import shutil
from ultralytics.data.converter import convert_coco


class CocoConverter():
    """
    Utility class for converting raw segmentation datasets into COCO-format
    datasets compatible with Ultralytics YOLO segmentation training.


    The class processes a folder structure containing subfolders such as
    `train`, `valid`, and `test`, converts annotations, and copies related images.
    """

    def __init__(self, raw_segmentation_path:str, save_segmentation_path:str, folders_in_annotations:list = ["train", "valid", "test"]) -> None:

        self.raw_segmentation_path = raw_segmentation_path
        self.folders = folders_in_annotations
        self.save_segmentation_path = save_segmentation_path

    def convert_split(self, labels_dir:str, save_subdir:str) -> None:
        """
        Convert a single dataset split (train/valid/test) into YOLO format
        and copy its images.


        Args:
        labels_dir (str): Path to folder containing segmentation annotations
        and images.
        save_subdir (str): Subfolder name for saving the converted data.
        """
        # Annotationen konvertieren
        convert_coco(
            labels_dir=labels_dir,
            save_dir=os.path.join(self.save_segmentation_path, save_subdir),
            use_segments=True
        )

        # Bilder kopieren
        self.copy_images(labels_dir, save_subdir)

    def copy_images(self, split_path:str, save_subdir:str) -> None:
        """
        Copy image files from a split folder into the converted dataset directory.


        Args:
        split_path (str): Directory containing images and annotations.
        save_subdir (str): Output subfolder name for converted split.
        """

        images_dst = os.path.join(self.save_segmentation_path, save_subdir, "images")
        os.makedirs(images_dst, exist_ok=True)

        for filename in os.listdir(split_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                shutil.copy2(
                    os.path.join(split_path, filename),
                    os.path.join(images_dst, filename)
                )

    def run(self) -> None:
        """
        Run the conversion for all dataset splits.


        Iterates through folders such as train, valid, and test and
        processes each via `convert_split`.
        """
        for i in self.folders:
            path = os.path.join(self.raw_segmentation_path, i)
            self.convert_split(path, i)
        
