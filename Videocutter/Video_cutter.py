import cv2
import os

class Video_Cutter:
    def __init__(self, video_path, output_path, target_fps=None):
        self.video_path = video_path
        self.output_path = output_path
        self.target_fps = target_fps
        os.makedirs(self.output_path, exist_ok=True)

    def cut_video(self):
        cap = cv2.VideoCapture(self.video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Original video FPS: {original_fps}")

        # Calculate frame skip if target_fps is set
        frame_skip = 1
        if self.target_fps and self.target_fps < original_fps:
            frame_skip = int(round(original_fps / self.target_fps))
            print(f"Extracting every {frame_skip}th frame (~{self.target_fps} FPS)")

        frame_idx = 0
        saved_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame if it matches the frame skip criteria
            if frame_idx % frame_skip == 0:
                filename = os.path.join(self.output_path, f"frame_{saved_idx:05d}.jpg")
                cv2.imwrite(filename, frame)
                saved_idx += 1

            frame_idx += 1

        cap.release()
        print(f"{saved_idx} Frames saved in '{self.output_path}'")

