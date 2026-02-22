import os
import csv
from urllib import request

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

import cv2
import mediapipe as mp
from tqdm import tqdm

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'models'))
MODEL_PATH = os.path.join(MODEL_DIR, 'hand_landmarker.task')
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "raw", "train"))
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "processed"))
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "landmarks.csv")


def ensure_model_exists():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if not os.path.exists(MODEL_PATH):
        try:
            request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise


def main():
    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return

    if os.path.exists(OUTPUT_CSV):
        print(f"Warning: File {OUTPUT_CSV} already exists and will be overwritten.")

    ensure_model_exists()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )

    total_processed = 0

    with HandLandmarker.create_from_options(options) as landmarker, open(OUTPUT_CSV, mode='w', newline='',
                                                                         encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['label_name', 'label_id'] + [f'coord_{i}' for i in range(42)]
        writer.writerow(header)

        for class_id, class_name in enumerate(classes):
            class_dir = os.path.join(DATA_DIR, class_name)

            files = [
                file_name for file_name in sorted(os.listdir(class_dir))
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
            ]

            valid_images = 0
            unreadable_files = 0
            no_hand_detected = 0

            batch_rows = []

            for file_name in tqdm(files, desc=f"Processing {class_name}"):
                img_path = os.path.join(class_dir, file_name)
                image = cv2.imread(img_path)

                if image is None:
                    unreadable_files += 1
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                result = landmarker.detect(mp_image)

                if not result.hand_landmarks:
                    no_hand_detected += 1
                    continue

                landmarks = result.hand_landmarks[0]

                base_x = landmarks[0].x
                base_y = landmarks[0].y

                translated_coords = []
                for landmark in landmarks:
                    translated_coords.append((landmark.x - base_x, landmark.y - base_y))

                flat_abs_coords = [abs(val) for pair in translated_coords for val in pair]
                max_value = max(flat_abs_coords) if flat_abs_coords else 1.0

                if max_value == 0:
                    max_value = 1.0

                row = [class_name, str(class_id)]
                for x, y in translated_coords:
                    norm_x = x / max_value
                    norm_y = y / max_value
                    row.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

                batch_rows.append(row)
                valid_images += 1

            if batch_rows:
                writer.writerows(batch_rows)

            total_processed += valid_images

            print(
                f"[{class_name}] Success: {valid_images} | No hands detected: {no_hand_detected} | Read errors: {unreadable_files}\n")

    print(f"Processing complete. Total samples saved: {total_processed}")


if __name__ == "__main__":
    main()
