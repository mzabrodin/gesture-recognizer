import os
import time
from urllib import request

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

CURRENT_RESULT = None

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 17),  # wrist
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
]


def ensure_model_exists():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if not os.path.exists(MODEL_PATH):
        try:
            request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    else:
        pass


def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print('hand landmarker result: {}'.format(result))
    global CURRENT_RESULT
    CURRENT_RESULT = result


def draw_landmarks(image, result):
    if result is None or not result.hand_landmarks:
        return image

    annotated_image = image.copy()
    height, width, _ = image.shape

    for hand_landmarks in result.hand_landmarks:
        points = []

        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]

            start_point = points[start_idx]
            end_point = points[end_idx]

            cv2.line(annotated_image, start_point, end_point, (255, 255, 255), 2)

        for point in points:
            cv2.circle(annotated_image, point, 5, (0, 255, 0), -1)

    return annotated_image


def main():
    ensure_model_exists()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with HandLandmarker.create_from_options(options) as landmarker:
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            timestamp_ms = int((time.time() - start_time) * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            if CURRENT_RESULT:
                frame = draw_landmarks(frame, CURRENT_RESULT)

            cv2.imshow("hand_landmarker_test", frame)

            if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
