import os
import time
import json
import threading
from typing import Optional, Tuple, List

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

import numpy as np
import cv2
import mediapipe as mp
from ai_edge_litert.interpreter import Interpreter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')
MP_MODEL_PATH = os.path.join(MODEL_DIR, 'hand_landmarker.task')
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, 'gesture_classifier.tflite')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.txt')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_params.json')

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

state_lock = threading.Lock()
CURRENT_RESULT: Optional[mp.tasks.vision.HandLandmarkerResult] = None

SMOOTHED_PROBS = None
EMA_ALPHA = 0.4

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # index
    (5, 9), (9, 13), (13, 17), (0, 17),  # palm base
    (9, 10), (10, 11), (11, 12),  # middle
    (13, 14), (14, 15), (15, 16),  # ring
    (17, 18), (18, 19), (19, 20)  # pinky
]


def load_inference_assets() -> Tuple[List[str], np.ndarray, np.ndarray, Interpreter, List, List]:
    """Load TFLite model, labels, and scaler parameters for inference."""
    if not os.path.exists(TFLITE_MODEL_PATH):
        raise FileNotFoundError(f"TFLite model not found: {TFLITE_MODEL_PATH}")

    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]

    with open(SCALER_PATH, 'r') as f:
        scaler_data = json.load(f)
        scaler_mean = np.array(scaler_data['mean'], dtype=np.float32)
        scaler_scale = np.array(scaler_data['scale'], dtype=np.float32)

    interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return labels, scaler_mean, scaler_scale, interpreter, input_details, output_details


def process_landmarks(
        landmarks: List,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        interpreter: Interpreter,
        input_details: List,
        output_details: List,
        labels: List[str]
) -> Tuple[str, float]:
    """Process hand landmarks, apply EMA smoothing, and return predicted gesture label with confidence."""
    global SMOOTHED_PROBS

    base_x, base_y = landmarks[0].x, landmarks[0].y
    translated = [(lm.x - base_x, lm.y - base_y) for lm in landmarks]

    flat_abs = [abs(val) for pair in translated for val in pair]
    max_val = max(flat_abs) if flat_abs else 1.0
    if max_val == 0:
        max_val = 1.0

    normalized_coords = []
    for x, y in translated:
        normalized_coords.extend([x / max_val, y / max_val])

    features = np.array([normalized_coords], dtype=np.float32)

    scaled_features = (features - scaler_mean) / scaler_scale

    interpreter.set_tensor(input_details[0]['index'], scaled_features)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    if SMOOTHED_PROBS is None:
        SMOOTHED_PROBS = predictions
    else:
        SMOOTHED_PROBS = (EMA_ALPHA * predictions) + ((1 - EMA_ALPHA) * SMOOTHED_PROBS)

    top_index = np.argmax(SMOOTHED_PROBS)
    return labels[top_index], float(SMOOTHED_PROBS[top_index])


def print_result(result: Optional[mp.tasks.vision.HandLandmarkerResult], output_image: mp.Image,
                 timestamp_ms: int) -> None:
    """Callback invoked by MediaPipe in a background thread with hand landmark detection results."""
    global CURRENT_RESULT
    with state_lock:
        CURRENT_RESULT = result


def draw_overlays(
        image: np.ndarray,
        result: Optional[mp.tasks.vision.HandLandmarkerResult],
        prediction: str,
        confidence: float,
        fps: float
) -> np.ndarray:
    """Draw prediction results and hand landmarks on the image."""
    annotated_image = image.copy()
    height, width, _ = image.shape

    if prediction == "No Hand Detected" or prediction == "Waiting":
        text = prediction
        color = (0, 0, 255)  # Red
    elif confidence < 0.65:
        text = f"? {prediction} ({confidence * 100:.1f}%)"
        color = (255, 0, 0)  # Blue
    else:
        text = f"{prediction} ({confidence * 100:.1f}%)"
        color = (0, 255, 0)  # Green

    cv2.putText(annotated_image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    fps_text = f"FPS: {int(fps)}"
    cv2.putText(annotated_image, fps_text, (width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    if result and result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            points = []
            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append((x, y))

            for connection in HAND_CONNECTIONS:
                start_point = points[connection[0]]
                end_point = points[connection[1]]
                cv2.line(annotated_image, start_point, end_point, (255, 255, 255), 2)

            for point in points:
                cv2.circle(annotated_image, point, 5, (0, 0, 0), -1)

    return annotated_image


def main() -> None:
    """Run real-time gesture recognition using webcam input."""
    print("Loading AI models and assets")
    labels, scaler_mean, scaler_scale, interpreter, input_details, output_details = load_inference_assets()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting webcam. Press 'ESC' to exit.")

    prev_time = time.time()

    with HandLandmarker.create_from_options(options) as landmarker:
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            timestamp_ms = int((time.time() - start_time) * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            local_result = None
            with state_lock:
                local_result = CURRENT_RESULT

            if local_result is not None and local_result.hand_landmarks:
                prediction, confidence = process_landmarks(
                    local_result.hand_landmarks[0],
                    scaler_mean,
                    scaler_scale,
                    interpreter,
                    input_details,
                    output_details,
                    labels
                )
            else:
                prediction = "No Hand Detected"
                confidence = 0.0

            curr_time = time.time()
            time_diff = curr_time - prev_time
            fps = 1.0 / time_diff if time_diff > 0 else 0.0
            prev_time = curr_time

            frame = draw_overlays(frame, local_result, prediction, confidence, fps)

            cv2.imshow('Real-Time Gesture Recognition', frame)

            if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed gracefully.")


if __name__ == "__main__":
    main()
