# Gesture Recognizer

A real-time hand gesture recognition system using MediaPipe for landmark extraction and a TensorFlow-Lite-based
classifier.
This project features a complete pipeline for dataset acquisition, landmark processing, model training, and a
PySide6-based desktop application for real-time inference and action triggering.

## General Overview

The Gesture Recognizer identifies hand gestures from a webcam stream to trigger system-level actions via `pyautogui`.
Key features include:

- **Action Triggering:** Map gestures to actions like taking screenshots, adjusting volume, or media control.
- **Real-Time Feedback:** GUI showing camera feed, detected landmarks, and classification results.
- **Customizable:** Settings for confidence thresholds, active gestures, and camera selection.

The system utilizes a multi-stage approach:

1. **Hand Detection & Landmark Extraction:** Using MediaPipe's Hand Landmarker to identify 21 3D hand
   landmarks (normalized X, Y, and Z coordinates).
2. **Preprocessing:** Normalizing and scaling landmarks to be scale and position invariant (relative to wrist).
   Currently, the classifier utilizes the 2D (X, Y) projection of these landmarks.
3. **Classification:** A deep neural network (TFLite) classifies the processed landmarks into gesture categories.

## Technical Stack

### Core Technologies

- **Language:** Python 3.11.14
- **Package Manager:** [uv](https://github.com/astral-sh/uv) (for fast, reproducible dependency management)
- **GUI Framework:** PySide6 (Qt for Python)
- **Inference:** AI Edge LiteRT (TensorFlow Lite)

### Key Libraries

- **Computer Vision:** `mediapipe` (landmark extraction), `opencv-python` (camera interface)
- **Machine Learning:** `tensorflow` (training), `scikit-learn` (scaling)
- **Automation:** `pyautogui` (system actions), `screen-brightness-control`
- **Utilities:** `huggingface-hub`, `python-dotenv`, `platformdirs`

## Project Structure

```text
gesture-recognizer/
├── data/               # Raw images and processed CSVs (ignored by git)
├── logs/               # Training history and TensorBoard logs
├── models/             # TFLite models, labels, and scalers
│   ├── gesture_classifier.tflite
│   ├── labels.txt
│   └── scaler_params.json
├── scripts/            # Pipeline utility notebooks and tools
│   ├── download_dataset.ipynb
│   ├── process_dataset.ipynb
│   ├── test_mediapipe.py   # MediaPipe verification tool
│   ├── train_model.ipynb
│   └── visualize_dataset.ipynb
├── src/                # Application source code
│   ├── actions.py          # Gesture-to-action mapping
│   ├── camera_thread.py    # Async camera capture
│   ├── config.py           # Constants and paths
│   ├── gesture_handler.py  # Logic for triggering actions
│   ├── inference.py        # TFLite inference wrapper
│   ├── inference_thread.py # Async inference execution
│   ├── main_window.py      # PySide6 GUI
│   └── settings_manager.py # JSON settings persistence
├── main.py             # Application entry point
├── pyproject.toml      # Project configuration and dependencies
└── uv.lock             # Dependency lockfile
```

## Setup

### Requirements

- **Python:** 3.11.14 (strict versioning via `pyproject.toml`)
- **Webcam:** A working camera for real-time recognition.
- **Package Manager:** `uv` is recommended.

### Installation

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/mzabrodin/gesture-recognizer.git
   cd gesture-recognizer
   ```

2. **Install dependencies:**
   Using `uv` (recommended):
   ```powershell
   uv sync
   ```
   Or using `pip`:
   ```powershell
   pip install .
   ```

3. **Environment Variables:**
   Create a `.env` file in the root directory for dataset acquisition:
   ```env
   HF_TOKEN=your_huggingface_token_here
   ```

## Usage

### Running the Application

To start the real-time recognition GUI:

```powershell
uv run main.py
```

- **Features:** Toggle gestures, adjust confidence, and see live landmark overlays.
- **Exit:** Close the window or press **ESC** (if implemented in the GUI).

### Creating a Custom Model

The project uses a subset of the **HaGrid** dataset. The pipeline is documented in Jupyter notebooks within the
`scripts/` folder:

1. **Download Data:** `scripts/download_dataset.ipynb`
    - Downloads images from Hugging Face.
2. **Process Landmarks:** `scripts/process_dataset.ipynb`
    - Extracts landmarks using MediaPipe and saves to `data/processed/landmarks.csv`.
3. **Train Model:** `scripts/train_model.ipynb`
    - Trains the Keras model, scales data, and exports to TFLite.
4. **Visualize:** `scripts/visualize_dataset.ipynb`
    - Explore the processed landmark distributions.

> **Note:** To run notebooks with `uv`, use `uv run jupyter lab` or configure your IDE to use the `.venv` created by
`uv`.

## Dataset Processing Detail

The dataset construction pipeline involves acquiring raw image data and converting it into a structured coordinate-based format.

### Data Acquisition
The raw images are sourced from a subset of the **HaGrid (Hand Gesture Recognition Dataset)** hosted on Hugging Face (`neilrigaud/hagrid-subset`).
- **Supported Classes:** `call`, `dislike`, `fist`, `four`, `grabbing`, `grip`, `like`, `middle_finger`, `mute`, `no_gesture`, `ok`, `one`, `palm`, `peace`, `peace_inverted`, `rock`, `stop`, `stop_inverted`, `three`, `three2`, `three3`, `two_up`, `two_up_inverted`.
- **Download logic:** `scripts/download_dataset.ipynb` uses the `huggingface_hub` API to fetch images for each gesture class.
- **Constraints:** A limit (e.g., 2500 images per class) is applied to maintain a manageable dataset size, with downloads performed concurrently for efficiency.
- **Organization:** Images are saved into `data/raw/train/<class_name>/`.

### Landmark Extraction & Filtering
The transformation from pixels to coordinates is performed in `scripts/process_dataset.ipynb` using MediaPipe's `HandLandmarker`.
- **Validation:** Every image is checked for readability. Images where MediaPipe fails to detect a hand (with a default confidence threshold of 0.5) are discarded.
- **Feature Extraction:** For each valid image, 21 landmarks are extracted. Each landmark consists of normalized $x$ and $y$ coordinates relative to the image dimensions.
- **Persistence:** The resulting dataset is saved as a single CSV file (`data/processed/landmarks.csv`) containing:
    - `label_name` and `label_id` (integer mapping).
    - 42 feature columns (`coord_0` to `coord_41`).

## Model Training Detail

The classifier is a multi-layer perceptron (MLP) trained on coordinate features extracted from hand landmarks.

### Data Preprocessing
To ensure the model is invariant to hand position and scale, landmarks undergo the following transformation in `process_dataset.ipynb`:
1. **Translation:** All 21 landmarks are translated so the wrist (landmark 0) is at $(0, 0)$.
2. **Scaling:** Coordinates are normalized by dividing by the maximum absolute coordinate value among all landmarks in that hand.
3. **Dimensionality:** Currently, only the 2D projection ($x, y$) is used, resulting in a 42-element feature vector ($21 \text{ points} \times 2 \text{ coordinates}$).
4. **Standardization:** A `StandardScaler` is applied to the final dataset to ensure zero mean and unit variance, with parameters saved to `models/scaler_params.json`.

### Architecture
The model is implemented using Keras and consists of:
- **Input Layer:** 42 neurons (21 pairs of $x, y$ coordinates).
- **Hidden Layer 1:** 128 neurons, ReLU activation, L2 regularization, Batch Normalization, and 20% Dropout.
- **Hidden Layer 2:** 64 neurons, ReLU activation, L2 regularization, Batch Normalization, and 20% Dropout.
- **Hidden Layer 3:** 32 neurons, ReLU activation, and 20% Dropout.
- **Output Layer:** Softmax activation with neurons equal to the number of gesture classes.

### Training Parameters
- **Optimizer:** Adam (learning rate = 0.001)
- **Loss Function:** Sparse Categorical Crossentropy
- **Batch Size:** 32
- **Epochs:** Up to 100 (with Early Stopping)
- **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard.

### Export
The final model is converted to a quantized **TensorFlow Lite (.tflite)** format for low-latency inference in the desktop application.

## Scripts

- `scripts/test_mediapipe.py`: A utility script to verify MediaPipe landmark extraction on your camera.

## License

This project is licensed under the **MIT License**. See [LICENSE.md](LICENSE.md) for details.
