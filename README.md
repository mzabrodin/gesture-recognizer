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

1. **Hand Detection & Landmark Extraction:** Using MediaPipe's Hand Landmarker to identify 21 3D hand landmarks.
2. **Preprocessing:** Normalizing and scaling landmark coordinates to be scale and position invariant (relative to
   wrist).
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
   git clone <repository-url>
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

## Scripts

- `scripts/test_mediapipe.py`: A utility script to verify MediaPipe landmark extraction on your camera.

## License

This project is licensed under the **MIT License**. See [LICENSE.md](LICENSE.md) for details.
