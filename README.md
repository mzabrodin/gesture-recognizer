# Gesture Recognizer

A real-time hand gesture recognition system using MediaPipe for landmark extraction and a TensorFlow-based classifier.
This project was developed as part of the DL4CV course.

> **Status:** Work in Progress. Current version focuses on dataset processing, landmark extraction, and model training.
> Real-time inference is functional but undergoing refinement.

## General Overview

The Gesture Recognizer is designed to identify specific hand gestures from a webcam stream to trigger system-level
actions. For example, specific gestures can be used for:

- **Taking a screenshot**
- **Adjusting volume**
- **Other custom shortcuts**

The system utilizes a multi-stage approach:

1. **Hand Detection & Landmark Extraction:** Using MediaPipe's Hand Landmarker to identify 21 3D hand landmarks.
2. **Preprocessing:** Normalizing and scaling landmark coordinates to be scale and position invariant.
3. **Classification:** A deep neural network (TFLite) classifies the processed landmarks into gesture categories.

## Technical Stack & Libraries

### Core Technologies

- **Language:** Python 3.11.14
- **Package Manager:** [uv](https://github.com/astral-sh/uv) (for fast, reproducible dependency management)

### Libraries

- **Computer Vision:** `mediapipe` (landmark extraction), `opencv-python` (image processing and camera interface)
- **Machine Learning:** `tensorflow` (model training and TFLite conversion), `scikit-learn` (data scaling and
  evaluation)
- **Data Handling:** `pandas`, `numpy`, `tqdm`
- **Utilities:** `huggingface-hub` (dataset acquisition), `python-dotenv`

## Creating a Model

The process of building the gesture recognition model follows a strictly defined pipeline:

### Data Acquisition

The project uses a subset of the **HaGrid (Hand Gesture Recognition Dataset)**.

- **Script:** `scripts/download_dataset.py`
- **Action:** Downloads training images from Hugging Face based on specific class labels.
- **Classes:** `call`, `dislike`, `fist`, `four`, `grabbing`, `grip`, `like`, `middle_finger`, `mute`, `no_gesture`, `ok`, `one`, `palm`, `peace`, `peace_inverted`, `rock`, `stop`, `stop_inverted`, `three`, `three2`, `three3`, `two_up`, `two_up_inverted`

### Landmark Extraction

Instead of training on raw pixels, we train on coordinate data to reduce model complexity and improve robustness.

- **Script:** `scripts/process_dataset.py`
- **Action:**
    - Loads raw images.
    - Uses MediaPipe to detect hand landmarks.
    - Translates landmarks to a relative coordinate system (origin at wrist).
    - Normalizes coordinates.
    - Saves the resulting vectors to `data/processed/landmarks.csv`.

### Training & Conversion

- **Script:** `scripts/train_model.py`
- **Action:**
    - Loads the CSV data.
    - Fits a `StandardScaler` to the data (saved as `scaler_params.json`).
    - Trains a Keras Sequential Neural Network with Dropout and BatchNormalization.
    - Evaluates performance and generates a classification report.
    - Converts the final Keras model to a quantized **TFLite** model for efficient real-time inference.

## Requirements

- Python 3.11.14
- A working webcam
- `uv` package manager (recommended) or `pip`

## Setup

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
   Create a `.env` file in the root directory if you need to download the dataset:
   ```env
   HF_TOKEN=your_huggingface_token_here
   ```

## Usage

### Running the Pipeline

Follow the steps in the [Pipeline](#creating-a-model) section above using the following commands:

```powershell
uv run scripts/download_dataset.py
uv run scripts/process_dataset.py
uv run scripts/train_model.py
```

### Running Real-Time Recognition

To start the application and test the trained model:

```powershell
uv run main.py
```

- Press **ESC** to exit the application.

## Project Structure

```text
gesture-recognizer/
├── data/               # Raw images and processed CSVs
├── logs/               # Training history and TensorBoard logs
├── main.py             # Application entry point
├── models/             # TFLite models, labels, and scalers
├── pyproject.toml      # Dependencies
├── scripts/            # Pipeline utility scripts
└── uv.lock             # Dependency lockfile
```

## License

This project is licensed under the **MIT License**. See [LICENSE.md](LICENSE.md) for details.
