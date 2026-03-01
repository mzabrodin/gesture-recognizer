from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from src.config import PATHS
from src.core.classifier import GestureClassifier, InferenceAssets, load_inference_assets

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")


sys.path.insert(0, str(ROOT_DIR / "src"))



@pytest.mark.skipif(
    not PATHS.gesture_classifier.exists() or not PATHS.scaler_params.exists(),
    reason="Model or scaler params are missing; train the model first.",
)
def test_classifier_predicts_label():
    assets: InferenceAssets = load_inference_assets()
    classifier = GestureClassifier(assets)

    class DummyLm:
        def __init__(self, x: float, y: float) -> None:
            self.x = x
            self.y = y

    landmarks = [DummyLm(float(i) / 40.0, float(i) / 40.0) for i in range(21)]

    label, conf = classifier.predict(landmarks)

    assert isinstance(label, str)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0
