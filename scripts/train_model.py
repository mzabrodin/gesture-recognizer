import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
tf.random.set_seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "processed", "landmarks.csv"))
MODEL_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models"))
MODEL_KERAS = os.path.join(MODEL_DIR, "gesture_classifier.keras")
MODEL_TFLITE = os.path.join(MODEL_DIR, "gesture_classifier.tflite")
LABELS_TXT = os.path.join(MODEL_DIR, "labels.txt")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler_params.json")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
LOGS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "logs", run_id))

EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 15
LEARNING_RATE = 0.001


def main():
    if not os.path.exists(DATA_CSV):
        print(f"Error: Dataset file not found at {DATA_CSV}")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_CSV)  # type: ignore[call-overload]

    print("\nClass distribution in the dataset:")
    print(df["label_name"].value_counts())
    print("\n")

    feature_columns = [col for col in df.columns if col.startswith("coord_")]
    X = df[feature_columns].values.astype(np.float32)
    y = df["label_id"].values

    labels_df = df[["label_id", "label_name"]].drop_duplicates().sort_values("label_id")
    num_classes = len(labels_df)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    with open(LABELS_TXT, "w", encoding="utf-8") as f:
        for _, row in labels_df.iterrows():
            f.write(f"{row['label_name']}\n")

    print(f"Saved {num_classes} class labels to {LABELS_TXT}")
    print(f"Dataset shape: {X.shape}, Classes: {num_classes}")

    # Split data: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Training samples: {len(X_train)} | Validation samples: {len(X_val)} | Testing samples: {len(X_test)}")

    print("Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    scaler_params = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    with open(SCALER_FILE, "w") as f:
        json.dump(scaler_params, f)
    print(f"Saved scaler parameters to {SCALER_FILE}")

    print("\nBuilding model...")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(42,)),
            tf.keras.layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nModel Summary:")
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_KERAS,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR, histogram_freq=1, write_graph=True),
    ]

    print("\nStarting model training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    print("\n" + "=" * 50)
    print("EVALUATING MODEL ON TEST SET")
    print("=" * 50)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    label_names = labels_df["label_name"].tolist()
    print(classification_report(y_test, y_pred_classes, target_names=label_names, zero_division=0))

    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)

    history_file = os.path.join(LOGS_DIR, "training_history.json")
    history_dict = {
        "loss": [float(x) for x in history.history["loss"]],
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
    }
    with open(history_file, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"\nTraining history saved to {history_file}")

    print("\n" + "=" * 50)
    print("CONVERTING TO TFLITE FORMAT")
    print("=" * 50)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(MODEL_TFLITE, "wb") as f:
        f.write(tflite_model)

    tflite_size = os.path.getsize(MODEL_TFLITE) / 1024
    keras_size = os.path.getsize(MODEL_KERAS) / 1024
    print(f"TFLite model saved to {MODEL_TFLITE}")
    print(f"TFLite model size: {tflite_size:.2f} KB")
    print(f"Keras model size: {keras_size:.2f} KB")
    print(f"Size reduction: {((keras_size - tflite_size) / keras_size * 100):.1f}%")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print("\nFiles created:")
    print(f"  - Model (Keras): {MODEL_KERAS}")
    print(f"  - Model (TFLite): {MODEL_TFLITE}")
    print(f"  - Labels: {LABELS_TXT}")
    print(f"  - Scaler params: {SCALER_FILE}")
    print(f"  - Training logs: {LOGS_DIR}")
    print("\nTo visualize training with TensorBoard, run:")
    print(f"  tensorboard --logdir {LOGS_DIR}")


if __name__ == "__main__":
    main()
