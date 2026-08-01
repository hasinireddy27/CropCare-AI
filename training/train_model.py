from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRAINING_DIR.parent

DATASET_DIR = TRAINING_DIR / "prepared_dataset"
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"

MODEL_DIR = PROJECT_DIR / "model"
OUTPUT_DIR = TRAINING_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "crop_disease_model.keras"
BEST_MODEL_PATH = MODEL_DIR / "best_crop_disease_model.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

TRAINING_HISTORY_PATH = OUTPUT_DIR / "training_history.json"
ACCURACY_GRAPH_PATH = OUTPUT_DIR / "accuracy_graph.png"
LOSS_GRAPH_PATH = OUTPUT_DIR / "loss_graph.png"


# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

BATCH_SIZE = 32
INITIAL_EPOCHS = 15
LEARNING_RATE = 0.001
RANDOM_SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


def check_required_folders() -> None:
    """Ensure the prepared dataset folders exist."""

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training dataset folder not found:\n{TRAIN_DIR}"
        )

    if not VALIDATION_DIR.exists():
        raise FileNotFoundError(
            f"Validation dataset folder not found:\n{VALIDATION_DIR}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_datasets() -> tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    list[str],
]:
    """Load the training and validation datasets."""

    print("\nLoading training dataset...")

    train_dataset = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    print("\nLoading validation dataset...")

    validation_dataset = keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_dataset.class_names

    print(f"\nNumber of classes: {len(class_names)}")

    for index, class_name in enumerate(class_names):
        print(f"{index}: {class_name}")

    train_dataset = train_dataset.prefetch(
        buffer_size=AUTOTUNE
    )

    validation_dataset = validation_dataset.prefetch(
        buffer_size=AUTOTUNE
    )

    return (
        train_dataset,
        validation_dataset,
        class_names,
    )


def save_class_labels(class_names: list[str]) -> None:
    """Save class names in the exact model-output order."""

    label_mapping = {
        str(index): class_name
        for index, class_name in enumerate(class_names)
    }

    with LABELS_PATH.open(
        "w",
        encoding="utf-8",
    ) as labels_file:
        json.dump(
            label_mapping,
            labels_file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nClass labels saved to:\n{LABELS_PATH}")


def build_model(number_of_classes: int) -> keras.Model:
    """Build a MobileNetV2 transfer-learning model."""

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.12),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.12),
        ],
        name="data_augmentation",
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            3,
        ),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = False

    inputs = keras.Input(
        shape=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            3,
        ),
        name="input_image",
    )

    x = data_augmentation(inputs)

    x = keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(
        x,
        training=False,
    )

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.Dropout(
        0.35,
        name="dropout",
    )(x)

    outputs = layers.Dense(
        number_of_classes,
        activation="softmax",
        name="crop_disease_predictions",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cropcare_mobilenetv2",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=[
            keras.metrics.CategoricalAccuracy(
                name="accuracy"
            ),
            keras.metrics.TopKCategoricalAccuracy(
                k=3,
                name="top_3_accuracy",
            ),
        ],
    )

    return model


def create_callbacks() -> list[keras.callbacks.Callback]:
    """Create callbacks for safe and efficient training."""

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            OUTPUT_DIR / "training_log.csv"
        ),
    ]


def save_training_history(
    history: keras.callbacks.History,
) -> None:
    """Save training metrics as JSON."""

    serializable_history = {
        metric_name: [
            float(value)
            for value in metric_values
        ]
        for metric_name, metric_values in history.history.items()
    }

    with TRAINING_HISTORY_PATH.open(
        "w",
        encoding="utf-8",
    ) as history_file:
        json.dump(
            serializable_history,
            history_file,
            indent=4,
        )


def create_accuracy_graph(
    history: keras.callbacks.History,
) -> None:
    """Create and save the accuracy graph."""

    epochs = range(
        1,
        len(history.history["accuracy"]) + 1,
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        history.history["accuracy"],
        label="Training accuracy",
    )

    plt.plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation accuracy",
    )

    plt.title("Crop Disease Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        ACCURACY_GRAPH_PATH,
        dpi=200,
    )

    plt.close()


def create_loss_graph(
    history: keras.callbacks.History,
) -> None:
    """Create and save the loss graph."""

    epochs = range(
        1,
        len(history.history["loss"]) + 1,
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        history.history["loss"],
        label="Training loss",
    )

    plt.plot(
        epochs,
        history.history["val_loss"],
        label="Validation loss",
    )

    plt.title("Crop Disease Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        LOSS_GRAPH_PATH,
        dpi=200,
    )

    plt.close()


def print_tensorflow_information() -> None:
    """Print TensorFlow and GPU information."""

    print("=" * 60)
    print("CropCare AI Model Training")
    print("=" * 60)

    print(f"TensorFlow version: {tf.__version__}")

    available_gpus = tf.config.list_physical_devices(
        "GPU"
    )

    if available_gpus:
        print(f"GPU detected: {available_gpus}")
    else:
        print("GPU not detected. Training will use the CPU.")


def main() -> None:
    print_tensorflow_information()
    check_required_folders()

    (
        train_dataset,
        validation_dataset,
        class_names,
    ) = create_datasets()

    save_class_labels(class_names)

    model = build_model(
        number_of_classes=len(class_names)
    )

    print("\nModel summary:")

    model.summary()

    callbacks = create_callbacks()

    print("\nStarting MobileNetV2 training...")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_PATH)

    save_training_history(history)
    create_accuracy_graph(history)
    create_loss_graph(history)

    print("\nTraining completed successfully.")

    print(f"\nFinal model:\n{MODEL_PATH}")
    print(f"\nBest model:\n{BEST_MODEL_PATH}")
    print(f"\nClass labels:\n{LABELS_PATH}")
    print(f"\nAccuracy graph:\n{ACCURACY_GRAPH_PATH}")
    print(f"\nLoss graph:\n{LOSS_GRAPH_PATH}")


if __name__ == "__main__":
    main()