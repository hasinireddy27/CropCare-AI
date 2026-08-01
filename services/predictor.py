from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "model"
    / "best_crop_disease_model.keras"
)

LABELS_PATH = PROJECT_DIR / "model" / "labels.json"

IMAGE_SIZE = (224, 224)
MINIMUM_CONFIDENCE = 0.55


class CropDiseasePredictor:
    """Load the trained model and predict crop diseases."""

    def __init__(self) -> None:
        self.model = self._load_model()
        self.labels = self._load_labels()

    def _load_model(self) -> tf.keras.Model:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found:\n{MODEL_PATH}"
            )

        print(f"Loading crop disease model from:\n{MODEL_PATH}")

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )

        print("Crop disease model loaded successfully.")

        return model

    def _load_labels(self) -> dict[int, str]:
        if not LABELS_PATH.exists():
            raise FileNotFoundError(
                f"Labels file not found:\n{LABELS_PATH}"
            )

        with LABELS_PATH.open(
            "r",
            encoding="utf-8",
        ) as labels_file:
            raw_labels = json.load(labels_file)

        return {
            int(index): class_name
            for index, class_name in raw_labels.items()
        }

    def preprocess_image(
        self,
        image_path: Path,
    ) -> np.ndarray:
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = image.resize(
                    IMAGE_SIZE,
                    Image.Resampling.LANCZOS,
                )

                image_array = np.asarray(
                    image,
                    dtype=np.float32,
                )

        except UnidentifiedImageError as error:
            raise ValueError(
                "The uploaded file is not a valid image."
            ) from error

        except OSError as error:
            raise ValueError(
                "The uploaded image could not be processed."
            ) from error

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        return image_array

    @staticmethod
    def format_class_name(
        raw_class_name: str,
    ) -> tuple[str, str]:
        cleaned_name = raw_class_name.replace("_", " ")

        if "___" in raw_class_name:
            crop_name, disease_name = raw_class_name.split(
                "___",
                maxsplit=1,
            )
        else:
            crop_name = "Unknown crop"
            disease_name = raw_class_name

        crop_name = (
            crop_name
            .replace("_", " ")
            .replace("(maize)", "")
            .replace(",", "")
            .strip()
        )

        disease_name = (
            disease_name
            .replace("_", " ")
            .strip()
        )

        return crop_name, disease_name

    def predict(
        self,
        image_path: Path,
    ) -> dict:
        image_array = self.preprocess_image(image_path)

        predictions = self.model.predict(
            image_array,
            verbose=0,
        )[0]

        top_indices = np.argsort(predictions)[-3:][::-1]

        top_predictions = []

        for index in top_indices:
            raw_label = self.labels[int(index)]
            crop_name, disease_name = self.format_class_name(
                raw_label
            )

            top_predictions.append(
                {
                    "class_index": int(index),
                    "raw_label": raw_label,
                    "crop": crop_name,
                    "disease": disease_name,
                    "confidence": float(
                        predictions[index] * 100
                    ),
                }
            )

        best_prediction = top_predictions[0]

        is_reliable = (
            best_prediction["confidence"]
            >= MINIMUM_CONFIDENCE * 100
        )

        is_healthy = (
            "healthy"
            in best_prediction["disease"].lower()
        )

        return {
            "crop": best_prediction["crop"],
            "disease": best_prediction["disease"],
            "confidence": round(
                best_prediction["confidence"],
                2,
            ),
            "raw_label": best_prediction["raw_label"],
            "is_reliable": is_reliable,
            "is_healthy": is_healthy,
            "top_predictions": top_predictions,
        }