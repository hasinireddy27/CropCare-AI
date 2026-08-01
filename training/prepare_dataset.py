from __future__ import annotations

import random
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = (
    BASE_DIR
    / "raw_dataset"
    / "plantvillage dataset"
    / "color"
)

OUTPUT_DIR = BASE_DIR / "prepared_dataset"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
RANDOM_SEED = 42

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def get_class_folders() -> list[Path]:
    """Find all crop-disease class folders automatically."""

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{SOURCE_DIR}"
        )

    class_folders = sorted(
        [
            folder
            for folder in SOURCE_DIR.iterdir()
            if folder.is_dir()
        ],
        key=lambda folder: folder.name.lower(),
    )

    if not class_folders:
        raise ValueError(
            f"No class folders found inside:\n{SOURCE_DIR}"
        )

    return class_folders


def get_image_files(class_folder: Path) -> list[Path]:
    """Return all supported images from one class folder."""

    return [
        image_path
        for image_path in class_folder.iterdir()
        if (
            image_path.is_file()
            and image_path.suffix.lower() in ALLOWED_EXTENSIONS
        )
    ]


def copy_images(
    image_paths: list[Path],
    destination_folder: Path,
) -> None:
    """Copy images into a train, validation or test folder."""

    destination_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    for image_path in image_paths:
        destination_path = destination_folder / image_path.name

        shutil.copy2(
            image_path,
            destination_path,
        )


def split_class_images(class_folder: Path) -> None:
    """Split one class into train, validation and test sets."""

    image_files = get_image_files(class_folder)

    if not image_files:
        print(
            f"Skipping {class_folder.name}: no images found."
        )
        return

    random.shuffle(image_files)

    total_images = len(image_files)

    train_count = int(total_images * TRAIN_RATIO)

    validation_count = int(
        total_images * VALIDATION_RATIO
    )

    train_images = image_files[:train_count]

    validation_images = image_files[
        train_count:
        train_count + validation_count
    ]

    test_images = image_files[
        train_count + validation_count:
    ]

    class_name = class_folder.name

    copy_images(
        train_images,
        OUTPUT_DIR / "train" / class_name,
    )

    copy_images(
        validation_images,
        OUTPUT_DIR / "validation" / class_name,
    )

    copy_images(
        test_images,
        OUTPUT_DIR / "test" / class_name,
    )

    print(f"\nClass: {class_name}")
    print(f"Total images: {total_images}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(validation_images)}")
    print(f"Test images: {len(test_images)}")


def main() -> None:
    random.seed(RANDOM_SEED)

    print(f"Reading dataset from:\n{SOURCE_DIR}\n")

    class_folders = get_class_folders()

    print(f"Found {len(class_folders)} classes.")

    for index, class_folder in enumerate(
        class_folders,
        start=1,
    ):
        print(f"{index}. {class_folder.name}")

    if OUTPUT_DIR.exists():
        print("\nRemoving previous prepared dataset...")
        shutil.rmtree(OUTPUT_DIR)

    print("\nPreparing crop disease dataset...")

    for class_folder in class_folders:
        split_class_images(class_folder)

    print("\nDataset preparation completed successfully.")
    print(f"Total classes prepared: {len(class_folders)}")
    print(f"Output location:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()