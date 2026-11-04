"""
validate.py
-----------
Phase 1 & 2 | Executed: Local (inside Airflow Docker container)

Airflow Task 1: Validate raw images before any processing.

Handles two dataset structures automatically:

Case 1 — Flat structure:
    raw/v1/malignant/ and raw/v1/benign/ at root level

Case 2 — Pre-split structure:
    raw/v1/train/malignant/, raw/v1/train/benign/
    raw/v1/test/malignant/,  raw/v1/test/benign/

Responsibilities:
- Detect dataset structure (flat or presplit)
- Check every file can be opened as a valid image
- Check file extensions are in the allowed list
- Check for zero-byte / corrupt files
- Check minimum image count per class is met
- Log counts of valid vs rejected images
- Fail the task if corrupt percentage exceeds threshold
"""

import logging
from pathlib import Path
from PIL import Image


def detect_structure(raw_data_dir: Path, classes: list) -> str:
    flat_dirs = [raw_data_dir / c for c in classes]
    if all(d.exists() and d.is_dir() for d in flat_dirs):
        return "flat"

    presplit_dirs = [raw_data_dir / "train", raw_data_dir / "test"]
    if all(d.exists() and d.is_dir() for d in presplit_dirs):
        return "presplit"

    raise ValueError(
        f"Cannot detect dataset structure at: {raw_data_dir}. "
        f"Expected flat (malignant/, benign/) or presplit (train/, test/) structure."
    )


def get_class_dirs(raw_data_dir: Path, classes: list, structure: str) -> list:
    dirs = []
    if structure == "flat":
        for class_name in classes:
            dirs.append(("root", class_name, raw_data_dir / class_name))
    else:
        for split in ["train", "test"]:
            for class_name in classes:
                dirs.append((split, class_name, raw_data_dir / split / class_name))
    return dirs


def validate_class_dir(
    class_dir, class_name, split_name,
    valid_extensions, min_images, max_corrupt_pct
) -> dict:
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")

    all_files = list(class_dir.iterdir())
    total = len(all_files)
    valid_count = 0
    corrupt_files = []

    for file_path in all_files:
        if file_path.suffix.lower() not in valid_extensions:
            corrupt_files.append(str(file_path))
            continue
        if file_path.stat().st_size == 0:
            corrupt_files.append(str(file_path))
            continue
        try:
            with Image.open(file_path) as img:
                img.verify()
            valid_count += 1
        except Exception as e:
            corrupt_files.append(str(file_path))

    corrupt_count = len(corrupt_files)
    corrupt_pct = corrupt_count / total if total > 0 else 1.0

    if corrupt_pct > max_corrupt_pct:
        raise ValueError(
            f"Split [{split_name}] Class [{class_name}]: Corrupt percentage "
            f"({corrupt_pct:.1%}) exceeds threshold ({max_corrupt_pct:.1%})."
        )

    if valid_count < min_images:
        raise ValueError(
            f"Split [{split_name}] Class [{class_name}]: Only {valid_count} "
            f"valid images. Minimum required: {min_images}."
        )

    return {
        "total": total,
        "valid": valid_count,
        "corrupt": corrupt_count,
        "corrupt_pct": round(corrupt_pct * 100, 2)
    }


def validate_images(config: dict) -> dict:
    raw_data_dir = Path(config["paths"]["raw_data_dir"])
    classes = config["classes"]
    valid_extensions = set(config["valid_extensions"])
    min_images = config["validation"]["min_images_per_class"]
    max_corrupt_pct = config["validation"]["max_corrupt_pct"]


    structure = detect_structure(raw_data_dir, classes)
    class_dirs = get_class_dirs(raw_data_dir, classes, structure)

    summary = {}
    for split_name, class_name, class_dir in class_dirs:
        key = f"{split_name}/{class_name}"
        summary[key] = validate_class_dir(
            class_dir, class_name, split_name,
            valid_extensions, min_images, max_corrupt_pct
        )

    return summary


def run(config: dict) -> None:
    try:
        validate_images(config)
    except (ValueError, FileNotFoundError) as e:
        raise