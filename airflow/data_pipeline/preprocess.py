"""
preprocess.py
-------------
Phase 1 & 2 | Executed: Local (inside Airflow Docker container)

Airflow Task 3: Preprocess images based on split manifests.

Responsibilities:
- Read train/val/test manifests produced by split.py
- Resize all images to target_size from config
- Convert all images to RGB (handles grayscale edge cases)
- Save processed images to processed_data_dir maintaining train/val/test structure
- Write FINAL manifests pointing to processed images using relative paths (Basically overriding the manifest csv craeted by split.py)
- Run post-processing validation to verify all is well
- Log throughput (images per second) and total processing time

Post-processing validation checks:
1. Processed folder exists and is not empty
2. All 3 splits exist (train/val/test)
3. Both classes exist in each split
4. Image count in processed matches manifest count
5. All processed images are the correct target size
6. Final manifests exist and have correct columns
7. Manifest filepaths are relative (not absolute)
8. Every filepath in manifest points to an existing file
"""


import time
import logging
import pandas as pd
from pathlib import Path
from PIL import Image

def make_relative_path(absolute_path: Path, anchor: str = "/opt/airflow/") -> str:
    """
    Convert an absolute container path to a relative path from project root.
    
    # yaml
    # Volume mount:
    #   ../data:/opt/airflow/data
    #
    # The docker-compose.yml file is located inside the `airflow/` directory:
    #
    #   project/
    #   ├── airflow/
    #   │   └── docker-compose-airflow.yml
    #   ├── data/
    #   ├── scripts/
    #   └── ...
    #
    # `../data` refers to the `data/` directory on the HOST machine.
    # `..` means "go one directory up from the directory containing this
    # Compose file", so:
    #
    #   airflow/../data  →  project/data
    #
    # The part after the colon, `/opt/airflow/data`, is the path INSIDE
    # the running Docker container.
    #
    # Therefore Docker creates the following mapping:
    #
    #   HOST                                CONTAINER
    #   project/data/  ──────────────────→  /opt/airflow/data/
    #
    # The data is NOT copied into the Docker image. Instead, Docker mounts
    # the host's `data/` directory into the container at container path
    # `/opt/airflow/data` when the Airflow container starts.
    #
    # Consequently, code running inside the Airflow container (e.g. split.py)
    # can access the host dataset using:
    #
    #   /opt/airflow/data/...
    #
    # even though `/opt/airflow/data` was not created by the Dockerfile.
    #
    # This is different from the Dockerfile:
    #
    #   Dockerfile → defines what is baked into the Docker image
    #   volumes    → defines host directories made available to the running
    #                container
    #
    # The same mechanism is used for other project directories:
    #
    #   ../training → /opt/airflow/training
    #   ../mlruns   → /opt/airflow/mlruns
    #   ../logs     → /opt/airflow/app_logs



    Inside Docker: /opt/airflow/data/processed/v1/train/malignant/img.jpg
    → relative:    data/processed/v1/train/malignant/img.jpg

    Args:
        absolute_path (Path): Absolute path inside Docker container.
        anchor (str): The container prefix to strip.

    Returns:
        str: Relative path from project root using forward slashes.
    """
    path_str = str(absolute_path).replace("\\", "/")
    if anchor in path_str:
        return path_str.split(anchor)[-1]    # Given this container path, remove the /opt/airflow/ prefix from the string.
    return path_str



def process_split(
    manifest_path: Path,
    split_name: str,
    output_dir: Path,
    target_size: tuple,
    convert_to_rgb: bool
) -> tuple:
    """
    Process all images for a single split (train, val, or test).

    Args:
        manifest_path (Path): Path to the intermediate split CSV manifest.
        split_name (str): One of 'train', 'val', 'test'.
        output_dir (Path): Root processed output directory.
        target_size (tuple): Target (width, height) for resizing.
        convert_to_rgb (bool): Whether to convert images to RGB mode.

    Returns:
        tuple: (summary dict, final_manifest_df with relative paths)
    """
    df = pd.read_csv(manifest_path)
    total = len(df)
    processed_count = 0
    failed_files = []
    final_manifest_rows = []

    start_time = time.time()

    for _, row in df.iterrows():
        src_path = Path(row["filepath"])
        label = row["label"]

        out_dir = output_dir / split_name / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / src_path.name

        try:
            with Image.open(src_path) as img:
                if convert_to_rgb and img.mode != "RGB":
                    img = img.convert("RGB")
                img_resized = img.resize(target_size, Image.LANCZOS)
                img_resized.save(out_path)
                processed_count += 1

                relative_path = make_relative_path(out_path)
                final_manifest_rows.append({
                    "filepath": relative_path,
                    "label": label
                })

        except Exception as e:
            failed_files.append(str(src_path))

    elapsed = time.time() - start_time
    throughput = processed_count / elapsed if elapsed > 0 else 0

    summary = {
        "split": split_name,
        "total": total,
        "processed": processed_count,
        "failed": len(failed_files),
        "elapsed_seconds": round(elapsed, 2),
        "throughput_imgs_per_sec": round(throughput, 2),
        "failed_files": failed_files
    }

    final_manifest_df = pd.DataFrame(final_manifest_rows)
    return summary, final_manifest_df



def validate_processed_output(
    processed_data_dir: Path,
    manifest_dir: Path,
    classes: list,
    target_size: tuple,
    summaries: dict
) -> None:
    """
    Run all post-processing validation checks.

    Checks:
        1. Processed folder exists and is not empty
        2. All 3 splits exist (train/val/test)
        3. Both classes exist in each split folder
        4. Image count in processed matches manifest count
        5. All processed images are the correct target size
        6. Final manifests exist and have correct columns
        7. Manifest filepaths are relative (not absolute)
        8. Every filepath in manifest points to an existing file

    Args:
        processed_data_dir (Path): Root of processed data output.
        manifest_dir (Path): Directory containing final manifest CSVs.
        classes (list): Expected class names.
        target_size (tuple): Expected (width, height) of processed images.
        summaries (dict): Processing summaries from process_split().

    Raises:
        ValueError: If any validation check fails.
    """
    errors = []

    # ── Check 1: Processed folder exists and is not empty ─────────────────
    if not processed_data_dir.exists():
        errors.append(f"Check 1 FAILED: Processed folder does not exist: {processed_data_dir}")
    else:
        all_files = list(processed_data_dir.rglob("*.jpg")) + \
            list(processed_data_dir.rglob("*.jpeg")) + \
            list(processed_data_dir.rglob("*.png"))
        if len(all_files) == 0:
            errors.append("Check 1 FAILED: Processed folder is empty.")

    # ── Check 2: All 3 splits exist ───────────────────────────────────────
    splits = ["train", "val", "test"]
    missing_splits = [s for s in splits if not (processed_data_dir / s).exists()]
    if missing_splits:
        errors.append(f"Check 2 FAILED: Missing split folders: {missing_splits}")

    # ── Check 3: Both classes exist in each split ─────────────────────────
    class_errors = []
    for split in splits:
        split_dir = processed_data_dir / split
        if not split_dir.exists():
            continue
        for class_name in classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                class_errors.append(f"{split}/{class_name}")
    if class_errors:
        errors.append(f"Check 3 FAILED: Missing class folders: {class_errors}")
    
    # ── Check 4: Image count matches manifest count ───────────────────────
    count_errors = []
    for split in splits:
        split_dir = processed_data_dir / split
        if not split_dir.exists():
            continue
        actual_count = len(list(split_dir.rglob("*.jpg"))) + \
            len(list(split_dir.rglob("*.jpeg"))) + \
            len(list(split_dir.rglob("*.png")))
        expected_count = summaries.get(split, {}).get("processed", 0)
        if actual_count != expected_count:
            count_errors.append(
                f"{split}: expected {expected_count}, found {actual_count}"
            )
    if count_errors:
        errors.append(f"Check 4 FAILED: Image count mismatch: {count_errors}")

    # ── Check 5: All processed images are the correct target size ─────────
    size_errors = []
    sample_limit = 10  # check first 10 images per split to keep it fast
    for split in splits:
        split_dir = processed_data_dir / split
        if not split_dir.exists():
            continue
        images = list(split_dir.rglob("*.jpg"))[:sample_limit]
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    if img.size != target_size:
                        size_errors.append(
                            f"{img_path.name}: expected {target_size}, got {img.size}"
                        )
            except Exception as e:
                size_errors.append(f"{img_path.name}: cannot open — {str(e)}")
    if size_errors:
        errors.append(f"Check 5 FAILED: Wrong image sizes: {size_errors[:3]}")


# ── Check 6: Final manifests exist and have correct columns ───────────
    manifest_errors = []
    required_columns = {"filepath", "label"}
    for split in splits:
        manifest_path = manifest_dir / f"{split}_manifest.csv"
        if not manifest_path.exists():
            manifest_errors.append(f"{split}_manifest.csv missing")
            continue
        df = pd.read_csv(manifest_path)
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            manifest_errors.append(f"{split}_manifest.csv missing columns: {missing_cols}")
        if len(df) == 0:
            manifest_errors.append(f"{split}_manifest.csv is empty")
    if manifest_errors:
        errors.append(f"Check 6 FAILED: Manifest issues: {manifest_errors}")

    # ── Check 7: Manifest filepaths are relative (not absolute) ───────────
    absolute_path_errors = []
    for split in splits:
        manifest_path = manifest_dir / f"{split}_manifest.csv"
        if not manifest_path.exists():
            continue
        df = pd.read_csv(manifest_path)
        absolute_paths = df[df["filepath"].str.startswith("/")]["filepath"].tolist()
        if absolute_paths:
            absolute_path_errors.append(
                f"{split}: {len(absolute_paths)} absolute paths found "
                f"(e.g. {absolute_paths[0]})"
            )
    if absolute_path_errors:
        errors.append(f"Check 7 FAILED: Absolute paths in manifests: {absolute_path_errors}")

    # ── Check 8: Every filepath in manifest points to existing file ────────
    missing_file_errors = []
    for split in splits:
        manifest_path = manifest_dir / f"{split}_manifest.csv"
        if not manifest_path.exists():
            continue
        df = pd.read_csv(manifest_path)
        # Check from /opt/airflow/ root (container) or local root
        # We check both the relative path and with /opt/airflow/ prefix
        missing = []
        for fp in df["filepath"].tolist():
            container_path = Path("/opt/airflow") / fp
            if not container_path.exists():
                missing.append(fp)
        if missing:
            missing_file_errors.append(
                f"{split}: {len(missing)} files missing "
                f"(e.g. {missing[0]})"
            )
    if missing_file_errors:
        errors.append(f"Check 8 FAILED: Missing files: {missing_file_errors}")

    # ── Final result ──────────────────────────────────────────────────────
    if errors:
        for err in errors:
            raise ValueError(
                f"Post-processing validation failed with {len(errors)} error(s). "
                f"See logs above."
            )

def run(config: dict) -> None:
    """
    Entry point called by Airflow PythonOperator.

    Args:
        config (dict): Parsed pipeline config YAML as a dictionary.
    """

    # Clear existing processed images before writing fresh ones
    # Prevents stale/feedback images from accumulating
    import shutil
    try:
        manifest_dir = Path(config["paths"]["split_manifest_dir"])
        processed_data_dir = Path(config["paths"]["processed_data_dir"])

        # Clear existing processed images before writing fresh ones
        for split in ['train', 'val', 'test']:
            split_dir = processed_data_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
                split_dir.mkdir(parents=True)
        target_size = tuple(config["preprocessing"]["target_size"])
        convert_to_rgb = config["preprocessing"]["convert_to_rgb"]
        classes = config["classes"]

        splits = ["train", "val", "test"]
        all_summaries = {}
        final_manifests = {}

        for split_name in splits:
            manifest_path = manifest_dir / f"{split_name}_manifest.csv"

            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Manifest not found: {manifest_path}. "
                    f"Run split.py first."
                )

            summary, final_df = process_split(
                manifest_path=manifest_path,
                split_name=split_name,
                output_dir=processed_data_dir,
                target_size=target_size,
                convert_to_rgb=convert_to_rgb
            )
            all_summaries[split_name] = summary
            final_manifests[split_name] = final_df

        # ── Write final manifests with relative paths ─────────────────────
        manifest_dir.mkdir(parents=True, exist_ok=True)
        for split_name, final_df in final_manifests.items():
            out_path = manifest_dir / f"{split_name}_manifest.csv"
            final_df.to_csv(out_path, index=False)


        # ── Overall throughput summary ────────────────────────────────────
        total_images = sum(s["processed"] for s in all_summaries.values())
        total_time = sum(s["elapsed_seconds"] for s in all_summaries.values())
        overall_throughput = total_images / total_time if total_time > 0 else 0

        # ── Post-processing validation ────────────────────────────────────
        validate_processed_output(
            processed_data_dir=processed_data_dir,
            manifest_dir=manifest_dir,
            classes=classes,
            target_size=target_size,
            summaries=all_summaries
        )


    except Exception as e:
        raise