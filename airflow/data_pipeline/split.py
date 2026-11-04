import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# Dataset structure detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_structure(raw_data_dir: Path, classes: list) -> str:
    """
    Detect whether the dataset is flat or pre-split.

    Args:
        raw_data_dir (Path): Root folder of the raw dataset.
        classes (list): Expected class names e.g. ['malignant', 'benign'].

    Returns:
        str: 'flat' if class folders are at root level,
             'presplit' if train/ and test/ subfolders exist.

    Raises:
        ValueError: If neither structure is detected.
    """
    # Check for flat structure: raw_data_dir/malignant/, raw_data_dir/benign/
    flat_dirs = [raw_data_dir / c for c in classes]
    if all(d.exists() and d.is_dir() for d in flat_dirs):
        return "flat"

    # Check for pre-split structure: raw_data_dir/train/, raw_data_dir/test/
    presplit_dirs = [raw_data_dir / "train", raw_data_dir / "test"]
    if all(d.exists() and d.is_dir() for d in presplit_dirs):
        return "presplit"

    raise ValueError(
        f"Cannot detect dataset structure at: {raw_data_dir}\n"
        f"Expected either:\n"
        f"  Flat: {raw_data_dir}/malignant/ and {raw_data_dir}/benign/\n"
        f"  Pre-split: {raw_data_dir}/train/ and {raw_data_dir}/test/"
    )


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame builders
# ─────────────────────────────────────────────────────────────────────────────
def build_dataframe_from_folder(folder: Path, classes: list, valid_extensions: set) -> pd.DataFrame:
    """
    Build a flat DataFrame from a folder containing one subfolder per class.

    Args:
        folder (Path): Folder containing class subfolders directly.
        classes (list): Expected class names.
        valid_extensions (set): Allowed file extensions.

    Returns:
        pd.DataFrame: DataFrame with columns [filepath, label].
    """
    records = []

    for class_name in classes:
        class_dir = folder / class_name

        if not class_dir.exists():
            continue

        for file_path in class_dir.iterdir():
            if file_path.suffix.lower() not in valid_extensions:
                continue
            if file_path.stat().st_size == 0:
                continue
            records.append({
                "filepath": str(file_path),
                "label": class_name
            })

    df = pd.DataFrame(records)
    return df



# ─────────────────────────────────────────────────────────────────────────────
# Split strategies
# ─────────────────────────────────────────────────────────────────────────────
def split_flat(df: pd.DataFrame, split_config: dict) -> tuple:
    """
    Case 1: Full stratified 70/15/15 split on a flat dataset.

    Args:
        df (pd.DataFrame): Full dataset with columns [filepath, label].
        split_config (dict): Split ratios and seed from config.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_ratio = split_config["train"]
    val_ratio = split_config["val"]
    test_ratio = split_config["test"]
    seed = split_config["random_seed"]

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # ── Stage 1: carve out test set ───────────────────────────────────────
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["label"],
        random_state=seed
    )

    # ── Stage 2: split remaining into train and val ───────────────────────
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df["label"],
        random_state=seed
    )

    return train_df, val_df, test_df


def split_presplit(
    raw_data_dir: Path,
    classes: list,
    valid_extensions: set,
    split_config: dict
) -> tuple:
    """
    Case 2: Dataset already has train/test folders.
    Only carve val out of train (stratified).
    Test set is used exactly as-is.

    Args:
        raw_data_dir (Path): Root folder containing train/ and test/ subfolders.
        classes (list): Expected class names.
        valid_extensions (set): Allowed file extensions.
        split_config (dict): Split config from config YAML.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    seed = split_config["random_seed"]

    # ── Load existing train and test sets ─────────────────────────────────
    full_train_df = build_dataframe_from_folder(
        raw_data_dir / "train", classes, valid_extensions
    )
    test_df = build_dataframe_from_folder(
        raw_data_dir / "test", classes, valid_extensions
    )

    # ── Carve val out of train (stratified) ──────────────────────────────
    val_ratio = split_config["val"]
    train_ratio = split_config["train"]

    # val ratio relative to train+val portion only
    val_ratio_of_train = val_ratio / (train_ratio + val_ratio)

    train_df, val_df = train_test_split(
        full_train_df,
        test_size=val_ratio_of_train,
        stratify=full_train_df["label"],
        random_state=seed
    )

    return train_df, val_df, test_df



def log_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """
    Log class distribution for each split.

    Args:
        train_df (pd.DataFrame): Training split.
        val_df (pd.DataFrame): Validation split.
        test_df (pd.DataFrame): Test split.
    """
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = df["label"].value_counts().to_dict()




def save_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    manifest_dir: Path
) -> None:
    """
    Save split manifests as CSV files.

    Args:
        train_df (pd.DataFrame): Training split.
        val_df (pd.DataFrame): Validation split.
        test_df (pd.DataFrame): Test split.
        manifest_dir (Path): Output directory for CSV files.
    """
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(manifest_dir / "train_manifest.csv", index=False)
    val_df.to_csv(manifest_dir / "val_manifest.csv", index=False)
    test_df.to_csv(manifest_dir / "test_manifest.csv", index=False)





# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run(config: dict) -> None:
    """
    Entry point called by Airflow PythonOperator.

    Detects dataset structure automatically and applies
    the appropriate split strategy.

    Args:
        config (dict): Parsed pipeline config YAML as a dictionary.
    """
    try:
        raw_data_dir = Path(config["paths"]["raw_data_dir"])
        manifest_dir = Path(config["paths"]["split_manifest_dir"])
        classes = config["classes"]
        valid_extensions = set(config["valid_extensions"])
        split_config = config["split"]

        # ── Auto-detect dataset structure ─────────────────────────────────
        structure = detect_structure(raw_data_dir, classes)

        # ── Apply appropriate split strategy ──────────────────────────────
        if structure == "flat":
            df = build_dataframe_from_folder(
                raw_data_dir, classes, valid_extensions
            )
            train_df, val_df, test_df = split_flat(df, split_config)

        else:  # presplit
            train_df, val_df, test_df = split_presplit(
                raw_data_dir, classes, valid_extensions, split_config
            )

        # ── Log and save ──────────────────────────────────────────────────
        log_split_summary(train_df, val_df, test_df)
        save_manifests(train_df, val_df, test_df, manifest_dir)


    except Exception as e:
        raise