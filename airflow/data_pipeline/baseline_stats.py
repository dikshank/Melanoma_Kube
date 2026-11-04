"""
baseline_stats.py
-----------------
Phase 1 & 2 | Executed: Local (inside Airflow Docker container)

Airflow Task 4: Compute and save baseline statistics for drift detection.

Responsibilities:
- Compute per-channel mean, variance, min, max across the full processed dataset
- Compute class distribution (count and percentage per class)
- Compute image size distribution (sanity check all images are correct size)
- Compute pixel intensity distribution (histogram buckets)
- Save all statistics as a JSON file to baseline_stats_path

This JSON is later used by Prometheus/monitoring to detect data drift
when new inference requests come in.

Output:
    baseline_stats/v1_stats.json (or v2_stats.json)
"""

import json
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
HISTOGRAM_BINS = 10



def collect_image_arrays(processed_data_dir: Path, classes: list) -> tuple:
    """
    Load all processed images from train split into numpy arrays.
    Uses train split only — val/test should not influence baselines.

    Args:
        processed_data_dir (Path): Root of processed data output.
        classes (list): Class names to iterate over.

    Returns:
        tuple: (pixel_arrays, class_counts)
            pixel_arrays: list of flattened float32 numpy arrays (one per image)
            class_counts: dict mapping class_name -> count
    """
    pixel_arrays = []
    class_counts = defaultdict(int)

    train_dir = processed_data_dir / "train"

    for class_name in classes:
        class_dir = train_dir / class_name

        if not class_dir.exists():
            continue

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            with Image.open(img_path) as img:
                arr = np.array(img, dtype=np.float32) / 255.0
                pixel_arrays.append(arr)
                class_counts[class_name] += 1


    return pixel_arrays, dict(class_counts)


def compute_channel_stats(pixel_arrays: list) -> dict:
    """
    Compute per-channel (R, G, B) statistics across all images.

    Args:
        pixel_arrays (list): List of numpy arrays with shape (H, W, C),
                             values normalized to [0, 1].

    Returns:
        dict: Per-channel mean, variance, std, min, max.

    Basically the return is :
                                {
                                    "R": {
                                        "mean": 0.52, # numbers are just placeholder
                                        "variance": 0.04,
                                        "std": 0.20,
                                        "min": 0.01,
                                        "max": 1.00
                                    },

                                    "G": {
                                        ...
                                    },

                                    "B": {
                                        ...
                                    }
                                }
    """
    # Stack all images: shape (N, H, W, C)
    stacked = np.stack(pixel_arrays, axis=0)

    channel_stats = {}
    channel_names = ["R", "G", "B"]

    for i, ch in enumerate(channel_names):
        channel_data = stacked[:, :, :, i].flatten()
        channel_stats[ch] = {
            "mean": float(np.mean(channel_data)),
            "variance": float(np.var(channel_data)),
            "std": float(np.std(channel_data)),
            "min": float(np.min(channel_data)),
            "max": float(np.max(channel_data))
        }


    return channel_stats
# Are we using the above function anywhere later to detect drift? What are the other methods to detect drift


def compute_pixel_histogram(pixel_arrays: list) -> dict:
    """
    Compute the global pixel-intensity distribution across all images/channels.

    Each image is flattened into a 1D array so that spatial location is ignored.
    The flattened arrays from all images are then concatenated into one large
    1D array containing every pixel value in the dataset.

    Example:
        Suppose we have two 2x2 grayscale images:

            Image 1:          Image 2:
            [[0.1, 0.2],      [[0.2, 0.3],
             [0.2, 0.1]]       [0.3, 0.2]]

        After flattening:

            Image 1 -> [0.1, 0.2, 0.2, 0.1]
            Image 2 -> [0.2, 0.3, 0.3, 0.2]

        After concatenating:

            all_pixels ->
            [0.1, 0.2, 0.2, 0.1, 0.2, 0.3, 0.3, 0.2]

        The histogram then counts how many pixel values fall into each
        intensity range. For example, with suitable bins:

            bin          pixel count
            0.0-0.2          2
            0.2-0.3          4
            0.3-0.4          2

        The important point is that we are measuring the overall distribution
        of pixel intensities, not where those pixels occur in the image.

        If another dataset contains fewer images, it will naturally contain
        fewer total pixels. Therefore, raw histogram counts should not be
        compared directly between datasets. For drift detection, the counts
        should be normalized by the total number of pixels:

            normalized_counts = counts / counts.sum()

        This converts the histogram into proportions, making datasets with
        different numbers of images/pixels comparable.

    Args:
        pixel_arrays (list):
            List of NumPy image arrays with pixel values normalized to [0, 1].

    Returns:
        dict:
            - bin_edges: Boundaries of the pixel-intensity histogram bins.
            - counts: Number of pixels falling into each bin.
            - total_pixels: Total number of pixel values used to build the
              histogram.
    """

    # Flatten every image and combine all pixels from all images/channels
    # into one 1D array. Spatial information is intentionally discarded
    # because this histogram measures only the global pixel-intensity
    # distribution.
    all_pixels = np.concatenate(
        [arr.flatten() for arr in pixel_arrays]
    )

    # Divide the normalized [0, 1] pixel-intensity range into
    # HISTOGRAM_BINS bins and count how many pixels fall into each bin.
    counts, bin_edges = np.histogram(
        all_pixels,
        bins=HISTOGRAM_BINS,
        range=(0.0, 1.0)
    )

    return {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
        "total_pixels": int(len(all_pixels))
    }
# Are we using the above function anywhere later to detect drift? What are the other methods to detect drift

def compute_image_size_distribution(processed_data_dir: Path, classes: list) -> dict:
    """
    Compute the distribution of image dimensions across the training dataset.

    Each image is represented by a size key in the format "WxH", for example
    "224x224". The function counts how many images have each unique
    width-height combination.

    Example:
        If the dataset contains:

            image1 -> 224x224
            image2 -> 224x224
            image3 -> 256x256
            image4 -> 224x224

        The resulting distribution is:

            {
                "224x224": 3,
                "256x256": 1
            }

        Here, "224x224": 3 means that 3 images have width=224 and height=224.

    This is primarily a data-quality and preprocessing consistency check.
    If all processed images are expected to have the same dimensions, the
    distribution should contain only the expected size. Unexpected sizes
    indicate possible preprocessing errors or anomalous input data.

    The same statistic can also be monitored on incoming production images.
    A change in the size distribution between the training baseline and
    production data can serve as an input/data-drift signal.

    Args:
        processed_data_dir (Path): Root directory of the processed dataset.
        classes (list): Class names to inspect.

    Returns:
        dict:
            Mapping from image size ("WxH") to the number of images with
            that dimension.
    """

    # Count how many images have each unique width × height combination.
    # defaultdict(int) initializes the count for a new image size to 0.
    size_counts = defaultdict(int)

    train_dir = processed_data_dir / "train"

    for class_name in classes:
        class_dir = train_dir / class_name

        if not class_dir.exists():
            continue

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            try:
                with Image.open(img_path) as img:
                    # Create a size identifier such as "224x224" and
                    # increment the number of images having that size.
                    size_key = f"{img.width}x{img.height}"
                    size_counts[size_key] += 1

            except Exception:
                # Ignore images that cannot be opened/read.
                pass

    return dict(size_counts)



def run(config: dict) -> None:
    """
    Entry point called by Airflow PythonOperator.

    Args:
        config (dict): Parsed pipeline config YAML as a dictionary.

    Returns: A basline json file that looks like:

                                        {
                                        "version": "v1",
                                        "total_train_images": 10000,

                                        "class_distribution": {
                                            "benign": {
                                            "count": 8000,
                                            "percentage": 80.0
                                            },
                                            "malignant": {
                                            "count": 2000,
                                            "percentage": 20.0
                                            }
                                        },

                                        "channel_stats": {
                                            "R": {
                                            "mean": 0.45,
                                            "variance": 0.03,
                                            "std": 0.17,
                                            "min": 0.0,
                                            "max": 1.0
                                            },
                                            "G": {
                                            "mean": 0.38,
                                            "variance": 0.02,
                                            "std": 0.14,
                                            "min": 0.0,
                                            "max": 1.0
                                            },
                                            "B": {
                                            "mean": 0.32,
                                            "variance": 0.02,
                                            "std": 0.14,
                                            "min": 0.0,
                                            "max": 1.0
                                            }
                                        },

                                        "pixel_histogram": {
                                            "bin_edges": [
                                            0.0,
                                            0.1,
                                            0.2,
                                            "...",
                                            1.0
                                            ],
                                            "counts": [
                                            100000,
                                            200000,
                                            "...",
                                            50000
                                            ],
                                            "total_pixels": 1505280000
                                        },

                                        "image_size_distribution": {
                                            "224x224": 10000
                                        }
                                        }

    """
    try:
        processed_data_dir = Path(config["paths"]["processed_data_dir"])
        baseline_stats_path = Path(config["paths"]["baseline_stats_path"])
        classes = config["classes"]
        version = config["version"]

        # ── Load images ───────────────────────────────────────────────────
        pixel_arrays, class_counts = collect_image_arrays(processed_data_dir, classes)

        if len(pixel_arrays) == 0:
            raise ValueError(
                "No images found in processed_data_dir train split. "
                "Run preprocess.py first."
            )

        # ── Compute stats ─────────────────────────────────────────────────
        channel_stats = compute_channel_stats(pixel_arrays)
        pixel_histogram = compute_pixel_histogram(pixel_arrays)
        size_distribution = compute_image_size_distribution(processed_data_dir, classes)

        # ── Class distribution percentages ────────────────────────────────
        total_images = sum(class_counts.values())
        class_distribution = {
            cls: {
                "count": count,
                "percentage": round(count / total_images * 100, 2)
            }
            for cls, count in class_counts.items()
        }

        # ── Assemble final stats object ───────────────────────────────────
        stats = {
            "version": version,
            "total_train_images": total_images,
            "class_distribution": class_distribution,
            "channel_stats": channel_stats,
            "pixel_histogram": pixel_histogram,
            "image_size_distribution": size_distribution
        }

        # ── Save to JSON ──────────────────────────────────────────────────
        baseline_stats_path.parent.mkdir(parents=True, exist_ok=True)

        with open(baseline_stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    except Exception as e:
        raise