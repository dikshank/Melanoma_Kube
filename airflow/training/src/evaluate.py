"""
evaluate.py
-----------
Phase 3 & 4 | Executed: Local (on-prem, CPU)

Evaluation functions for melanoma classification.

Responsibilities:
- Compute recall, F1, accuracy, AUC on a given dataloader
- Auto-tune classification threshold on validation set to maximize recall
- Generate confusion matrix
- Generate classification report
- Save all evaluation artifacts for MLflow logging
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from pathlib import Path

# ── Class names ───────────────────────────────────────────────────────────────
CLASS_NAMES = ["benign", "malignant"]

def get_predictions(model: nn.Module, dataloader: DataLoader, device: str) -> tuple:
    """
    Run inference on a dataloader and collect predictions and probabilities.

    Args:
        model (nn.Module): Trained PyTorch model.
        dataloader (DataLoader): DataLoader for val or test split.
        device (str): 'cuda' or 'cpu'.

    Returns:
        tuple: (all_labels, all_probs, all_preds)
            all_labels: ground truth labels (list of int)
            all_probs:  malignant class probabilities (list of float)
            all_preds:  predicted class indices at threshold=0.5 (list of int)
    """
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []


    with torch.no_grad():  # So that pytorch does not track the gradient, in my custom autograd I will try to see how this works
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]  # malignant probability
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    return all_labels, all_probs, all_preds


def tune_threshold(all_labels: list, all_probs: list) -> tuple:
    """
    Auto-tune classification threshold on validation set to maximize recall
    while keeping precision above a minimum floor.

    Searches thresholds from 0.1 to 0.9 in steps of 0.01.
    Picks the threshold that maximizes recall with precision >= min_precision.

    This is the core clinical decision: in cancer screening, missing a
    malignant case (false negative) is far more dangerous than a false alarm
    (false positive). So we lower the threshold to catch more positives.

    Args:
        all_labels (list): Ground truth labels.
        all_probs (list): Malignant class probabilities from model.

    Returns:
        tuple: (best_threshold, best_recall, best_f1)
    """
    min_precision = 0.3   # floor to avoid predicting everything as malignant

    best_threshold = 0.5
    best_recall = 0.0
    best_f1 = 0.0

    thresholds = np.arange(0.1, 0.9, 0.01)

    for thresh in thresholds:
        preds = [1 if p >= thresh else 0 for p in all_probs]

        recall = recall_score(all_labels, preds, pos_label=1, zero_division=0)
        f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)

        # Compute precision manually to apply floor
        tp = sum(1 for lbl, p in zip(all_labels, preds) if lbl == 1 and p == 1)
        fp = sum(1 for lbl, p in zip(all_labels, preds) if lbl == 0 and p == 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if recall > best_recall and precision >= min_precision:
            best_recall = recall
            best_threshold = thresh
            best_f1 = f1

    return float(best_threshold), float(best_recall), float(best_f1)


def compute_metrics(all_labels: list, all_probs: list, threshold: float) -> dict:
    """
    Compute all evaluation metrics at a given threshold.

    Args:
        all_labels (list): Ground truth labels.
        all_probs (list): Malignant class probabilities.
        threshold (float): Classification threshold.

    Returns:
        dict: All computed metrics.
    """
    preds = [1 if p >= threshold else 0 for p in all_probs]

    recall = recall_score(all_labels, preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
    accuracy = accuracy_score(all_labels, preds)

    # AUC requires both classes to be present
    if len(set(all_labels)) == 2:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.0

    metrics = {
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "auc": round(auc, 4),
        "threshold": round(threshold, 4)
    }

    return metrics



def compute_metrics(
    all_labels: list,
    all_probs: list,
    threshold: float
) -> dict:
    """
    Compute all evaluation metrics at a given threshold.

    Args:
        all_labels (list): Ground truth labels.
        all_probs (list): Malignant class probabilities.
        threshold (float): Classification threshold.

    Returns:
        dict: All computed metrics.
    """
    preds = [1 if p >= threshold else 0 for p in all_probs]

    recall = recall_score(all_labels, preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
    accuracy = accuracy_score(all_labels, preds)

    # AUC requires both classes to be present
    if len(set(all_labels)) == 2:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.0

    metrics = {
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "auc": round(auc, 4),
        "threshold": round(threshold, 4)
    }

    return metrics


def save_confusion_matrix(
    all_labels: list,
    all_preds: list,
    save_path: Path
) -> None:
    """
    Save confusion matrix as a JSON file for MLflow artifact logging.

    Args:
        all_labels (list): Ground truth labels.
        all_preds (list): Predicted labels.
        save_path (Path): Output path for JSON file.
    """
    cm = confusion_matrix(all_labels, all_preds)

    cm_dict = {
        "true_negative": int(cm[0][0]),
        "false_positive": int(cm[0][1]),
        "false_negative": int(cm[1][0]),
        "true_positive": int(cm[1][1])
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(cm_dict, f, indent=2)


def save_classification_report(
    all_labels: list,
    all_preds: list,
    save_path: Path
) -> None:
    """
    Save sklearn classification report as a text file for MLflow artifact logging.

    Args:
        all_labels (list): Ground truth labels.
        all_preds (list): Predicted labels.
        save_path (Path): Output path for text file.
    """
    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        f.write(report)

    print("\n📋 Classification Report:\n", report)




def evaluate(model: nn.Module, dataloader: DataLoader, device: str, split: str, artifacts_dir: Path, threshold: float = None) -> tuple:
    """
    Full evaluation pipeline for a model on a given dataloader.

    If threshold is None (val split), auto-tunes threshold to maximize recall.
    If threshold is provided (test split), uses that fixed threshold.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Val or test dataloader.
        device (str): Device to run inference on.
        split (str): 'val' or 'test' — affects threshold tuning.
        artifacts_dir (Path): Directory to save confusion matrix and report.
        threshold (float): Fixed threshold. If None, auto-tunes on val set.

    Returns:
        tuple: (metrics_dict, best_threshold)
    """

    # ── Get predictions ───────────────────────────────────────────────────
    all_labels, all_probs, _ = get_predictions(model, dataloader, device)

    # ── Threshold tuning or fixed ─────────────────────────────────────────
    if threshold is None:
        # Auto-tune on val set
        best_threshold, _, _ = tune_threshold(all_labels, all_probs)
    else:
        best_threshold = threshold

    # ── Compute metrics ───────────────────────────────────────────────────
    metrics = compute_metrics(all_labels, all_probs, best_threshold)

    # ── Final predictions at best threshold ───────────────────────────────
    final_preds = [1 if p >= best_threshold else 0 for p in all_probs]

    # ── Save artifacts ────────────────────────────────────────────────────
    save_confusion_matrix(
        all_labels, final_preds,
        artifacts_dir / f"{split}_confusion_matrix.json"
    )
    save_classification_report(
        all_labels, final_preds,
        artifacts_dir / f"{split}_classification_report.txt"
    )

    return metrics, best_threshold