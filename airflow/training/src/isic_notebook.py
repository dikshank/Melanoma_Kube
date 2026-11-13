"""
isic_notebook.py
----------------
Everything the serving model's training notebook did to images and scores,
in one place, so the cluster treats data exactly like the notebook:

    p2a_efficientnet_b0_imagenet_full_15ep  (EfficientNet-B0, ImageNet start,
    one sigmoid output, label = malignant (1) vs benign (0), ISIC images)

Used by:
    register_pretrained.py  - scores a model on the eval set
    finetune.py             - retraining (fine-tunes the Production model)
    tools/isic_subset/prepare_isic_subset.py has a copy of stage 1 (cache)
    backend/predictor.py has a copy of stage 1 + the val transform

Image stages (identical to the notebook):
    stage 1  cache_image():   draft -> RGB -> short side 320 (bilinear) -> JPEG q95
    stage 2  val:             Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize
             train:           square RandomResizedCrop(224, 50-100%) -> jitter
                              -> gamma 0.8-1.25 -> 50%: blur or downscale -> 50%: JPEG 30-70
                              -> ToTensor -> Normalize

Scores: the model's malignant probability is softmax(logits)[:, 1]. For the
served model (one logit z, wrapped as [0, z]) that equals sigmoid(z); the
logit used for BCE in fine-tuning is logits[:, 1] - logits[:, 0] (= z).
"""

import io
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ── notebook constants ───────────────────────────────────────────────────────
PRE_RESIZE_SHORT_SIDE = 320
CACHE_JPEG_QUALITY = 95
IMG_SIZE = 224
TARGET_RECALL = 0.90

CROP_AREA_RANGE = (0.5, 1.0)
CROP_RATIO = (1.0, 1.0)
JITTER = (0.2, 0.2, 0.2, 0.05)
GAMMA_RANGE = (0.8, 1.25)
BLUR_RADIUS_RANGE = (0.5, 2.0)
DOWNSCALE_RANGE = (0.35, 1.0)
JPEG_QUALITY_RANGE = (30, 70)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: the notebook's image cache
# ─────────────────────────────────────────────────────────────────────────────

def cache_image(source) -> Image.Image:
    """Path or bytes -> the notebook's cached image (short side 320, re-read from JPEG q95)."""
    opened = Image.open(io.BytesIO(source)) if isinstance(source, (bytes, bytearray)) else Image.open(source)
    with opened as image:
        image.draft("RGB", (2 * PRE_RESIZE_SHORT_SIDE, 2 * PRE_RESIZE_SHORT_SIDE))
        image = image.convert("RGB")
        width, height = image.size
        scale = PRE_RESIZE_SHORT_SIDE / min(width, height)
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=CACHE_JPEG_QUALITY)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: val and train transforms
# ─────────────────────────────────────────────────────────────────────────────

normalize = T.Normalize(MEAN, STD)
to_tensor = T.ToTensor()
center_crop = T.Compose([T.Resize(256), T.CenterCrop(IMG_SIZE)])
_color_jitter = T.ColorJitter(*JITTER)
_square_crop = T.RandomResizedCrop(IMG_SIZE, scale=CROP_AREA_RANGE, ratio=CROP_RATIO)


def val_tensor(cached: Image.Image) -> torch.Tensor:
    return normalize(to_tensor(center_crop(cached)))


def _jpeg(image, rng):
    q = int(rng.integers(JPEG_QUALITY_RANGE[0], JPEG_QUALITY_RANGE[1] + 1))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=q)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _downscale(image, rng):
    w, h = image.size
    f = rng.uniform(*DOWNSCALE_RANGE)
    small = image.resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def train_tensor(cached: Image.Image, rng: np.random.Generator) -> torch.Tensor:
    """The notebook's training_augmentation() + normalize."""
    image = _color_jitter(_square_crop(cached))
    image = TF.adjust_gamma(image, float(rng.uniform(*GAMMA_RANGE)))
    if rng.random() < 0.5:
        if rng.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(*BLUR_RADIUS_RANGE))))
        else:
            image = _downscale(image, rng)
    if rng.random() < 0.5:
        image = _jpeg(image, rng)
    return normalize(to_tensor(image))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LesionDataset(Dataset):
    """
    Rows with columns `filepath` (relative to cwd /opt/airflow, or absolute)
    and `label` (1 = malignant, 0 = benign).

    already_cached=True  : files are stage-1 output already (eval/replay sets)
    already_cached=False : raw files (feedback uploads) -> stage 1 is applied
    """

    def __init__(self, table: pd.DataFrame, is_training: bool):
        self.paths = table["filepath"].tolist()
        self.labels = table["label"].astype(np.float32).values
        self.cached = table.get("already_cached", pd.Series(True, index=table.index)).astype(bool).tolist()
        self.is_training = is_training

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB") if self.cached[i] else cache_image(self.paths[i])
        if self.is_training:
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
            x = train_tensor(image, rng)
        else:
            x = val_tensor(image)
        return x, self.labels[i]


def read_manifest(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"filepath", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df["label"] = df["label"].astype(int)
    df["already_cached"] = True
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Model outputs and metrics
# ─────────────────────────────────────────────────────────────────────────────

def malignant_logit(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Logit of malignant vs benign for a 2-output model (z for the [0, z] wrapper)."""
    out = model(x)
    if out.shape[1] == 1:
        return out[:, 0]
    return out[:, 1] - out[:, 0]


@torch.no_grad()
def predict_logits(model: nn.Module, loader) -> np.ndarray:
    model.eval()
    return np.concatenate([malignant_logit(model, x).float().numpy() for x, _ in loader])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def threshold_for_recall(y: np.ndarray, z: np.ndarray, target: float = TARGET_RECALL) -> float:
    """The notebook's weighted_quantile (weights 1): logit flagging `target` of the positives."""
    pos = np.sort(z[y == 1])
    if len(pos) == 0:
        return float("nan")
    cumulative = np.arange(1, len(pos) + 1) / len(pos)
    return float(np.interp(1 - target, cumulative, pos))


def metrics_at(y: np.ndarray, z: np.ndarray, t_logit: float) -> dict:
    flagged = z >= t_logit
    tp = int((flagged & (y == 1)).sum()); fn = int((~flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum()); tn = int((~flagged & (y == 0)).sum())
    div = lambda a, b: a / b if b else float("nan")
    recall, precision = div(tp, tp + fn), div(tp, tp + fp)
    return {
        "recall": recall, "precision": precision,
        "specificity": div(tn, tn + fp), "accuracy": div(tp + tn, len(y)),
        "f1": div(2 * precision * recall, precision + recall) if tp else 0.0,
    }


def evaluate(y: np.ndarray, z: np.ndarray, threshold_prob: float = None) -> dict:
    """
    Notebook-style scores on one set of logits:
      auc, spec_at_90recall (and its threshold), and the usual metrics at
      `threshold_prob` (the served threshold) if given, else at the 90% point.
    """
    t90 = threshold_for_recall(y, z)
    at90 = metrics_at(y, z, t90)
    t_logit = t90 if threshold_prob is None else float(np.log(threshold_prob / (1 - threshold_prob)))
    at_t = metrics_at(y, z, t_logit)
    return {
        "auc": roc_auc(y, z),
        "spec_at_90recall": at90["specificity"],
        "threshold_at_90recall": float(sigmoid(t90)),
        "threshold": float(sigmoid(t_logit)),
        **{k: v for k, v in at_t.items()},
        "n": int(len(y)), "positives": int((y == 1).sum()),
    }
