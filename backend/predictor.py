"""
predictor.py
------------
Phase 4 | Executed: Local (backend Docker container)

Inference logic for melanoma classification.

Responsibilities:
- Preprocess uploaded image bytes to tensor
- Run inference through loaded model
- Apply recall-optimized threshold from training
- Return structured prediction result
"""

import io
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from logger import get_logger

logger = get_logger("predictor")

# ── Class mapping ─────────────────────────────────────────────────────────────
IDX_TO_LABEL = {0: "benign", 1: "malignant"}

# ── Recommendations for non-technical users ───────────────────────────────────
RECOMMENDATIONS = {
    "malignant": (
        "⚠️ High risk detected. This image shows signs that may indicate "
        "melanoma. Please consult a dermatologist or medical professional "
        "as soon as possible. This tool is for screening only and does not "
        "replace professional medical diagnosis."
    ),
    "benign": (
        "✅ Low risk detected. This image does not show strong signs of "
        "melanoma. However, if you have concerns about a skin lesion, "
        "please consult a medical professional. Regular skin checks are "
        "always recommended."
    )
}


def preprocess_image(image_bytes: bytes, img_size: int = 64) -> torch.Tensor:
    """
    Preprocess raw image bytes into a model-ready tensor.

    Applies the same normalization used during training:
        - Resize to img_size x img_size
        - Convert to RGB
        - Normalize with ImageNet mean/std

    Args:
        image_bytes (bytes): Raw image bytes from upload.
        img_size (int): Target size. Must match training img_size.

    Returns:
        torch.Tensor: Preprocessed tensor of shape (1, 3, img_size, img_size).

    Raises:
        ValueError: If image cannot be opened or processed.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image: {str(e)}")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(img).unsqueeze(0)   # add batch dimension → (1, 3, H, W)
    return tensor


def predict(
    model: nn.Module,
    image_bytes: bytes,
    threshold: float,
    img_size: int = 64
) -> dict:
    """
    Run inference on a single image.

    Args:
        model (nn.Module): Loaded PyTorch model (in eval mode).
        image_bytes (bytes): Raw image bytes from upload.
        threshold (float): Classification threshold for malignant class.
                           Lower threshold → higher recall (fewer missed cancers).
        img_size (int): Image size to resize to before inference.

    Returns:
        dict: Prediction result with label, confidence, probability, threshold.

    Raises:
        ValueError: If image preprocessing fails.
        RuntimeError: If model inference fails.
    """
    # ── Preprocess ────────────────────────────────────────────────────────
    tensor = preprocess_image(image_bytes, img_size)

    # ── Inference ─────────────────────────────────────────────────────────
    try:
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)

        malignant_prob = float(probs[0][1].item())
        benign_prob = float(probs[0][0].item())

    except Exception as e:
        raise RuntimeError(f"Model inference failed: {str(e)}")

    # ── Apply threshold ───────────────────────────────────────────────────
    # Using recall-optimized threshold from training
    # Lower than 0.5 means we flag more cases as malignant → fewer missed cancers
    if malignant_prob >= threshold:
        label = "malignant"
        confidence = malignant_prob
    else:
        label = "benign"
        confidence = benign_prob

    result = {
        "label": label,
        "confidence": round(confidence, 4),
        "malignant_prob": round(malignant_prob, 4),
        "threshold_used": round(threshold, 4),
        "recommendation": RECOMMENDATIONS[label]
    }

    logger.info(
        "Prediction: label=%s | confidence=%.4f | "
        "malignant_prob=%.4f | threshold=%.4f",
        label, confidence, malignant_prob, threshold
    )

    return result