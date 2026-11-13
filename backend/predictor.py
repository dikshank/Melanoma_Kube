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
        "a malignant skin lesion (such as melanoma or a skin carcinoma). Please consult a dermatologist or medical professional "
        "as soon as possible. This tool is for screening only and does not "
        "replace professional medical diagnosis."
    ),
    "benign": (
        "✅ Low risk detected. This image does not show strong signs of "
        "a malignant skin lesion. However, if you have concerns about a skin lesion, "
        "please consult a medical professional. Regular skin checks are "
        "always recommended."
    )
}


# ── Preprocessing: an exact copy of the training notebook ─────────────────────
# The served model is p2a_efficientnet_b0_imagenet_full_15ep (EfficientNet-B0,
# one sigmoid output, trained on ISIC). Its notebook prepared val images in two
# stages, and the backend repeats both, step for step:
#
#   stage 1 = load_and_resize_image() + cache_resized_image():
#       Image.open -> draft("RGB", (640, 640)) -> convert("RGB")
#       -> resize so the SHORT side is 320 (bilinear, aspect ratio kept)
#       -> saved as JPEG quality 95 (the image cache)
#   stage 2 = LesionDataset(is_training=False):
#       open the cached JPEG -> convert("RGB") -> Resize(256) -> CenterCrop(224)
#       -> ToTensor -> Normalize(ImageNet mean/std)
#
# Nothing is stretched: the centre square of the photo is used.
# Do not change any of these numbers without retraining the model.
PRE_RESIZE_SHORT_SIDE = 320
DRAFT_SIZE = (2 * PRE_RESIZE_SHORT_SIDE, 2 * PRE_RESIZE_SHORT_SIDE)
CACHE_JPEG_QUALITY = 95

_model_crop = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])
_to_model_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def model_input_crop(image_bytes: bytes) -> Image.Image:
    """
    The exact 224x224 image the model sees (before normalization), using the
    notebook's two stages (see above). Also used for drift detection, so the
    drift baseline and live uploads are compared on the same pixels.

    Raises:
        ValueError: If the image cannot be opened.
    """
    try:
        # ── stage 1: notebook load_and_resize_image() ────────────────────
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.draft("RGB", DRAFT_SIZE)        # JPEG: decode at reduced scale, like the notebook
            image = image.convert("RGB")
            width, height = image.size
            scale = PRE_RESIZE_SHORT_SIDE / min(width, height)
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.BILINEAR,
            )

        # ── stage 1b: the notebook stored this as a JPEG (quality 95) ─────
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=CACHE_JPEG_QUALITY)
        buffer.seek(0)

        # ── stage 2: notebook LesionDataset, validation path ─────────────
        cached = Image.open(buffer).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image: {str(e)}")

    return _model_crop(cached)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Turn uploaded image bytes into the (1, 3, 224, 224) tensor the model was
    validated on, using exactly the notebook's two stages (see above).
    """
    return _to_model_tensor(model_input_crop(image_bytes)).unsqueeze(0)


def predict(
    model: nn.Module,
    image_bytes: bytes,
    threshold: float,
) -> dict:
    """
    Run inference on a single image.

    Args:
        model (nn.Module): Loaded PyTorch model (in eval mode).
        image_bytes (bytes): Raw image bytes from upload.
        threshold (float): Classification threshold for malignant class.
                           Lower threshold → higher recall (fewer missed cancers).

    Returns:
        dict: Prediction result with label, confidence, probability, threshold.

    Raises:
        ValueError: If image preprocessing fails.
        RuntimeError: If model inference fails.
    """
    # ── Preprocess ────────────────────────────────────────────────────────
    tensor = preprocess_image(image_bytes)

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