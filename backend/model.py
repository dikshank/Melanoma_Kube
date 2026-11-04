"""
model.py
--------
Phase 3 & 4 | Executed: Local (on-prem, CPU)

Model definitions for melanoma classification.

Supports three architectures:
    - mobilenet_v3_small   : Lightweight, pretrained ImageNet weights
    - efficientnet_b0      : Stronger baseline, pretrained ImageNet weights
    - simple_cnn           : Small custom CNN trained from scratch
                             No pretrained weights — pure baseline
                             Proves that architecture alone cannot overcome
                             the information loss in 32x32 images

Training strategy (pretrained models):
    Phase 1 (frozen base): Only the classifier head is trained.
    Phase 2 (fine-tuning): Last few layers of base are unfrozen.

Training strategy (simple_cnn):
    No freezing — all layers trained from scratch from epoch 1.
"""

import logging
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    EfficientNet_B0_Weights
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("model")

SUPPORTED_MODELS = ["mobilenet_v3_small", "efficientnet_b0", "simple_cnn"]


# ─────────────────────────────────────────────────────────────────────────────
# SimpleCNN — trained from scratch, no pretrained weights
# ─────────────────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    Small custom CNN trained from scratch for melanoma classification.

    Architecture:
        Block 1: Conv(3→32)  + BN + ReLU + MaxPool → 32x16x16 (for 32x32 input)
        Block 2: Conv(32→64) + BN + ReLU + MaxPool → 64x8x8
        Block 3: Conv(64→128)+ BN + ReLU + MaxPool → 128x4x4
        Global Average Pooling → 128
        FC(128→64) + ReLU + Dropout(0.5)
        FC(64→num_classes)

    Designed to work with any square input — img_size is passed at init
    so the FC layer is sized correctly.

    Args:
        num_classes (int): Number of output classes.
        img_size (int): Input image size (square). Default: 32.
    """

    def __init__(self, num_classes: int = 2, img_size: int = 32):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # ── Block 1 ───────────────────────────────────────────────────
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),         # → 32 x (img_size/2) x (img_size/2)

            # ── Block 2 ───────────────────────────────────────────────────
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),         # → 64 x (img_size/4) x (img_size/4)

            # ── Block 3 ───────────────────────────────────────────────────
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),         # → 128 x (img_size/8) x (img_size/8)
        )

        # Global Average Pooling — removes spatial dimensions
        self.gap = nn.AdaptiveAvgPool2d(1)   # → 128 x 1 x 1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def get_simple_cnn(num_classes: int, img_size: int = 32) -> nn.Module:
    """
    Create a SimpleCNN trained from scratch.

    No pretrained weights — all parameters randomly initialised.
    freeze_base is ignored for this model (nothing to freeze).

    Args:
        num_classes (int): Number of output classes.
        img_size (int): Input image size (square).

    Returns:
        nn.Module: SimpleCNN model.
    """
    model = SimpleCNN(num_classes=num_classes, img_size=img_size)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "SimpleCNN loaded | from_scratch=True | out_classes=%d | "
        "total_params=%d",
        num_classes, total
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Pretrained models
# ─────────────────────────────────────────────────────────────────────────────

def get_mobilenet_v3_small(num_classes: int, freeze_base: bool) -> nn.Module:
    """
    Load MobileNetV3-Small with pretrained ImageNet weights.

    Args:
        num_classes (int): Number of output classes.
        freeze_base (bool): If True, freeze all base layers except classifier.

    Returns:
        nn.Module: MobileNetV3-Small model.
    """
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False
        logger.info("MobileNetV3-Small: base frozen, training head only")

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    logger.info(
        "MobileNetV3-Small loaded | freeze_base=%s | out_classes=%d",
        freeze_base, num_classes
    )
    return model


def get_efficientnet_b0(num_classes: int, freeze_base: bool) -> nn.Module:
    """
    Load EfficientNet-B0 with pretrained ImageNet weights.

    Args:
        num_classes (int): Number of output classes.
        freeze_base (bool): If True, freeze all base layers except classifier.

    Returns:
        nn.Module: EfficientNet-B0 model.
    """
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False
        logger.info("EfficientNet-B0: base frozen, training head only")

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    logger.info(
        "EfficientNet-B0 loaded | freeze_base=%s | out_classes=%d",
        freeze_base, num_classes
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_model(
    model_name: str,
    num_classes: int = 2,
    freeze_base: bool = True,
    img_size: int = 32
) -> nn.Module:
    """
    Factory function to get a model by name.

    Args:
        model_name (str): One of 'mobilenet_v3_small', 'efficientnet_b0',
                          'simple_cnn'.
        num_classes (int): Number of output classes. Default: 2.
        freeze_base (bool): Whether to freeze base layers initially.
                            Ignored for simple_cnn.
        img_size (int): Input image size. Used by simple_cnn. Default: 32.

    Returns:
        nn.Module: Requested model.

    Raises:
        ValueError: If model_name is not supported.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            f"Choose from: {SUPPORTED_MODELS}"
        )

    if model_name == "mobilenet_v3_small":
        return get_mobilenet_v3_small(num_classes, freeze_base)

    elif model_name == "efficientnet_b0":
        return get_efficientnet_b0(num_classes, freeze_base)

    elif model_name == "simple_cnn":
        return get_simple_cnn(num_classes, img_size)


def unfreeze_last_layers(
    model: nn.Module,
    model_name: str,
    num_layers: int = 2
) -> nn.Module:
    """
    Unfreeze the last N layers of the base for fine-tuning phase.

    No-op for simple_cnn since it has no frozen layers.

    Args:
        model (nn.Module): Model with frozen base.
        model_name (str): Architecture name.
        num_layers (int): Number of feature layers to unfreeze from the end.

    Returns:
        nn.Module: Model with last N base layers unfrozen.
    """
    if model_name == "simple_cnn":
        logger.info("simple_cnn: no layers to unfreeze (trained from scratch)")
        return model

    if model_name in ["mobilenet_v3_small", "efficientnet_b0"]:
        layers = list(model.features.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Unfroze last %d layers of %s | Trainable params: %d",
        num_layers, model_name, trainable
    )
    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters in a model.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        dict: total, trainable, and frozen parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable
    }