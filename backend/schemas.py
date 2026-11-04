"""
schemas.py
----------
Phase 4 | Executed: Local (backend Docker container)

Pydantic schemas for FastAPI request and response models.

Defines the API contract — matches exactly what is specified in LLD.md.
All endpoints use these schemas for input validation and response formatting.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class PredictResponse(BaseModel):
    """
    Response schema for POST /predict endpoint.

    Attributes:
        image_id      : Unique ID for this prediction — pass to /feedback
        label         : Predicted class — 'malignant' or 'benign'
        confidence    : Model confidence for the predicted class (0.0 to 1.0)
        malignant_prob: Raw probability of malignant class (0.0 to 1.0)
        threshold_used: Classification threshold applied to malignant_prob
        recommendation: Human-readable advice for non-technical users
    """
    model_config = ConfigDict(protected_namespaces=())

    image_id: str = Field(..., example="img_a1b2c3d4_1714000000.jpg")
    label: str = Field(..., example="malignant")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.87)
    malignant_prob: float = Field(..., ge=0.0, le=1.0, example=0.87)
    threshold_used: float = Field(..., ge=0.0, le=1.0, example=0.35)
    recommendation: str = Field(
        ...,
        example="High risk detected. Please consult a dermatologist immediately."
    )


class HealthResponse(BaseModel):
    """
    Response schema for GET /health endpoint.

    Attributes:
        status: Service status — always 'ok' if service is running
    """
    status: str = Field(..., example="ok")


class ReadyResponse(BaseModel):
    """
    Response schema for GET /ready endpoint.

    Attributes:
        model_loaded            : Whether the model has been loaded successfully
        model_name              : Name of the loaded model architecture
        model_version           : MLflow registry version of the loaded model
        classification_threshold: Threshold used for malignant classification
        status                  : 'ready' if model is loaded, 'not_ready' otherwise
    """
    model_config = ConfigDict(protected_namespaces=())

    model_loaded: bool = Field(..., example=True)
    model_name: Optional[str] = Field(None, example="mobilenet_v3_small")
    model_version: Optional[str] = Field(None, example="3")
    classification_threshold: Optional[float] = Field(None, example=0.35)
    status: str = Field(..., example="ready")
class FeedbackRequest(BaseModel):
    """
    Request schema for POST /feedback endpoint.

    Used to log ground truth labels for previous predictions.
    The image_id returned by /predict is used to retrieve
    the saved image for retraining.

    Attributes:
        image_id       : ID returned by /predict endpoint
        predicted_label: What the model predicted
        true_label     : Actual ground truth label confirmed by doctor
    """
    image_id: str = Field(..., example="img_a1b2c3d4_1714000000.jpg")
    predicted_label: str = Field(..., example="malignant")
    true_label: str = Field(..., example="malignant")


class FeedbackResponse(BaseModel):
    """
    Response schema for POST /feedback endpoint.

    Attributes:
        received: Whether feedback was successfully recorded
        message : Confirmation message
    """
    received: bool = Field(..., example=True)
    message: str = Field(..., example="Feedback recorded. Thank you.")