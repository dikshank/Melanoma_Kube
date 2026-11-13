"""
main.py
-------
Phase 4 | Executed: Local (backend Docker container)

FastAPI backend for melanoma classification.

Endpoints:
    POST /predict   → upload image, get malignant/benign prediction
    GET  /health    → service liveness check
    GET  /ready     → model readiness check + auto-reload detection
    GET  /metrics   → Prometheus scrape endpoint
    POST /feedback  → submit ground truth label for real-world recall tracking

Key additions over base version:
    - Images saved at /predict time for retraining use
    - Drift detection on every prediction (vs baseline_stats.json)
    - Misclassification rate tracking
    - Automatic retraining trigger when thresholds exceeded
"""

import io
import os
import time
import json
import hashlib
import numpy as np
from collections import deque
from pathlib import Path
from contextlib import asynccontextmanager
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response, JSONResponse

from logger import get_logger
from model_loader import load_model, get_model, get_model_meta, check_and_reload
from predictor import predict, model_input_crop
from schemas import (
    PredictResponse,
    HealthResponse,
    ReadyResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from monitoring import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    PREDICTION_COUNTER,
    CONFIDENCE_HISTOGRAM,
    MALIGNANT_PROBABILITY,
    ERROR_COUNTER,
    FEEDBACK_COUNTER,
    REAL_WORLD_RECALL,
    REAL_WORLD_PRECISION,
    MISCLASSIFICATION_COUNTER,
    MISCLASSIFICATION_RATE,
    DRIFT_SCORE,
    DRIFT_DETECTED,
    RETRAINING_TRIGGERED,
    AVG_LATENCY,
)


logger = get_logger("main")
# All the paths below are host binded so they will exist locally in the host
# ── Config from environment ───────────────────────────────────────────────────
# Thresholds for triggering retraining
MISCLASSIFICATION_THRESHOLD = float(
    os.environ.get("MISCLASSIFICATION_THRESHOLD", "0.10")
)
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.20"))
# Minimum feedback count before misclassification rate is considered reliable
MIN_FEEDBACK_FOR_TRIGGER = int(os.environ.get("MIN_FEEDBACK_FOR_TRIGGER", "10"))

# ── Paths ─────────────────────────────────────────────────────────────────────
LOGS_DIR = Path("logs")  # relative path so the logger writes to /app/logs that is why in the volumeMount it have to given this path
FEEDBACK_LOG_PATH = (
    LOGS_DIR / "feedback.jsonl"
)  # .jsonl means JSON Lines. Instead of one big JSON object, the file contains one JSON object per line so each feedback record stored as a separate JSON line.
RETRAINING_FLAG_PATH = (
    LOGS_DIR / "retrain_needed.flag"
)  # the .flag file is useful when the signal needs to survive restarts of containers.

# Images saved at predict time — keyed by image_id
PENDING_FEEDBACK_DIR = LOGS_DIR / "pending_feedback"

# Images confirmed via feedback — organised by true label for retraining
FEEDBACK_DATA_DIR = LOGS_DIR / "feedback_data"
# Baseline stats from Airflow pipeline — used for drift detection
BASELINE_STATS_PATH = Path(
    os.environ.get("BASELINE_STATS_PATH", "data/baseline_stats/v1_stats.json")
)

# Create all required directories with open permissions
# so Airflow container can read/delete files written by backend container and then retraining dag combines then with the processed data and no we don't need to have the feedback manifest csv, the retraining_dag will take care of it
for d in [
    LOGS_DIR,
    PENDING_FEEDBACK_DIR,
    FEEDBACK_DATA_DIR / "malignant",
    FEEDBACK_DATA_DIR / "benign",
]:
    d.mkdir(
        parents=True, exist_ok=True
    )  # parents=True means create any missing parent directories too.
    os.chmod(d, 0o777)  # This 777 will create the read + write + execute permission

# ── In-memory state ───────────────────────────────────────────────────────────
_feedback_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
_baseline_histogram = None  # loaded once at startup

# In backend we want them inorder to detect drift when actual images come, so the production images histogram will be compared against the baseline_histogram
def _compute_image_histogram(image_bytes: bytes, n_bins: int = 10) -> list:
    """
    Compute normalised pixel intensity histogram for an image.

    Args:
        image_bytes (bytes): Raw image bytes.
        n_bins (int): Number of histogram bins.

    Returns:
        list: Normalised histogram counts.
    """
    try:
        # Same pixels the model sees (224x224 notebook crop, before normalization),
        # the same way the baseline (isic_v1_stats.json) was computed.
        img = model_input_crop(image_bytes)
        arr = np.array(img, dtype=np.float32) / 255.0
        counts, _ = np.histogram(arr.flatten(), bins=n_bins, range=(0.0, 1.0))
        total = sum(counts)
        return [c / total for c in counts] if total > 0 else list(counts)
    except Exception:
        return None


# We will load the baseline histogram so we can use them to compare them with the production image histograms
def _load_baseline_histogram() -> list:
    """
    Load pixel histogram from baseline_stats.json for drift detection.

    Returns:
        list: Normalised histogram counts, or None if file not found.
    """
    if not BASELINE_STATS_PATH.exists():
        logger.warning(
            "Baseline stats not found at %s. Drift detection disabled.",
            BASELINE_STATS_PATH,
        )
        return None

    try:
        with open(BASELINE_STATS_PATH) as f:
            stats = json.load(f)
        counts = stats["pixel_histogram"]["counts"]
        total = sum(counts)
        # Normalise to proportions
        normalised = [c / total for c in counts] if total > 0 else counts
        logger.info("Baseline histogram loaded (%d bins)", len(normalised))
        return normalised
    except Exception as e:
        logger.warning("Could not load baseline histogram: %s", str(e))
        return None


# Now the actual comparison
def _compute_drift_score(image_histogram: list) -> float:
    """
    Drift score = total variation distance between the image histogram and the
    baseline histogram: half the sum of the absolute differences.

    Both histograms are proportions summing to 1, so this is 0.0 for identical
    distributions and 1.0 for completely different ones. (The old version took
    the MEAN over 10 bins, which can never exceed 0.2, so a 0.20 threshold
    could never fire.)

    Args:
        image_histogram (list): Normalised histogram of incoming image.

    Returns:
        float: Drift score (0.0 = no drift, 1.0 = maximum drift).
    """
    if _baseline_histogram is None or image_histogram is None:
        return 0.0

    if len(_baseline_histogram) != len(image_histogram):
        raise ValueError("Histogram bin counts do not match")

    diff = [
        abs(_baseline_histogram[i] - image_histogram[i])
        for i in range(len(_baseline_histogram))
    ]

    return float(sum(diff) / 2)


# This will be used when we need to register that something is not right and model needs retraining
# It will be used in the prediction function, when drift occurs in the incoming user images and the baseline images
# Another place it will be used when the misclassification rate on the images have crossed a thershold and yes the user will let us know if he is an expert, so far any user can mess it up for us but we need to have a wall so we only allow experts to enter that detail
def _check_and_flag_retraining(reason: str) -> None:
    """
    Write a retraining flag file and update Prometheus counter.
    The Airflow retraining DAG polls this file to trigger retraining.

    Args:
        reason (str): 'misclassification' or 'drift'
    """
    flag_content = {
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feedback_counts": _feedback_counts.copy(),  # feedback_counts is a global variable so when this event happens we are just taking a snapshot of that state.
    }

    with open(RETRAINING_FLAG_PATH, "w") as f:
        json.dump(flag_content, f, indent=2)

    RETRAINING_TRIGGERED.labels(
        reason=reason
    ).inc()  # update the prometheus global state for retraining
    logger.warning(
        "⚠️ RETRAINING FLAGGED — reason=%s | " "Flag written to: %s",
        reason,
        RETRAINING_FLAG_PATH,
    )

########################################### TO start FAST API ############################################
# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _baseline_histogram
    logger.info("FastAPI starting up — loading Production model...")
    success = load_model()  # from the model_loader script
    if success:
        logger.info("✅ Model loaded successfully at startup")
    else:
        logger.error(
            "❌ Model failed to load at startup. "
            "/predict will return 503 until model is available."
        )
    _baseline_histogram = (
        _load_baseline_histogram()
    )  # load the baseline histogram and make it global
    yield  # pause the function, similar to what we have in ocaml
    logger.info(
        "FastAPI shutting down"
    )  # when this function is called again it will start after the yield

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Melanoma Detection API",
    description=(
        "Low-resolution melanoma detection for resource-constrained environments."
        "Upload a skin lesion image to get a malignant/benign classification."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the request even if it was not generated from the same origin (Cross origin Allowed)
# app.add_middleware(
#     CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
# )

# ─────────────────────────────────────────────────────────────────────────────
# POST /predict
# ─────────────────────────────────────────────────────────────────────────────

DRIFT_WINDOW_SIZE = 100
# Rolling window: always the last 100 scores of this pod. (The old plain list
# was checked only when it was exactly 100 long, i.e. once per pod lifetime.)
_drift_scores = deque(maxlen=DRIFT_WINDOW_SIZE)
_latency_sum = 0.0
_latency_count = 0


def _update_drift_and_check() -> float:
    if len(_drift_scores) < DRIFT_WINDOW_SIZE:
        return 0.0

    avg_drift = float(np.mean(_drift_scores))

    if avg_drift > DRIFT_THRESHOLD:
        DRIFT_DETECTED.set(1)
        _check_and_flag_retraining(reason="drift")
    else:
        DRIFT_DETECTED.set(0)

    return avg_drift


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Classify an uploaded skin lesion image as malignant or benign.

    Also:
    - Saves image to pending_feedback/ for retraining use
    - Computes drift score vs baseline
    - Returns image_id for use with /feedback endpoint
    """
    global _latency_sum, _latency_count

    start_time = time.time()

    model = get_model()

    if model is None:
        ERROR_COUNTER.labels(endpoint="/predict", error_type="model_not_loaded").inc()
        REQUEST_COUNT.labels(
            endpoint="/predict", method="POST", status_code="503"
        ).inc()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for the model to be ready.",
        )

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        ERROR_COUNTER.labels(endpoint="/predict", error_type="invalid_file_type").inc()
        REQUEST_COUNT.labels(
            endpoint="/predict", method="POST", status_code="400"
        ).inc()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only JPEG and PNG supported.",
        )

    try:
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise ValueError("Uploaded file is empty.")

        # ── Generate unique image_id ──────────────────────────────────────
        # Use hash of image bytes + timestamp for uniqueness
        image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        image_id = f"img_{image_hash}_{int(time.time())}.jpg"

        # ── Save image for potential retraining use ───────────────────────
        pending_path = PENDING_FEEDBACK_DIR / image_id
        with open(pending_path, "wb") as f:
            f.write(image_bytes)
        os.chmod(pending_path, 0o777)  # allow Airflow container to delete

        # ── Run inference ─────────────────────────────────────────────────
        meta = get_model_meta()
        result = predict(
            model=model,
            image_bytes=image_bytes,
            threshold=meta["threshold"],
            # preprocessing is fixed in predictor.py (copy of the training notebook)
        )

        # ── Drift detection ───────────────────────────────────────────────
        img_histogram = _compute_image_histogram(image_bytes)
        drift = _compute_drift_score(img_histogram)

        _drift_scores.append(drift)

        if len(_drift_scores) >= DRIFT_WINDOW_SIZE:
            avg_drift = _update_drift_and_check()
            DRIFT_SCORE.set(avg_drift)

        # ── Update Prometheus metrics ─────────────────────────────────────
        PREDICTION_COUNTER.labels(label=result["label"]).inc()
        CONFIDENCE_HISTOGRAM.labels(label=result["label"]).observe(result["confidence"])
        MALIGNANT_PROBABILITY.observe(result["malignant_prob"])

        latency = time.time() - start_time
        _latency_sum += latency
        _latency_count += 1
        avg_latency = _latency_sum / _latency_count
        AVG_LATENCY.labels(endpoint="/predict").set(avg_latency)

        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        REQUEST_COUNT.labels(
            endpoint="/predict", method="POST", status_code="200"
        ).inc()

        logger.info(
            "Prediction | image_id=%s | label=%s | latency=%.3fs | drift=%.4f",
            image_id,
            result["label"],
            latency,
            drift,
        )

        return PredictResponse(
            image_id=image_id,
            is_fallback=meta.get("is_fallback", False),   # UI adds a warning to the result
            **result,
        )

    except ValueError as e:
        ERROR_COUNTER.labels(endpoint="/predict", error_type="invalid_image").inc()
        REQUEST_COUNT.labels(
            endpoint="/predict", method="POST", status_code="400"
        ).inc()
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        ERROR_COUNTER.labels(endpoint="/predict", error_type="inference_error").inc()
        REQUEST_COUNT.labels(
            endpoint="/predict", method="POST", status_code="500"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        ERROR_COUNTER.labels(endpoint="/predict", error_type="unexpected_error").inc()
        REQUEST_COUNT.labels(
            endpoint="/predict", method="POST", status_code="500"
        ).inc()
        logger.error("Unexpected error in /predict: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error.")


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────
# This answers if the FastAPI server running and responding to requests, if FastAPI is unhealthy then this endpoint won't even work, that is why it is minimal
@app.get("/health", response_model=HealthResponse)
async def health():
    REQUEST_COUNT.labels(endpoint="/health", method="GET", status_code="200").inc()
    return HealthResponse(status="ok")


# ─────────────────────────────────────────────────────────────────────────────
# GET /ready
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/ready", response_model=ReadyResponse)
async def ready():
    reloaded = check_and_reload()
    if reloaded:
        logger.info("Model auto-reloaded at /ready check")

    model = get_model()
    meta = get_model_meta()

    REQUEST_COUNT.labels(endpoint="/ready", method="GET", status_code="200").inc()

    if model is None:
        # Return HTTP 503 (not 200) when no model is loaded.
        # The Kubernetes readinessProbe only looks at the status code: with a 200
        # the pod would be marked Ready and the backend Service would send
        # /predict traffic to a pod that can only answer "503 model not loaded".
        # With 503 the pod is removed from the Service endpoints until a model
        # appears in the registry (check_and_reload() above picks it up).
        return JSONResponse(
            status_code=503,
            content=ReadyResponse(
                model_loaded=False,
                model_name=None,
                model_version=None,
                classification_threshold=None,
                status="not_ready",
            ).model_dump(),
        )

    return ReadyResponse(
        model_loaded=True,
        model_name=meta.get("name"),
        model_version=meta.get("version"),
        classification_threshold=meta.get("threshold"),
        is_fallback=meta.get("is_fallback", False),
        status="ready",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /metrics
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ─────────────────────────────────────────────────────────────────────────────
# POST /feedback
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    """
    Submit ground truth label for a previous prediction.

    Also:
    - Moves saved image to feedback_data/<true_label>/ for retraining
    - Updates misclassification rate
    - Triggers retraining flag if rate exceeds threshold
    """

    try:
        predicted = request.predicted_label.lower()
        actual = request.true_label.lower()
        image_id = request.image_id

        valid_labels = {"malignant", "benign"}
        if predicted not in valid_labels or actual not in valid_labels:
            raise HTTPException(
                status_code=400, detail=f"Labels must be one of: {valid_labels}"
            )

        # ── Move image from pending to feedback_data ───────────────────────
        pending_path = PENDING_FEEDBACK_DIR / image_id
        feedback_path = FEEDBACK_DATA_DIR / actual / image_id

        if pending_path.exists():
            import shutil

            shutil.move(str(pending_path), str(feedback_path))
            os.chmod(feedback_path, 0o777)  # allow Airflow container to delete
            logger.info("Image moved to feedback_data/%s/%s", actual, image_id)
        else:
            logger.warning(
                "Pending image not found for image_id=%s "
                "(may have expired or already processed)",
                image_id,
            )

        # ── Update confusion matrix ───────────────────────────────────────
        if predicted == "malignant" and actual == "malignant":
            _feedback_counts["tp"] += 1
        elif predicted == "malignant" and actual == "benign":
            _feedback_counts["fp"] += 1
            MISCLASSIFICATION_COUNTER.labels(type="false_positive").inc()
        elif predicted == "benign" and actual == "malignant":
            _feedback_counts["fn"] += 1
            MISCLASSIFICATION_COUNTER.labels(type="false_negative").inc()
        else:
            _feedback_counts["tn"] += 1

        tp = _feedback_counts["tp"]
        fp = _feedback_counts["fp"]
        fn = _feedback_counts["fn"]
        tn = _feedback_counts["tn"]
        total_feedback = tp + fp + fn + tn

        # ── Update recall and precision gauges ────────────────────────────
        if (tp + fn) > 0:
            REAL_WORLD_RECALL.set(tp / (tp + fn))
        if (tp + fp) > 0:
            REAL_WORLD_PRECISION.set(tp / (tp + fp))

        # ── Compute and update misclassification rate ─────────────────────
        misclassifications = fp + fn
        if total_feedback > 0:
            rate = misclassifications / total_feedback
            MISCLASSIFICATION_RATE.set(rate)

            # ── Check retraining trigger ──────────────────────────────────
            if (
                total_feedback >= MIN_FEEDBACK_FOR_TRIGGER
                and rate > MISCLASSIFICATION_THRESHOLD
            ):
                logger.warning(
                    "Misclassification rate %.2f%% exceeds threshold %.2f%%",
                    rate * 100,
                    MISCLASSIFICATION_THRESHOLD * 100,
                )
                _check_and_flag_retraining(reason="misclassification")

        # ── Update feedback counter ───────────────────────────────────────
        FEEDBACK_COUNTER.labels(predicted=predicted, actual=actual).inc()

        # ── Log to JSONL ──────────────────────────────────────────────────
        log_entry = {
            "image_id": image_id,
            "predicted_label": predicted,
            "true_label": actual,
            "image_saved_to": str(feedback_path) if pending_path.exists() else None,
            "feedback_counts": _feedback_counts.copy(),
            "total_feedback": total_feedback,
        }
        with open(FEEDBACK_LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        REQUEST_COUNT.labels(
            endpoint="/feedback", method="POST", status_code="200"
        ).inc()

        logger.info(
            "Feedback | image=%s | predicted=%s | actual=%s | "
            "total_feedback=%d | misclassification_rate=%.4f",
            image_id,
            predicted,
            actual,
            total_feedback,
            (fp + fn) / total_feedback if total_feedback > 0 else 0.0,
        )

        return FeedbackResponse(received=True, message="Feedback recorded. Thank you.")

    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNTER.labels(endpoint="/feedback", error_type="unexpected_error").inc()
        logger.error("Error in /feedback: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error.")
