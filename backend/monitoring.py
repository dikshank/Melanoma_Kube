"""
monitoring.py
-------------
Phase 4 | Executed: Local (backend Docker container)

Prometheus metrics definitions for the FastAPI backend.

All metrics are scraped by Prometheus at GET /metrics endpoint
and visualized in Grafana dashboards.

Metrics tracked:
    - request_count           : Total requests per endpoint and status
    - request_latency         : Response time histogram per endpoint
    - prediction_counter      : Count of malignant vs benign predictions
    - confidence_histogram    : Distribution of model confidence scores
    - error_counter           : Count of errors per endpoint
    - feedback_counter        : Count of feedback submissions
    - real_world_recall       : Running recall computed from feedback data
    - real_world_precision    : Running precision from feedback data
    - misclassification_counter: Count of wrong predictions from feedback
    - misclassification_rate  : Current misclassification rate gauge
    - drift_score             : Pixel distribution drift vs baseline
    - retraining_triggered    : Counter for retraining events
    - model_info              : Gauge with current model version info
"""

from prometheus_client import Counter, Histogram, Gauge, Info

# ── Request metrics ───────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    name="melanoma_request_total",
    documentation="Total number of requests per endpoint and HTTP status",
    labelnames=["endpoint", "method", "status_code"]
)

REQUEST_LATENCY = Histogram(
    name="melanoma_request_latency_seconds",
    documentation="Request latency in seconds per endpoint",
    labelnames=["endpoint"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

AVG_LATENCY = Gauge(
    name = "melanoma_avg_latency_seconds",
    documentation  ="Average prediction latency",
    labelnames = ["endpoint"]
)

# ── Prediction metrics ────────────────────────────────────────────────────────
PREDICTION_COUNTER = Counter(
    name="melanoma_prediction_total",
    documentation="Total predictions by class label",
    labelnames=["label"]
)

CONFIDENCE_HISTOGRAM = Histogram(
    name="melanoma_confidence_score",
    documentation="Distribution of model confidence scores",
    labelnames=["label"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

MALIGNANT_PROBABILITY = Histogram(
    name="melanoma_malignant_probability",
    documentation="Raw malignant class probability distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# ── Error metrics ─────────────────────────────────────────────────────────────
ERROR_COUNTER = Counter(
    name="melanoma_error_total",
    documentation="Total errors per endpoint and error type",
    labelnames=["endpoint", "error_type"]
)

# ── Feedback and real-world recall ────────────────────────────────────────────
FEEDBACK_COUNTER = Counter(
    name="melanoma_feedback_total",
    documentation="Total feedback submissions",
    labelnames=["predicted", "actual"]
)

REAL_WORLD_RECALL = Gauge(
    name="melanoma_real_world_recall",
    documentation="Running recall computed from feedback ground truth labels"
)

REAL_WORLD_PRECISION = Gauge(
    name="melanoma_real_world_precision",
    documentation="Running precision computed from feedback ground truth labels"
)

# ── Misclassification metrics ─────────────────────────────────────────────────
MISCLASSIFICATION_COUNTER = Counter(
    name="melanoma_misclassification_total",
    documentation="Total misclassifications from feedback (FP + FN)",
    labelnames=["type"]   # 'false_positive' or 'false_negative'
)

MISCLASSIFICATION_RATE = Gauge(
    name="melanoma_misclassification_rate",
    documentation=(
        "Current misclassification rate from feedback. "
        "Retraining triggered when this exceeds threshold."
    )
)

# ── Drift metrics ─────────────────────────────────────────────────────────────
DRIFT_SCORE = Gauge(
    name="melanoma_drift_score",
    documentation=(
        "Pixel distribution drift score vs training baseline. "
        "Computed as mean absolute difference of pixel histograms. "
        "Retraining triggered when this exceeds threshold."
    )
)

DRIFT_DETECTED = Gauge(
    name="melanoma_drift_detected",
    documentation="1 if drift exceeds threshold, 0 otherwise"
)

# ── Retraining metrics ────────────────────────────────────────────────────────
RETRAINING_TRIGGERED = Counter(
    name="melanoma_retraining_triggered_total",
    documentation="Number of times retraining has been triggered",
    labelnames=["reason"]   # 'misclassification' or 'drift'
)

# ── Model info ────────────────────────────────────────────────────────────────
MODEL_INFO = Info(
    name="melanoma_model",
    documentation="Information about the currently loaded model"
)

MODEL_LOAD_STATUS = Gauge(
    name="melanoma_model_loaded",
    documentation="1 if model is loaded successfully, 0 otherwise"
)


