"""
model_loader.py
---------------
Phase 4 | Executed: Local (backend Docker container)

Loads the Production model from local MLflow registry at startup.
Caches model in memory — no reloading per request.

The mlruns/ folder is mounted as a Docker volume so the backend
always sees the latest Production model without rebuilding the container.

Key design: loads model directly from the mlruns/ folder using run_id
and artifact path — avoids Windows absolute path issues that occur when
training happens on Windows but serving happens in Linux Docker container.
"""

import os
import glob
import mlflow
import mlflow.pytorch
from pathlib import Path
from mlflow.tracking import MlflowClient

from logger import get_logger
from monitoring import MODEL_INFO, MODEL_LOAD_STATUS, MODEL_IS_FALLBACK

logger = get_logger("model_loader")

# ── Registry config ───────────────────────────────────────────────────────────
REGISTERED_MODEL_NAME = "melanoma_classifier"
MODEL_STAGE = "Production"

# ── Global model cache ────────────────────────────────────────────────────────
_model = None
_model_meta = {
    "version": None,
    "name": None,
    "run_id": None,
    "threshold": 0.35,
    # Input size the model was trained on. Read from the run's params/img_size
    # (logged by train.py and by register_pretrained.py). predictor.py resizes
    # every upload to img_size x img_size, so this MUST match training.
    # 64 is the fallback for the old simple_cnn runs.
    "img_size": 64,
    # True when the Production model is the ImageNet fallback (tags/is_fallback
    # == "true", written by register_pretrained.py). Models from train.py have no
    # such tag -> False.
    "is_fallback": False,
}


def _get_mlruns_path() -> str:
    """Get MLflow tracking URI from environment or default."""
    return os.environ.get("MLFLOW_TRACKING_URI", "/mlruns")


def _get_production_version() -> tuple:
    """
    Get the current Production model version and run_id from registry.

    Returns:
        tuple: (version, run_id) or (None, None) if no Production model.
    """
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(
            name=REGISTERED_MODEL_NAME,
            stages=[MODEL_STAGE]
        )
        if not versions:
            return None, None
        v = versions[0]
        return v.version, v.run_id
    except Exception as e:
        logger.error("Failed to query MLflow registry: %s", str(e))
        return None, None


def _find_model_path(run_id: str, mlruns_path: str) -> str:
    """
    Find the pytorch model directory directly in mlruns/ folder.

    Searches multiple possible locations because MLflow versions differ
    in where they store model artifacts:

    Location 1 (older MLflow):
        mlruns/<exp_id>/<run_id>/artifacts/pytorch_model

    Location 2 (newer MLflow 2.x):
        mlruns/<exp_id>/models/<model_hash>/artifacts/

    Args:
        run_id (str): MLflow run ID.
        mlruns_path (str): Path to mlruns/ folder inside container.

    Returns:
        str: Path to the model directory containing MLmodel file.

    Raises:
        FileNotFoundError: If model directory cannot be found.
    """
    mlruns = Path(mlruns_path)

    # ── Location 1: classic run artifacts ────────────────────────────────
    pattern1 = str(mlruns / "*" / run_id / "artifacts" / "pytorch_model")
    matches1 = glob.glob(pattern1)
    if matches1:
        logger.info("Found model (location 1): %s", matches1[0])
        return matches1[0]

    # ── Location 2: newer MLflow models/ folder ───────────────────────────
    # mlruns/<exp_id>/models/<model_hash>/artifacts/
    # Find by looking for MLmodel files and matching run_id in meta.yaml
    pattern2 = str(mlruns / "*" / "models" / "*" / "artifacts")
    matches2 = glob.glob(pattern2)

    for candidate in matches2:
        # Check meta.yaml in parent to confirm this is the right run
        meta_path = Path(candidate).parent / "meta.yaml"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    content = f.read()
                if run_id in content:
                    # Verify MLmodel file exists (confirms it's a valid model)
                    mlmodel_path = Path(candidate) / "MLmodel"
                    if mlmodel_path.exists():
                        logger.info("Found model (location 2): %s", candidate)
                        return candidate
            except Exception:
                continue

    # ── Location 3: search all MLmodel files as last resort ───────────────
    pattern3 = str(mlruns / "**" / "MLmodel")
    all_mlmodels = glob.glob(pattern3, recursive=True)

    for mlmodel_file in all_mlmodels:
        try:
            with open(mlmodel_file) as f:
                content = f.read()
            if run_id in content:
                model_dir = str(Path(mlmodel_file).parent)
                logger.info("Found model (location 3): %s", model_dir)
                return model_dir
        except Exception:
            continue

    raise FileNotFoundError(
        f"Could not find model for run_id={run_id} "
        f"in mlruns path={mlruns_path}. "
        f"Searched 3 locations. Check that mlruns/ is mounted correctly."
    )


# def _get_threshold_from_run(run_id: str) -> float:
#     """
#     Retrieve the best_threshold param logged during training.

#     Args:
#         run_id (str): MLflow run ID.

#     Returns:
#         float: Threshold value, or 0.35 as fallback.
#     """
#     try:
#         client = MlflowClient()
#         run = client.get_run(run_id)
#         threshold = run.data.params.get("best_threshold", None)
#         if threshold is not None:
#             return float(threshold)
#         logger.warning("No best_threshold in run %s. Using 0.35", run_id)
#         return 0.35
#     except Exception as e:
#         logger.warning(
#             "Could not fetch threshold from run %s: %s. Using 0.35",
#             run_id, str(e)
#         )
#         return 0.35

def _get_threshold_from_run(run_id: str) -> tuple:
    """
    Retrieve best_threshold and model_name directly from mlruns meta files.
    Avoids MLflow client version mismatch issues.

    Returns:
        tuple: (threshold, model_name, img_size)
    """
    import yaml
    mlruns_path = _get_mlruns_path()
    
    # Search for the run's meta.yaml and params directly in filesystem
    run_dirs = glob.glob(str(Path(mlruns_path) / "*" / run_id))
    
    for run_dir in run_dirs:
        try:
            # Read threshold from params file
            threshold_file = Path(run_dir) / "params" / "best_threshold"
            model_name_file = Path(run_dir) / "tags" / "model_name"
            
            img_size_file = Path(run_dir) / "params" / "img_size"

            threshold = 0.35
            model_name = "unknown"
            img_size = 64
            
            if threshold_file.exists():
                threshold = float(threshold_file.read_text().strip())
                logger.info("Threshold from file: %.4f", threshold)
            
            if model_name_file.exists():
                model_name = model_name_file.read_text().strip()
                logger.info("Model name from file: %s", model_name)

            if img_size_file.exists():
                img_size = int(img_size_file.read_text().strip())
                logger.info("Image size from file: %d", img_size)

            return threshold, model_name, img_size
        except Exception as e:
            logger.warning("Could not read run files: %s", e)
    
    return 0.35, "unknown", 64


# Photo types the model knows (the notebook's three). Each can have its own
# threshold, logged by the register Job / fine-tune as params/threshold_<key>.
IMAGE_TYPES = ["dermoscopic", "clinical: close-up", "clinical: overview"]


def threshold_param(image_type: str) -> str:
    """'clinical: close-up' -> 'threshold_clinical_close_up' (same rule as isic_notebook.py)."""
    import re
    return "threshold_" + re.sub(r"[^a-z0-9]+", "_", image_type.lower()).strip("_")


def _get_type_thresholds(run_id: str) -> dict:
    """Per-photo-type thresholds from params/threshold_<type>; missing types are left out."""
    out = {}
    for run_dir in glob.glob(str(Path(_get_mlruns_path()) / "*" / run_id)):
        for t in IMAGE_TYPES:
            f = Path(run_dir) / "params" / threshold_param(t)
            if f.exists():
                out[t] = float(f.read_text().strip())
    return out


def _get_is_fallback(run_id: str) -> bool:
    """Read tags/is_fallback straight from the mlruns folder (same style as above)."""
    for run_dir in glob.glob(str(Path(_get_mlruns_path()) / "*" / run_id)):
        tag_file = Path(run_dir) / "tags" / "is_fallback"
        if tag_file.exists():
            return tag_file.read_text().strip().lower() == "true"
    return False


def load_model() -> bool:
    """
    Load the Production model from MLflow registry into memory.

    Uses run_id to find model directly in mlruns/ folder,
    bypassing any Windows absolute paths stored in MLflow metadata.

    Returns:
        bool: True if model loaded successfully, False otherwise.
    """
    global _model, _model_meta

    mlruns_path = _get_mlruns_path()
    mlflow.set_tracking_uri(mlruns_path)

    version, run_id = _get_production_version()

    if version is None:
        logger.error(
            "No Production model found in MLflow registry. "
            "Train a model first."
        )
        MODEL_LOAD_STATUS.set(0)
        return False

    try:
        # ── Find model directly in mlruns/ folder ─────────────────────────
        # This avoids Windows path issues in stored artifact URIs
        model_path = _find_model_path(run_id, mlruns_path)

        logger.info(
            "Loading model from local path: %s (version %s)",
            model_path, version
        )

        # ── Load PyTorch model ────────────────────────────────────────────
        model = mlflow.pytorch.load_model(model_path, map_location="cpu")
        model.eval()

        #── Fetch threshold and model name ────────────────────────────────
        threshold, model_name_tag, img_size = _get_threshold_from_run(run_id)

        # ── Cache in memory ───────────────────────────────────────────────
        _model = model
        _model_meta["version"] = str(version)
        _model_meta["run_id"] = run_id
        _model_meta["threshold"] = threshold
        _model_meta["name"] = model_name_tag
        _model_meta["img_size"] = img_size
        _model_meta["is_fallback"] = _get_is_fallback(run_id)
        _model_meta["type_thresholds"] = _get_type_thresholds(run_id)
        logger.info("Per-type thresholds: %s (others use %.4f)", _model_meta["type_thresholds"], threshold)
        MODEL_IS_FALLBACK.set(1 if _model_meta["is_fallback"] else 0)
        if _model_meta["is_fallback"]:
            logger.warning(
                "⚠️ Serving the FALLBACK model (ImageNet EfficientNet-B1 + quick head). "
                "Copy your efficientnet_b1.pt into data-pvc and re-run the register-b1-model Job."
            )
        
        # ── Update Prometheus metrics ─────────────────────────────────────
        MODEL_INFO.info({
            "version": str(version),
            "run_id": run_id,
            "threshold": str(threshold),
            "img_size": str(img_size),
            "stage": MODEL_STAGE
        })
        MODEL_LOAD_STATUS.set(1)

        logger.info(
            "✅ Model loaded | version=%s | threshold=%.4f | img_size=%d | run_id=%s",
            version, threshold, img_size, run_id
        )
        return True

    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        MODEL_LOAD_STATUS.set(0)
        return False


def get_model():
    """Get the cached model instance."""
    return _model


def get_model_meta() -> dict:
    """Get metadata about the currently loaded model."""
    return _model_meta.copy()


def check_and_reload() -> bool:
    """
    Check if a new Production model version is available and reload if so.

    Returns:
        bool: True if model was reloaded, False if no change.
    """
    current_version = _model_meta.get("version")
    latest_version, _ = _get_production_version()

    if latest_version is None:
        return False

    if str(latest_version) != str(current_version):
        logger.info(
            "New Production model detected (version %s → %s). Reloading...",
            current_version, latest_version
        )
        load_model()
        
        return True

    return False