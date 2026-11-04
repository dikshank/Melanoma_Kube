"""
register_pretrained.py
----------------------
Kubernetes | Executed: inside the `register-b1-model` Job (airflow-image:v1)

Registers an ALREADY-TRAINED model file (your EfficientNet-B1 .pt) into the
MLflow Model Registry as `melanoma_classifier` and moves it to the
"Production" stage, so the FastAPI backend loads it on startup.

Why this script exists
    In docker-compose, mlruns/ was a folder on your laptop that already had
    models in it. In Kubernetes, mlruns lives on a fresh, empty PVC
    (mlruns-pvc). The backend asks the registry "what is in Production?" and
    gets nothing, so /ready stays 503 forever. This script fills that gap
    without retraining anything.

Where the data comes from / goes to (inside the Job pod)
    /opt/airflow/data/seed_model/<file>.pt   <- you `kubectl cp` it here (data-pvc)
    /opt/airflow/data/processed/v1/...       <- written by the data pipeline DAG (data-pvc)
    /opt/airflow/mlruns                      -> MLflow file store (mlruns-pvc)
                                                the backend mounts the SAME PVC at /mlruns

What it logs (the backend and the retraining DAG read these back)
    params/img_size        -> backend resizes uploads to this (224 for your B1)
    params/best_threshold  -> backend's malignant cut-off
    tags/model_name        -> shown in the UI; retraining DAG picks the config by it
    metrics/val_recall     -> champion/challenger: a retrained model must BEAT this
    metrics/val_f1            to replace your B1 (see mlflow_utils.log_model)

Supported .pt formats (you weren't sure which one you have, so all are handled)
    1. TorchScript         torch.jit.save(torch.jit.script(model), "x.pt")
    2. Whole model         torch.save(model, "x.pt")
    3. state_dict          torch.save(model.state_dict(), "x.pt")
    4. Checkpoint dict     torch.save({"model_state_dict": ..., "epoch": ...}, "x.pt")
       (also "state_dict" / "model" keys, and "module." prefixes from DataParallel)

Usage
    python training/src/register_pretrained.py \
        --weights /opt/airflow/data/seed_model/efficientnet_b1.pt \
        --img-size 224
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader
from torchvision import models

# train.py / evaluate.py / dataset.py live next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import MelanomaDataset            # noqa: E402  same transforms as training
from evaluate import get_predictions, tune_threshold, compute_metrics  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,   # mlflow/torch imports configure logging first; without this our INFO lines vanish
)
logger = logging.getLogger("register_pretrained")

REGISTERED_MODEL_NAME = "melanoma_classifier"   # must match backend/model_loader.py
NUM_CLASSES = 2                                 # 0 = benign, 1 = malignant


# ─────────────────────────────────────────────────────────────────────────────
# Loading the .pt file
# ─────────────────────────────────────────────────────────────────────────────

def _build_b1(num_classes: int) -> nn.Module:
    """Empty EfficientNet-B1 with a `num_classes` head (no ImageNet download)."""
    model = models.efficientnet_b1(weights=None)
    in_features = model.classifier[1].in_features          # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _clean_state_dict(sd: dict) -> dict:
    """Strip 'module.' (DataParallel) / '_orig_mod.' (torch.compile) prefixes."""
    cleaned = {}
    for k, v in sd.items():
        for prefix in ("module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v
    return cleaned


def load_any_checkpoint(path: Path) -> nn.Module:
    """
    Load a .pt/.pth file in whichever of the 4 formats it was saved in
    and return a ready-to-use nn.Module in eval mode.
    """
    # ── 1. TorchScript? (a zip archive with code + weights) ────────────────
    try:
        model = torch.jit.load(str(path), map_location="cpu")
        logger.info("Loaded as TorchScript module")
        return model.eval()
    except Exception:
        pass  # not TorchScript, fall through to torch.load

    # weights_only=False because formats 2 and 4 contain pickled Python objects.
    # Only do this with files you trust (your own training output).
    obj = torch.load(str(path), map_location="cpu", weights_only=False)

    # ── 2. Whole pickled model ────────────────────────────────────────────
    if isinstance(obj, nn.Module):
        logger.info("Loaded as whole pickled model: %s", type(obj).__name__)
        return obj.eval()

    if not isinstance(obj, dict):
        raise ValueError(f"Don't know how to load object of type {type(obj)}")

    # ── 4. Checkpoint dict → dig out the real state_dict / model ──────────
    for key in ("model_state_dict", "state_dict", "model"):
        if key in obj:
            inner = obj[key]
            if isinstance(inner, nn.Module):
                logger.info("Checkpoint dict, key '%s' holds a whole model", key)
                return inner.eval()
            logger.info("Checkpoint dict, using state_dict from key '%s'", key)
            obj = inner
            break

    # ── 3. Plain state_dict → rebuild B1 and load weights into it ────────
    sd = _clean_state_dict(obj)
    head_key = "classifier.1.weight"
    if head_key not in sd:
        raise ValueError(
            f"'{head_key}' not found - this does not look like a torchvision "
            f"efficientnet_b1 state_dict. First keys: {list(sd)[:5]}"
        )
    num_classes = sd[head_key].shape[0]       # infer head size from the weights
    model = _build_b1(num_classes)
    model.load_state_dict(sd, strict=True)    # strict: fail loudly on mismatch
    logger.info("Loaded state_dict into torchvision efficientnet_b1 (%d classes)", num_classes)
    return model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (so the retraining DAG has numbers to compare against)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_split(model: nn.Module, manifest: Path, img_size: int, split: str):
    """Return (labels, malignant_probs) for one manifest, or None if missing."""
    if not manifest.exists():
        logger.warning("No %s manifest at %s - skipping %s metrics", split, manifest, split)
        return None
    ds = MelanomaDataset(str(manifest), img_size=img_size, split=split)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    labels, probs, _ = get_predictions(model, dl, device="cpu")
    return labels, probs


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def already_registered(client: MlflowClient, digest: str) -> bool:
    """True if a Production version was already created from this exact file."""
    try:
        for v in client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"]):
            run = client.get_run(v.run_id)
            if run.data.tags.get("weights_sha256") == digest:
                logger.info("Version %s already in Production from this file - nothing to do", v.version)
                return True
    except Exception:
        pass  # registered model doesn't exist yet
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, type=Path, help=".pt/.pth file to register")
    p.add_argument("--model-name", default="efficientnet_b1", help="value of tags/model_name")
    p.add_argument("--img-size", type=int, default=224, help="input size the model was trained on")
    p.add_argument("--threshold", type=float, default=None,
                   help="fixed malignant threshold; default = tune on val set (or 0.5 if no val set)")
    p.add_argument("--mlruns", default="/opt/airflow/mlruns", help="MLflow tracking dir (mlruns-pvc)")
    p.add_argument("--manifest-dir", default="data/processed/v1/manifests", type=Path)
    p.add_argument("--experiment", default="pretrained-models")
    p.add_argument("--no-promote", action="store_true", help="register only, don't move to Production")
    args = p.parse_args()

    if not args.weights.exists():
        sys.exit(f"Weights file not found: {args.weights}\n"
                 f"Copy it in first (see K8S_RUNBOOK.md, step 'Load data + model').")

    mlflow.set_tracking_uri(args.mlruns)
    client = MlflowClient()

    digest = sha256_of(args.weights)
    if already_registered(client, digest):
        return  # Job is safe to re-run

    # ── Load + sanity check ───────────────────────────────────────────────
    model = load_any_checkpoint(args.weights)
    with torch.no_grad():
        out = model(torch.zeros(1, 3, args.img_size, args.img_size))
    if out.shape != (1, NUM_CLASSES):
        sys.exit(f"Model output shape {tuple(out.shape)} != (1, {NUM_CLASSES}). "
                 f"Backend expects 2 logits: [benign, malignant].")
    logger.info("Sanity check OK: input 1x3x%dx%d -> output %s", args.img_size, args.img_size, tuple(out.shape))

    # ── Evaluate on val (tunes threshold) and test (reporting only) ───────
    val = evaluate_split(model, args.manifest_dir / "val_manifest.csv", args.img_size, "val")
    test = evaluate_split(model, args.manifest_dir / "test_manifest.csv", args.img_size, "test")

    if args.threshold is not None:
        threshold = args.threshold
    elif val is not None:
        threshold, _, _ = tune_threshold(*val)      # same recall-first rule as training
    else:
        threshold = 0.5
    logger.info("Using malignant threshold = %.4f", threshold)

    # ── Log run + register model ──────────────────────────────────────────
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"{args.model_name}-pretrained") as run:
        mlflow.log_params({
            "model_name": args.model_name,
            "img_size": args.img_size,          # <- backend reads this
            "best_threshold": threshold,        # <- backend reads this
            "num_classes": NUM_CLASSES,
            "source": "pretrained_file",
        })
        mlflow.set_tags({
            "model_name": args.model_name,      # <- backend + retraining DAG read this
            "experiment_type": "pretrained",
            "weights_file": args.weights.name,
            "weights_sha256": digest,           # makes the Job idempotent
        })
        for split, res in (("val", val), ("test", test)):
            if res is None:
                continue
            m = compute_metrics(*res, threshold=threshold)
            # val_recall / val_f1 are what the champion/challenger check compares
            mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items() if k != "threshold"})
            logger.info("%s metrics: %s", split, m)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="pytorch_model",      # backend's _find_model_path looks here first
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        run_id = run.info.run_id

    # ── Promote to Production (archive whatever was there before) ─────────
    version = max(int(v.version) for v in client.search_model_versions(f"run_id='{run_id}'"))
    if args.no_promote:
        logger.info("Registered as version %s (not promoted)", version)
        return
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info("✅ %s registered as %s v%s and moved to Production (run_id=%s)",
                args.model_name, REGISTERED_MODEL_NAME, version, run_id)


if __name__ == "__main__":
    main()
