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

FALLBACK (when your .pt is missing)
    Instead of failing, the Job registers a clearly-labelled FALLBACK model so
    the app still runs:  tags/is_fallback = "true", model_name =
    "efficientnet_b1_fallback". The backend passes is_fallback to the UI, which
    shows a yellow "fallback model" banner.

    The fallback is EfficientNet-B1 with the ORIGINAL ImageNet weights (baked
    into airflow-image by the Dockerfile) + a new 2-class head. A bare ImageNet
    head would output 1000 ImageNet classes, and a random 2-class head would be
    a coin flip, so the head is trained here in a few seconds:
        frozen B1 backbone -> 1280-d features for every train image (once)
        -> train only Linear(1280, 2) on those features (class weights 1:2)
    It's weak (backbone never adapted to skin lesions) but it's a real model.

    Where the fallback comes from, in order:
      1. airflow/training/fallback/efficientnet_b1_fallback.pt   (committed in
         git - .gitignore has an exception - and copied into the image)
      2. otherwise: build it as above, and save it to
         /opt/airflow/data/seed_model/efficientnet_b1_fallback.pt so you can
         `kubectl cp` it out and commit it as (1) for next time.

    Safety rule: the fallback is ONLY promoted if nothing is in Production yet.
    It never replaces your real B1 or a retrained model. Registering your real
    B1 later always replaces the fallback.

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
FALLBACK_MODEL_NAME = "efficientnet_b1_fallback"
# committed copy, inside the image at /opt/airflow/training/fallback/
BUNDLED_FALLBACK = Path(__file__).resolve().parents[1] / "fallback" / "efficientnet_b1_fallback.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Loading the .pt file
# ─────────────────────────────────────────────────────────────────────────────

# Architectures a plain state_dict can be loaded into. B0 and B1 have the
# same head (Linear 1280 -> n) but a different number of blocks, so the
# state_dict only fits one of them when loaded with strict=True.
# To support another torchvision model, add it here.
ARCH_BUILDERS = {
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
}


def _build_efficientnet(arch: str, num_classes: int) -> nn.Module:
    """Empty EfficientNet (b0/b1) with a `num_classes` head (no ImageNet download)."""
    model = ARCH_BUILDERS[arch](weights=None)
    in_features = model.classifier[1].in_features          # 1280 for both b0 and b1
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _build_b1(num_classes: int) -> nn.Module:
    """Empty EfficientNet-B1 (used by the fallback)."""
    return _build_efficientnet("efficientnet_b1", num_classes)


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

    # ── 3. Plain state_dict → rebuild the architecture, load weights into it ─
    sd = _clean_state_dict(obj)
    head_key = "classifier.1.weight"
    if head_key not in sd:
        raise ValueError(
            f"'{head_key}' not found - this does not look like a torchvision "
            f"EfficientNet state_dict. First keys: {list(sd)[:5]}"
        )
    num_classes = sd[head_key].shape[0]       # infer head size from the weights

    # Try each known architecture; strict=True means only the right one fits
    # (B0 and B1 differ in block count, so the wrong one has missing/extra keys).
    errors = {}
    for arch in ARCH_BUILDERS:
        model = _build_efficientnet(arch, num_classes)
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            errors[arch] = str(e).splitlines()[0]
            continue
        logger.info("Loaded state_dict into torchvision %s (%d classes)", arch, num_classes)
        return model.eval()

    raise ValueError(
        "state_dict does not match any known architecture "
        f"({', '.join(ARCH_BUILDERS)}): {errors}"
    )


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


def already_registered(client: MlflowClient, digest: str, threshold: float = None) -> bool:
    """True if a Production version was already created from this exact file
    (and, when --threshold is given, with that same threshold)."""
    try:
        for v in client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"]):
            run = client.get_run(v.run_id)
            if run.data.tags.get("weights_sha256") != digest:
                continue
            old = run.data.params.get("best_threshold")
            if threshold is not None and (old is None or abs(float(old) - threshold) > 1e-9):
                logger.info("Same file is in Production (v%s) but with threshold %s -> "
                            "registering again with threshold %s", v.version, old, threshold)
                return False
            logger.info("Version %s already in Production from this file - nothing to do", v.version)
            return True
    except Exception:
        pass  # registered model doesn't exist yet
    return False


def production_exists(client: MlflowClient) -> bool:
    try:
        return bool(client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"]))
    except Exception:
        return False   # registered model doesn't exist yet


# ─────────────────────────────────────────────────────────────────────────────
# Fallback model: ImageNet EfficientNet-B1 + head trained on our features
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(backbone: nn.Module, manifest: Path, img_size: int):
    """Run every image through the frozen backbone once -> (N, 1280) features."""
    # split="val" = no augmentation, just resize + ImageNet normalisation
    ds = MelanomaDataset(str(manifest), img_size=img_size, split="val")
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            f = backbone.avgpool(backbone.features(x))   # same path as EfficientNet.forward
            feats.append(torch.flatten(f, 1))
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def build_fallback(img_size: int, manifest_dir: Path, epochs: int = 300) -> nn.Module:
    """EfficientNet-B1 (ImageNet) with a 2-class head trained on the train split."""
    from torchvision.models import EfficientNet_B1_Weights

    train_manifest = manifest_dir / "train_manifest.csv"
    if not train_manifest.exists():
        sys.exit(f"Can't build fallback: {train_manifest} missing. Copy data/processed "
                 f"into data-pvc first (K8S_RUNBOOK.md step 2) or run the data pipeline DAG.")

    # Loaded from TORCH_HOME (baked into the image) - no internet needed
    model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT).eval()
    for prm in model.parameters():
        prm.requires_grad = False

    logger.info("Fallback: extracting ImageNet features for %s ...", train_manifest)
    X, y = _extract_features(model, train_manifest, img_size)
    logger.info("Fallback: %d train images -> features %s", len(y), tuple(X.shape))

    torch.manual_seed(42)                                   # reproducible head
    head = nn.Linear(X.shape[1], NUM_CLASSES)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    # same class weights as the training configs: missing a melanoma costs more
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]))
    for epoch in range(epochs):                             # full-batch: 164 rows, takes seconds
        opt.zero_grad()
        loss = loss_fn(head(X), y)
        loss.backward()
        opt.step()
    logger.info("Fallback: head trained, final loss %.4f", loss.item())

    model.classifier[1] = head
    for prm in model.parameters():
        prm.requires_grad = False
    return model.eval()


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
    p.add_argument("--fallback-export", type=Path,
                   default=Path("/opt/airflow/data/seed_model/efficientnet_b1_fallback.pt"),
                   help="where a freshly built fallback is saved so you can commit it")
    args = p.parse_args()

    mlflow.set_tracking_uri(args.mlruns)
    client = MlflowClient()

    # ── Decide: real model or fallback ────────────────────────────────────
    is_fallback = not args.weights.exists()
    if is_fallback:
        logger.warning("⚠️  Weights file not found: %s", args.weights)
        if production_exists(client):
            # a real/retrained model (or an earlier fallback) is already serving
            logger.info("A Production model already exists - keeping it, no fallback needed.")
            return
        logger.warning("⚠️  No Production model either -> registering the FALLBACK model "
                       "(the UI will show a warning banner).")
        args.model_name = FALLBACK_MODEL_NAME
        if BUNDLED_FALLBACK.exists():
            logger.info("Using committed fallback: %s", BUNDLED_FALLBACK)
            model = load_any_checkpoint(BUNDLED_FALLBACK)
            digest = sha256_of(BUNDLED_FALLBACK)
        else:
            model = build_fallback(args.img_size, args.manifest_dir)
            try:
                args.fallback_export.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.fallback_export)
                digest = sha256_of(args.fallback_export)
                logger.info("Saved fallback to %s - copy it to airflow/training/fallback/ "
                            "and commit it to skip this step next time.", args.fallback_export)
            except OSError as e:          # e.g. data-pvc mounted read-only
                logger.warning("Could not save fallback (%s) - continuing anyway", e)
                digest = "built-in-job"
    else:
        digest = sha256_of(args.weights)
        if already_registered(client, digest, args.threshold):
            return  # Job is safe to re-run
        model = load_any_checkpoint(args.weights)

    # ── Sanity check ──────────────────────────────────────────────────────
    with torch.no_grad():
        out = model(torch.zeros(1, 3, args.img_size, args.img_size))
    if out.shape == (1, 1):
        # Single-logit model (trained with sigmoid / BCEWithLogitsLoss): the one
        # output z is the malignant logit. The backend expects 2 logits and
        # takes softmax(...)[1]. Append a FIXED layer z -> [0, z]:
        #     softmax([0, z])[1] = e^z / (1 + e^z) = sigmoid(z)
        # so predictions are exactly the same as the original model.
        # Only standard torch layers (Sequential + Linear) are used, so the
        # pickled model loads in the backend without any custom class.
        to_two = nn.Linear(1, 2)
        with torch.no_grad():
            to_two.weight.copy_(torch.tensor([[0.0], [1.0]]))
            to_two.bias.zero_()
        to_two.requires_grad_(False)
        model = nn.Sequential(model, to_two).eval()
        logger.info("Single-logit model: added fixed layer z -> [0, z] "
                    "(softmax malignant prob == sigmoid(z))")
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
            "source": "imagenet_fallback" if is_fallback else "pretrained_file",
        })
        mlflow.set_tags({
            "model_name": args.model_name,      # <- backend + retraining DAG read this
            "experiment_type": "pretrained",
            "weights_file": "efficientnet_b1_fallback.pt" if is_fallback else args.weights.name,
            "weights_sha256": digest,           # makes the Job idempotent
            # Always written, "true" or "false". backend/model_loader.py reads it and
            # /ready + /predict always return is_fallback -> yellow banner in the UI.
            "is_fallback": "true" if is_fallback else "false",
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
    logger.info("✅ %s registered as %s v%s and moved to Production (run_id=%s)%s",
                args.model_name, REGISTERED_MODEL_NAME, version, run_id,
                "  [FALLBACK - copy your efficientnet_b1.pt in and re-run this Job]" if is_fallback else "")


if __name__ == "__main__":
    main()
