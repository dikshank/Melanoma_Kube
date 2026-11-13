"""
finetune.py
-----------
Retraining = fine-tune the CURRENT Production model on new feedback, with the
training notebook's recipe. It never trains from scratch and never uses the
whole ISIC dataset.

    start      : the Production model from MLflow (its exact weights)
    data       : feedback images (feedback batch folder: malignant/, benign/)
                 + the replay set (~1000 ISIC train images, so nothing is forgotten)
    recipe     : notebook augmentations (isic_notebook.train_tensor), BCE on the
                 malignant logit, AdamW (wd 0.05), warmup + cosine, 50/50 labels
    scoring    : champion (Production) and challenger (fine-tuned) on the SAME
                 eval set with the notebook's crop (isic_notebook.evaluate)
    promotion  : challenger replaces champion only if
                     spec@90%recall >= champion + margin   AND
                     AUC            >= champion - auc tolerance
                 otherwise it is registered in Staging for you to look at
    threshold  : the challenger's own 90%-recall point on the eval set

Called by the retraining DAG:
    python training/src/finetune.py --feedback-dir /opt/airflow/app_logs/feedback_batches/<run>
Writes a one-line JSON summary to --result-file for the DAG to log.
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isic_notebook as nb  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] finetune: %(message)s", force=True)
logger = logging.getLogger("finetune")

REGISTERED_MODEL_NAME = "melanoma_classifier"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def feedback_table(folder: Path) -> pd.DataFrame:
    rows = []
    for class_name, label in (("malignant", 1), ("benign", 0)):
        d = folder / class_name
        if d.exists():
            rows += [{"filepath": str(p), "label": label, "already_cached": False, "source": "feedback"}
                     for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    return pd.DataFrame(rows, columns=["filepath", "label", "already_cached", "source"])


def score(model, eval_table, batch_size):
    loader = DataLoader(nb.LesionDataset(eval_table, is_training=False), batch_size=batch_size, num_workers=0)
    return nb.predict_logits(model, loader)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feedback-dir", required=True, type=Path)
    ap.add_argument("--replay-manifest", default="data/isic_v1/manifests/replay.csv", type=Path)
    ap.add_argument("--eval-manifest", default="data/isic_v1/manifests/eval.csv", type=Path)
    ap.add_argument("--mlruns", default="/opt/airflow/mlruns")
    ap.add_argument("--experiment", default="finetune")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5, help="small: we start from a trained model")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--feedback-weight", type=float, default=3.0,
                    help="a feedback image is drawn this many times as often as a replay image")
    ap.add_argument("--margin", type=float, default=0.02, help="spec@90%%recall the challenger must gain")
    ap.add_argument("--auc-tolerance", type=float, default=0.005)
    ap.add_argument("--min-feedback", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2, help="CPU threads (scheduler pod limit is 2 CPUs)")
    ap.add_argument("--result-file", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    result = {"promoted": False}

    def finish(**kw):
        result.update(kw)
        logger.info("result: %s", result)
        if args.result_file:
            args.result_file.write_text(json.dumps(result))

    # ── data ─────────────────────────────────────────────────────────────
    for p in (args.replay_manifest, args.eval_manifest):
        if not p.exists():
            sys.exit(f"{p} not found - copy isic_v1 onto data-pvc (tools/isic_subset/prepare_isic_subset.py)")
    fb = feedback_table(args.feedback_dir)
    if len(fb) < args.min_feedback:
        logger.info("only %d feedback images in %s - nothing new to learn, skipping", len(fb), args.feedback_dir)
        return finish(reason="no feedback images", feedback=len(fb))
    replay = nb.read_manifest(args.replay_manifest).assign(source="replay")
    eval_table = nb.read_manifest(args.eval_manifest)
    train_table = pd.concat([fb, replay[["filepath", "label", "already_cached", "source"]]], ignore_index=True)
    logger.info("train: %d feedback (%d malignant) + %d replay; eval: %d (%d malignant)",
                len(fb), int(fb["label"].sum()), len(replay), len(eval_table), int(eval_table["label"].sum()))

    # ── champion = current Production model ──────────────────────────────
    mlflow.set_tracking_uri(args.mlruns)
    client = MlflowClient()
    prod = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    if not prod:
        sys.exit("no Production model to fine-tune")
    champ_version, champ_run_id = prod[0].version, prod[0].run_id
    champ_run = client.get_run(champ_run_id)
    model_name = champ_run.data.tags.get("model_name", "unknown")
    model = mlflow.pytorch.load_model(f"runs:/{champ_run_id}/pytorch_model", map_location="cpu")
    logger.info("champion: v%s (%s, run %s)", champ_version, model_name, champ_run_id)

    y_eval = eval_table["label"].values
    champ = nb.evaluate(y_eval, score(model, eval_table, 32))
    logger.info("champion on eval: AUC %.4f, spec@90%%recall %.4f", champ["auc"], champ["spec_at_90recall"])

    # ── fine-tune (notebook recipe) ──────────────────────────────────────
    labels = train_table["label"].values
    weight = np.where(train_table["source"] == "feedback", args.feedback_weight, 1.0)
    for lab in (0, 1):                                      # 50/50 malignant / benign
        m = labels == lab
        if m.any():
            weight[m] = weight[m] / weight[m].sum()
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(torch.tensor(weight, dtype=torch.double), num_samples=len(train_table),
                                    replacement=True, generator=generator)
    train_loader = DataLoader(nb.LesionDataset(train_table, is_training=True), batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, drop_last=len(train_table) > args.batch_size)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(1, len(train_loader))

    def lr_factor(step):
        return min(1.0, (step + 1) / warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for step, (x, yb) in enumerate(train_loader, 1):
            loss = F.binary_cross_entropy_with_logits(nb.malignant_logit(model, x), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            if step % 20 == 0:
                logger.info("epoch %d/%d step %d/%d loss %.4f", epoch + 1, args.epochs, step,
                            len(train_loader), loss_sum / step)
        logger.info("epoch %d/%d done, mean loss %.4f", epoch + 1, args.epochs, loss_sum / len(train_loader))
    model.eval()

    # ── challenger on the same eval set ──────────────────────────────────
    chal = nb.evaluate(y_eval, score(model, eval_table, 32))
    threshold = chal["threshold_at_90recall"]
    chal_at_t = nb.evaluate(y_eval, score(model, eval_table, 32), threshold_prob=threshold)
    better = (chal["spec_at_90recall"] >= champ["spec_at_90recall"] + args.margin
              and chal["auc"] >= champ["auc"] - args.auc_tolerance)
    logger.info("challenger on eval: AUC %.4f, spec@90%%recall %.4f, threshold %.4f -> %s",
                chal["auc"], chal["spec_at_90recall"], threshold, "PROMOTE" if better else "keep champion")

    # ── register ─────────────────────────────────────────────────────────
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"{model_name}-finetune-from-v{champ_version}") as run:
        mlflow.log_params({
            "model_name": model_name, "img_size": nb.IMG_SIZE, "best_threshold": threshold,
            "base_version": champ_version, "base_run_id": champ_run_id,
            "feedback_images": len(fb), "replay_images": len(replay),
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "weight_decay": args.weight_decay, "feedback_weight": args.feedback_weight,
            "preprocess": "notebook: short320 + jpeg95 -> resize256 -> crop224",
        })
        mlflow.set_tags({"model_name": model_name, "is_fallback": "false", "experiment_type": "finetune",
                         "promotion_result": "promoted" if better else "kept_champion"})
        mlflow.log_metrics({**{f"val_{k}": v for k, v in chal_at_t.items() if k != "threshold" and v == v},
                            **{f"champion_{k}": v for k, v in champ.items() if k != "threshold" and v == v}})
        mlflow.pytorch.log_model(model, artifact_path="pytorch_model", registered_model_name=REGISTERED_MODEL_NAME)
        run_id = run.info.run_id

    version = max(int(v.version) for v in client.search_model_versions(f"run_id='{run_id}'"))
    client.transition_model_version_stage(REGISTERED_MODEL_NAME, version,
                                          stage="Production" if better else "Staging",
                                          archive_existing_versions=better)
    logger.info("registered v%s -> %s", version, "Production" if better else "Staging")
    finish(promoted=better, version=version, run_id=run_id, base_version=champ_version,
           feedback=len(fb), champion_auc=champ["auc"], champion_spec90=champ["spec_at_90recall"],
           challenger_auc=chal["auc"], challenger_spec90=chal["spec_at_90recall"], threshold=threshold)


if __name__ == "__main__":
    main()
