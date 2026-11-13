"""
threshold_from_npz.py - the 90%-recall threshold of the served model, from the
val logits the training notebook saved (p2a_..._s0.npz), labelled with the ISIC
metadata.csv.

The .npz holds, for every epoch, the val logits (L), which epoch's weights were
saved (best_epoch) and the isic_id of every val image. This script:
  1. checks which label the notebook trained on, by printing the ROC AUC per
     photo type for two candidate labels; the one that reproduces the
     notebook's numbers (dermoscopic 0.959, close-up 0.991, overview 0.870)
     is the training label
        malignant : diagnosis_1 Malignant (1) vs Benign (0)
        melanoma  : diagnosis_2/3 mentions melanoma (1), every other image (0)
  2. gives ONE threshold (the app does not know the photo type): the score at
     which 90% of the positives in all of val are flagged, as a probability
  3. shows recall / specificity per photo type at that threshold, and an
     honest check: threshold from one patient half of val (the notebook's
     halves, seed 7), applied to the other half

Usage (cmd, from Melanoma_Kube):
    python tools\\eval_api\\threshold_from_npz.py p2a_efficientnet_b0_imagenet_full_15ep__s0.npz C:\\Users\\Dikshank\\Documents\\ISIC-images\\metadata.csv
"""

import csv
import sys

import numpy as np

TARGET_RECALL = 0.90
SPLIT_SEED = 7          # the notebook's patient halves


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def weighted_quantile(values, quantile):
    """The notebook's weighted_quantile() with every weight = 1."""
    order = np.argsort(values)
    cumulative = np.cumsum(np.ones(len(values))) / len(values)
    return float(np.interp(quantile, cumulative, values[order]))


def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.empty(len(s))
    order = np.argsort(s, kind="mergesort")
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                      # average ranks for ties
        tie = s == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    npz_path, meta_path = sys.argv[1], sys.argv[2]

    d = np.load(npz_path, allow_pickle=True)
    best = int(d["best_epoch"])
    logits_all = d["L"]
    ids = [str(i) for i in d["isic_id"]]
    print(f"{npz_path}: {logits_all.shape[0]} epochs x {logits_all.shape[1]} val images; "
          f"saved weights = epoch {best + 1}")

    meta = {}
    with open(meta_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            meta[r["isic_id"]] = r
    keep = np.array([i in meta for i in ids])
    print(f"val images found in metadata.csv: {keep.sum()} of {len(ids)}")
    if keep.sum() == 0:
        sys.exit("none of the val images are in this metadata.csv")

    z = logits_all[best][keep].astype(np.float64)
    rows = [meta[i] for i, k in zip(ids, keep) if k]
    image_type = np.array([r.get("image_type", "") for r in rows])
    patient = np.array([r.get("patient_id") or r["isic_id"] for r in rows])
    d1 = np.array([(r.get("diagnosis_1") or "").strip() for r in rows])
    text = [f'{r.get("diagnosis_2", "")} {r.get("diagnosis_3", "")}'.lower() for r in rows]
    labels = {
        "malignant": (np.isin(d1, ["Benign", "Malignant"]), (d1 == "Malignant").astype(int)),
        "melanoma": (np.ones(len(rows), bool), np.array([int("melanoma" in t) for t in text])),
    }
    types = [t for t in ["dermoscopic", "clinical: close-up", "clinical: overview"] if (image_type == t).any()]

    # ── 1. which label reproduces the notebook? ──────────────────────────
    print("\n1) ROC AUC per photo type (notebook: dermoscopic 0.959, close-up 0.991, overview 0.870)")
    print(f"   {'label':12s}" + "".join(f"{t:>22s}" for t in types))
    for name, (usable, y) in labels.items():
        cells = [auc(y[usable & (image_type == t)], z[usable & (image_type == t)]) for t in types]
        print(f"   {name:12s}" + "".join(f"{c:22.3f}" for c in cells))
    print("   -> use the label whose row matches the notebook")

    # ── 2 + 3. threshold for each label ─────────────────────────────────
    order_patients = list(dict.fromkeys(patient))         # first-appearance order, like pandas unique()
    half_a = set(np.random.default_rng(SPLIT_SEED).permutation(order_patients)[: len(order_patients) // 2])
    in_a = np.array([p in half_a for p in patient])

    def line(name, t_logit, mask, y):
        flagged = z >= t_logit
        mel, ben = mask & (y == 1), mask & (y == 0)
        rec = flagged[mel].mean() if mel.any() else float("nan")
        spec = (~flagged[ben]).mean() if ben.any() else float("nan")
        prec = (y[mask & flagged] == 1).mean() if (mask & flagged).any() else float("nan")
        print(f"   {name:34s} n {mask.sum():6d}  positives {mel.sum():5d}  "
              f"recall {rec:.3f}  specificity {spec:.3f}  precision {prec:.3f}")

    for name, (usable, y) in labels.items():
        pos = usable & (y == 1)
        t_logit = weighted_quantile(z[pos], 1 - TARGET_RECALL)
        print(f"\n2) label = {name}: ONE threshold for 90% recall on all of val")
        print(f"   logit {t_logit:.4f}  ->  probability (use this) {sigmoid(t_logit):.4f}")
        line("ALL", t_logit, usable, y)
        for t in types:
            line(t, t_logit, usable & (image_type == t), y)
        print("   honest check (threshold from one patient half, tested on the other):")
        for nm, tune, test in [("tuned on A, tested on B", in_a, ~in_a), ("tuned on B, tested on A", ~in_a, in_a)]:
            tl = weighted_quantile(z[usable & tune & (y == 1)], 1 - TARGET_RECALL)
            line(f"{nm} (p={sigmoid(tl):.4f})", tl, usable & test, y)


if __name__ == "__main__":
    main()
