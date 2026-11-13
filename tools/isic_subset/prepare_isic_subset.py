"""
prepare_isic_subset.py - build the small ISIC data the cluster needs, on your laptop.

Your ISIC folder is too big to copy into the cluster, so this writes a small
folder (tens of MB) that is copied onto data-pvc instead:

    isic_v1/
      eval/<isic_id>.jpg          val images of the notebook (ids from the .npz)
      replay/<isic_id>.jpg        ~1000 TRAIN images (ids from train.csv), 50% malignant
      manifests/eval.csv          filepath,label,isic_id,image_type,patient_id
      manifests/replay.csv
      baseline_stats/isic_v1_stats.json   drift baseline, computed on the 224x224
                                          crop the model sees (eval + replay images)

Every image is stored exactly like the notebook's cache: short side 320,
bilinear, JPEG quality 95. Label = diagnosis_1: Malignant (1) / Benign (0),
the label the model was trained on. Only the notebook's three photo types are
used (TBP tiles are dropped, as in the notebook).

    eval   -> scores models (register Job, champion vs challenger), threshold
    replay -> mixed into every fine-tune so the model does not forget
Test images are never used (they are not in train.csv or in the .npz).

Usage (cmd, from Melanoma_Kube; needs only Pillow + numpy):
    python tools\\isic_subset\\prepare_isic_subset.py ^
        --folder C:\\Users\\Dikshank\\Documents\\ISIC-images ^
        --npz p2a_efficientnet_b0_imagenet_full_15ep__s0.npz ^
        --train-csv C:\\Users\\Dikshank\\Downloads\\train.csv
Then copy isic_v1 onto data-pvc (the script prints the commands).
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TYPES = ["dermoscopic", "clinical: close-up", "clinical: overview"]
PRE_RESIZE_SHORT_SIDE = 320
CACHE_JPEG_QUALITY = 95
HISTOGRAM_BINS = 10
DATA_ROOT_IN_POD = "data/isic_v1"      # manifests are read with cwd /opt/airflow


# ── the notebook's stage 1 (same code as backend/predictor.py) ───────────────
def cache_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.draft("RGB", (2 * PRE_RESIZE_SHORT_SIDE, 2 * PRE_RESIZE_SHORT_SIDE))
        image = image.convert("RGB")
        width, height = image.size
        scale = PRE_RESIZE_SHORT_SIDE / min(width, height)
        return image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR)


# ── torchvision Resize(256) + CenterCrop(224) on a PIL image, without torch ──
def model_crop(image: Image.Image) -> Image.Image:
    w, h = image.size
    short, long_ = (w, h) if w <= h else (h, w)
    new_short, new_long = 256, int(256 * long_ / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    image = image.resize((new_w, new_h), Image.BILINEAR)
    top = int(round((new_h - 224) / 2.0))
    left = int(round((new_w - 224) / 2.0))
    return image.crop((left, top, left + 224, top + 224))


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True, type=Path, help="ISIC images + metadata.csv")
    ap.add_argument("--npz", required=True, type=Path, help="the notebook's val logits file")
    ap.add_argument("--train-csv", required=True, type=Path, help="isic-datasplit train.csv")
    ap.add_argument("--out", type=Path, default=Path("isic_v1"))
    ap.add_argument("--replay", type=int, default=1000, help="number of replay (train) images")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ── what is available locally ────────────────────────────────────────
    meta = {r["isic_id"]: r for r in read_csv(args.folder / "metadata.csv")}
    files = {p.stem: p for p in args.folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}

    def usable(isic_id):
        r = meta.get(isic_id)
        return (r is not None and isic_id in files
                and (r.get("diagnosis_1") or "").strip() in ("Benign", "Malignant")
                and (r.get("image_type") or "").strip() in TYPES)

    def label(isic_id):
        return 1 if meta[isic_id]["diagnosis_1"].strip() == "Malignant" else 0

    # ── eval = the notebook's val images that are in the folder ──────────
    val_ids = [str(i) for i in np.load(args.npz, allow_pickle=True)["isic_id"]]
    eval_ids = [i for i in val_ids if usable(i)]

    # ── replay = train images in the folder, 50% malignant ───────────────
    train_rows = read_csv(args.train_csv)
    if not train_rows or "isic_id" not in train_rows[0]:
        sys.exit(f"{args.train_csv} has no isic_id column")
    train_ids = sorted({r["isic_id"] for r in train_rows} - set(val_ids))
    candidates = [i for i in train_ids if usable(i)]
    rng = np.random.default_rng(args.seed)
    mal = [i for i in candidates if label(i) == 1]
    ben = [i for i in candidates if label(i) == 0]
    n_mal = min(len(mal), args.replay // 2)
    n_ben = min(len(ben), args.replay - n_mal)
    replay_ids = list(rng.permutation(mal)[:n_mal]) + list(rng.permutation(ben)[:n_ben])

    print(f"metadata rows {len(meta)}, image files {len(files)}")
    print(f"eval   : {len(eval_ids)} of {len(val_ids)} notebook val images are in the folder "
          f"({sum(label(i) for i in eval_ids)} malignant)")
    print(f"replay : {len(replay_ids)} train images ({n_mal} malignant, {n_ben} benign) "
          f"from {len(candidates)} train images in the folder")
    if not eval_ids:
        sys.exit("no eval images - check --npz and --folder")
    overlap = set(eval_ids) & set(replay_ids)
    assert not overlap, f"{len(overlap)} images in both eval and replay"

    # ── write images, manifests, baseline ────────────────────────────────
    out = args.out
    for sub in ("eval", "replay", "manifests", "baseline_stats"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    hist = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    ch_sum = np.zeros(3); ch_sq = np.zeros(3); ch_min = np.full(3, 1.0); ch_max = np.zeros(3)
    n_pix = 0
    class_counts = {"malignant": 0, "benign": 0}

    for split, ids in (("eval", eval_ids), ("replay", replay_ids)):
        rows = []
        for k, isic_id in enumerate(ids, 1):
            dest = out / split / f"{isic_id}.jpg"
            if not dest.exists():
                cache_image(files[isic_id]).save(dest, quality=CACHE_JPEG_QUALITY)
            r = meta[isic_id]
            rows.append({"filepath": f"{DATA_ROOT_IN_POD}/{split}/{isic_id}.jpg", "label": label(isic_id),
                         "isic_id": isic_id, "image_type": r.get("image_type", ""),
                         "patient_id": r.get("patient_id") or isic_id})

            # baseline on the exact 224x224 crop the model sees (re-read the cached JPEG)
            crop = model_crop(Image.open(dest).convert("RGB"))
            arr = np.asarray(crop, dtype=np.float32) / 255.0
            hist += np.histogram(arr, bins=HISTOGRAM_BINS, range=(0.0, 1.0))[0]
            flat = arr.reshape(-1, 3)
            ch_sum += flat.sum(0); ch_sq += (flat ** 2).sum(0)
            ch_min = np.minimum(ch_min, flat.min(0)); ch_max = np.maximum(ch_max, flat.max(0))
            n_pix += flat.shape[0]
            class_counts["malignant" if label(isic_id) else "benign"] += 1
            if k % 200 == 0 or k == len(ids):
                print(f"  {split}: {k}/{len(ids)}")

        with open(out / "manifests" / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["filepath", "label", "isic_id", "image_type", "patient_id"])
            w.writeheader(); w.writerows(rows)

    mean = ch_sum / n_pix
    var = ch_sq / n_pix - mean ** 2
    total = sum(class_counts.values())
    stats = {
        "version": "isic_v1",
        "computed_on": "224x224 model-input crop (notebook cache -> Resize(256) -> CenterCrop(224)), "
                       "pixel values in [0, 1] before normalization, eval + replay images",
        "total_train_images": total,
        "class_distribution": {c: {"count": n, "percentage": round(100 * n / total, 2)}
                               for c, n in class_counts.items()},
        "channel_stats": {ch: {"mean": float(mean[i]), "variance": float(var[i]), "std": float(np.sqrt(var[i])),
                               "min": float(ch_min[i]), "max": float(ch_max[i])}
                          for i, ch in enumerate("RGB")},
        "pixel_histogram": {"bin_edges": np.linspace(0, 1, HISTOGRAM_BINS + 1).tolist(),
                            "counts": hist.tolist(), "total_pixels": int(hist.sum())},
        "image_size_distribution": {"224x224": total},
    }
    with open(out / "baseline_stats" / "isic_v1_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"\nwrote {out}  ({size_mb:.1f} MB)")
    print("\ncopy it onto data-pvc (cmd, from this folder):")
    print("  kubectl apply -f k8s/storage/pvc-loader.yaml")
    print("  kubectl wait --for=condition=Ready pod/pvc-loader -n melanoma-production --timeout=120s")
    print(f"  kubectl cp {out} melanoma-production/pvc-loader:/data/isic_v1")
    print("  kubectl exec pvc-loader -n melanoma-production -- chmod -R 777 /data/isic_v1")
    print("  kubectl exec pvc-loader -n melanoma-production -- ls /data/isic_v1")
    print("  kubectl delete pod pvc-loader -n melanoma-production")


if __name__ == "__main__":
    main()
