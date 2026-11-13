"""
eval_api.py - send every image in a folder to the running backend's /predict
and score the answers against the ISIC metadata.csv in the same folder.

Ground truth (two targets, because the model was trained for MELANOMA, not
for "malignant"):
    malignant : diagnosis_1 == "Malignant"  (1)  /  "Benign" (0)
                -> includes basal cell and squamous cell carcinoma, which the
                   model was never trained to flag
    melanoma  : diagnosis_2 or diagnosis_3 mentions "melanoma"  (1), else 0
Rows whose diagnosis_1 is neither Benign nor Malignant (Indeterminate, empty)
are skipped.

Results are appended to a CSV as they arrive, so the script can be stopped
and restarted: images already in the CSV are not sent again.

Usage (Windows cmd, from the Melanoma_Kube folder):
    pip install requests
    kubectl port-forward -n melanoma-production svc/backend 8000:8000      (separate window)
    python tools\\eval_api\\eval_api.py --folder C:\\Users\\Dikshank\\Documents\\ISIC-images

    # only score images from your test split (recommended, avoids training images):
    python tools\\eval_api\\eval_api.py --folder ... --only-ids C:\\path\\to\\test.csv
    # quick try on 200 images first:
    python tools\\eval_api\\eval_api.py --folder ... --limit 200

Why port 8000 (backend) and not the website's port 80 (nginx):
    nginx has no client_max_body_size, so its default 1 MB limit rejects
    larger photos with HTTP 413. Calling the backend directly avoids that.

Side effects on the cluster (every /predict call does these):
    - a copy of the image is saved to logs-pvc/pending_feedback/
    - Prometheus counters and the drift window include these calls
The script prints a cleanup command for the saved copies at the end.
"""

import argparse
import csv
import mimetypes
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing package: run  pip install requests")

KNOWN_TYPES = {"dermoscopic", "clinical: close-up", "clinical: overview"}
RESULT_FIELDS = ["isic_id", "image_type", "diagnosis_1", "y_malignant", "y_melanoma",
                 "label", "malignant_prob", "threshold_used", "image_id", "error"]


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────

def read_metadata(folder: Path) -> list:
    """Rows of metadata.csv with a usable Benign/Malignant label and an image file."""
    meta_path = folder / "metadata.csv"
    if not meta_path.exists():
        sys.exit(f"metadata.csv not found in {folder}")

    files = {}
    for p in folder.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            files[p.stem] = p

    rows, skipped_label, missing_file = [], 0, 0
    with open(meta_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d1 = (r.get("diagnosis_1") or "").strip()
            if d1 not in ("Benign", "Malignant"):
                skipped_label += 1
                continue
            path = files.get(r["isic_id"])
            if path is None:
                missing_file += 1
                continue
            text = f'{r.get("diagnosis_2", "")} {r.get("diagnosis_3", "")}'.lower()
            rows.append({
                "isic_id": r["isic_id"],
                "image_type": (r.get("image_type") or "unknown").strip() or "unknown",
                "diagnosis_1": d1,
                "y_malignant": 1 if d1 == "Malignant" else 0,
                "y_melanoma": 1 if "melanoma" in text else 0,
                "path": path,
            })
    print(f"metadata: {len(rows)} usable images "
          f"({skipped_label} skipped: not Benign/Malignant, {missing_file} skipped: no image file)")
    return rows


def read_ids(path: Path) -> set:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "isic_id" not in reader.fieldnames:
            sys.exit(f"{path} has no isic_id column")
        return {r["isic_id"] for r in reader}


def already_done(results_path: Path) -> set:
    if not results_path.exists():
        return set()
    with open(results_path, newline="", encoding="utf-8") as f:
        return {r["isic_id"] for r in csv.DictReader(f) if not r.get("error")}


# ─────────────────────────────────────────────────────────────────────────────
# Calling the API
# ─────────────────────────────────────────────────────────────────────────────

def predict_one(session: requests.Session, url: str, row: dict, retries: int = 3) -> dict:
    out = {k: row.get(k, "") for k in RESULT_FIELDS}
    mime = mimetypes.guess_type(row["path"].name)[0] or "image/jpeg"
    for attempt in range(retries):
        try:
            with open(row["path"], "rb") as f:
                # send the photo type like the UI dropdown does, so the backend uses
                # that type's threshold (types it doesn't know are sent as nothing)
                data = {"image_type": row["image_type"]} if row["image_type"] in KNOWN_TYPES else {}
                resp = session.post(url, files={"file": (row["path"].name, f, mime)}, data=data, timeout=120)
            if resp.status_code == 200:
                j = resp.json()
                out.update(label=j["label"], malignant_prob=j["malignant_prob"],
                           threshold_used=j["threshold_used"], image_id=j["image_id"], error="")
                return out
            out["error"] = f"HTTP {resp.status_code}: {resp.text[:120]}"
            if resp.status_code < 500:
                return out          # 4xx: retrying won't help
        except requests.RequestException as e:
            out["error"] = f"{type(e).__name__}: {e}"[:200]
        time.sleep(2 * (attempt + 1))
    return out


def run(rows: list, url: str, results_path: Path, workers: int) -> None:
    new_file = not results_path.exists()
    session = requests.Session()
    done = errors = 0
    start = time.time()
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(predict_one, session, url, r) for r in rows]
            for fut in as_completed(futures):
                res = fut.result()
                writer.writerow(res)
                f.flush()
                done += 1
                errors += bool(res["error"])
                if res["error"] and errors <= 5:
                    print(f"  error on {res['isic_id']}: {res['error']}")
                if done % 50 == 0 or done == len(rows):
                    rate = done / (time.time() - start)
                    left = (len(rows) - done) / rate if rate else 0
                    print(f"  {done}/{len(rows)} done, {errors} errors, "
                          f"{rate:.1f} img/s, ~{left / 60:.0f} min left")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def auc(y: list, p: list) -> float:
    """ROC AUC via the rank-sum formula (ties get average ranks)."""
    pos = sum(y)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks = [0.0] * len(p)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, t in zip(ranks, y) if t == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def scores(y: list, p: list, thr) -> dict:
    """thr: one threshold, or a list with each image's own threshold (per photo type)."""
    th = thr if isinstance(thr, list) else [thr] * len(y)
    tp = sum(1 for t, q, c in zip(y, p, th) if t == 1 and q >= c)
    fn = sum(1 for t, q, c in zip(y, p, th) if t == 1 and q < c)
    fp = sum(1 for t, q, c in zip(y, p, th) if t == 0 and q >= c)
    tn = sum(1 for t, q, c in zip(y, p, th) if t == 0 and q < c)
    div = lambda a, b: a / b if b else float("nan")
    return {"n": len(y), "pos": tp + fn, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "recall": div(tp, tp + fn), "precision": div(tp, tp + fp),
            "specificity": div(tn, tn + fp), "accuracy": div(tp + tn, len(y)),
            "AUC": auc(y, p)}


def threshold_for_recall(y: list, p: list, target: float) -> float:
    """Largest threshold that still flags at least `target` of the positives."""
    pos = sorted((q for t, q in zip(y, p) if t == 1), reverse=True)
    if not pos:
        return float("nan")
    k = max(1, -(-int(round(target * len(pos) * 1000)) // 1000))   # ceil(target * n)
    k = min(k, len(pos))
    return pos[k - 1]


def report(results_path: Path) -> None:
    with open(results_path, newline="", encoding="utf-8") as f:
        res = [r for r in csv.DictReader(f) if not r["error"]]
    if not res:
        print("no successful predictions to score")
        return
    p = [float(r["malignant_prob"]) for r in res]
    thr = [float(r["threshold_used"]) for r in res]            # each image's own (per photo type)
    thr_text = ", ".join(sorted({f"{t:.4f}" for t in thr}))
    types = sorted({r["image_type"] for r in res})

    cols = ["n", "pos", "TP", "FP", "FN", "TN", "recall", "precision", "specificity", "accuracy", "AUC"]
    def line(name, s):
        vals = [f"{s[c]:.3f}" if isinstance(s[c], float) else str(s[c]) for c in cols]
        return f"  {name:28s}" + "".join(f"{v:>12s}" for v in vals)
    header = f"  {'':28s}" + "".join(f"{c:>12s}" for c in cols)

    for target in ("y_melanoma", "y_malignant"):
        y = [int(r[target]) for r in res]
        title = "MELANOMA vs not (what the model was trained for)" if target == "y_melanoma" \
            else "MALIGNANT vs benign (diagnosis_1; includes BCC/SCC)"
        print(f"\n=== {title} | app threshold(s) {thr_text} ===")
        print(header)
        print(line("ALL", scores(y, p, thr)))
        for t in types:
            idx = [i for i, r in enumerate(res) if r["image_type"] == t]
            print(line(t, scores([y[i] for i in idx], [p[i] for i in idx], [thr[i] for i in idx])))

        t90 = threshold_for_recall(y, p, 0.90)
        print(f"\n  threshold that would give 90% recall here: {t90:.4f}")
        print(line("ALL @ that threshold", scores(y, p, t90)))

    print(f"\n{len(res)} images scored; raw answers in {results_path}")
    print("note: a threshold picked on these same images is optimistic for these images.")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True, type=Path, help="folder with images + metadata.csv")
    ap.add_argument("--url", default="http://localhost:8000/predict")
    ap.add_argument("--out", type=Path, default=Path("eval_results.csv"))
    ap.add_argument("--only-ids", type=Path, help="CSV with an isic_id column (e.g. your test.csv)")
    ap.add_argument("--limit", type=int, help="send at most this many new images")
    ap.add_argument("--workers", type=int, default=2, help="parallel requests")
    ap.add_argument("--report-only", action="store_true", help="just score the existing results CSV")
    args = ap.parse_args()

    if not args.report_only:
        rows = read_metadata(args.folder)
        if args.only_ids:
            keep = read_ids(args.only_ids)
            rows = [r for r in rows if r["isic_id"] in keep]
            print(f"--only-ids: {len(rows)} images are in {args.only_ids.name}")
        done = already_done(args.out)
        rows = [r for r in rows if r["isic_id"] not in done]
        if done:
            print(f"{len(done)} already in {args.out}, skipping them")
        if args.limit:
            rows = rows[: args.limit]

        try:
            requests.get(args.url.replace("/predict", "/ready"), timeout=10).raise_for_status()
        except requests.RequestException as e:
            sys.exit(f"backend not reachable at {args.url} ({e}).\n"
                     "Start: kubectl port-forward -n melanoma-production svc/backend 8000:8000")

        started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(f"sending {len(rows)} images to {args.url} with {args.workers} workers")
        if rows:
            run(rows, args.url, args.out, args.workers)
            print("\nTo delete the copies this run saved in logs-pvc/pending_feedback:")
            print('  kubectl exec -n melanoma-production deploy/backend-deployment -- '
                  f'find /app/logs/pending_feedback -type f -newermt "{started_utc} UTC" -delete')

    report(args.out)


if __name__ == "__main__":
    main()
