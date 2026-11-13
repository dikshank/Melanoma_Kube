"""
retraining_dag.py
-----------------
Airflow DAG: melanoma_retraining_pipeline

Retraining = FINE-TUNING the current Production model (training/src/finetune.py)
on the feedback images users confirmed, mixed with a small replay set of ISIC
training images. Never trains from scratch, never uses the whole dataset.

Trigger (checked every 30 min by check_trigger):
    retrain_needed.flag in logs-pvc (written by the backend: misclassification
    rate or drift), or a manual run from the Airflow UI.

Tasks:
    check_trigger    flag present (or manual)?
          |
    claim_feedback   flag -> retrain_in_progress.flag; feedback_data/* moved into
          |          feedback_batches/<run>/  (feedback that arrives while training
          |          stays in feedback_data/ for the next run; a new flag written
          |          meanwhile is not deleted)
    finetune         finetune.py on that batch: champion vs challenger on the
          |          ISIC eval set, promote only if better (else Staging)
    report           log the result
          |
    cleanup          batch -> feedback_data_archive/<run>/, remove in-progress flag
"""

import json
import logging
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("retraining_dag")

# ── Paths inside the scheduler pod (logs-pvc at /opt/airflow/app_logs) ───────
APP_LOGS            = Path("/opt/airflow/app_logs")
RETRAIN_FLAG_PATH   = APP_LOGS / "retrain_needed.flag"
IN_PROGRESS_FLAG    = APP_LOGS / "retrain_in_progress.flag"
FEEDBACK_DATA_DIR   = APP_LOGS / "feedback_data"
BATCHES_DIR         = APP_LOGS / "feedback_batches"
ARCHIVE_DIR         = APP_LOGS / "feedback_data_archive"
FINETUNE_SCRIPT     = Path("/opt/airflow/training/src/finetune.py")


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Check trigger
# ─────────────────────────────────────────────────────────────────────────────

def check_trigger(**context) -> str:
    """
    Check if retraining should proceed.

    Reads the retrain_needed.flag file written by FastAPI.
    If flag exists → proceed with retraining.
    If manually triggered → always proceed.

    Returns:
        str: Trigger reason ('misclassification', 'drift', or 'manual')

    Raises:
        ValueError: If no trigger condition is met (stops pipeline cleanly).
    """
    # Manual trigger always proceeds
    is_manual = context.get("dag_run") and \
                context["dag_run"].run_type == "manual"

    if RETRAIN_FLAG_PATH.exists():
        with open(RETRAIN_FLAG_PATH) as f:
            flag = json.load(f)
        reason = flag.get("reason", "unknown")
        logger.info(
            "✅ Retraining triggered | reason=%s | flag_time=%s",
            reason, flag.get("timestamp", "unknown")
        )
        return reason

    elif is_manual:
        logger.info("✅ Manual retraining triggered from Airflow UI")
        return "manual"

    else:
        logger.info(
            "ℹ️ No retraining trigger found. "
            "Flag file not present. Skipping pipeline."
        )
        raise ValueError(
            "No retraining trigger condition met. "
            "Pipeline stopped cleanly."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Claim the feedback (so new feedback / new flags during training are kept)
# ─────────────────────────────────────────────────────────────────────────────

def claim_feedback(**context) -> str:
    run_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    batch = BATCHES_DIR / run_name
    if RETRAIN_FLAG_PATH.exists():
        RETRAIN_FLAG_PATH.replace(IN_PROGRESS_FLAG)          # atomic rename
    moved = 0
    for class_name in ("malignant", "benign"):
        src = FEEDBACK_DATA_DIR / class_name
        dst = batch / class_name
        dst.mkdir(parents=True, exist_ok=True)
        if src.exists():
            for f in src.iterdir():
                if f.is_file():
                    f.replace(dst / f.name)                    # same volume -> atomic rename
                    moved += 1
    logger.info("claimed %d feedback images into %s", moved, batch)
    return str(batch)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Fine-tune
# ─────────────────────────────────────────────────────────────────────────────

def finetune(**context) -> None:
    batch = Path(context["ti"].xcom_pull(task_ids="claim_feedback"))
    result_file = batch / "finetune_result.json"
    cmd = ["python", str(FINETUNE_SCRIPT), "--feedback-dir", str(batch), "--result-file", str(result_file)]
    logger.info("running: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd="/opt/airflow", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:                                   # stream into the Airflow task log
        logger.info(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError(f"finetune.py failed (exit {proc.returncode}); the batch is kept in {batch}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: Report
# ─────────────────────────────────────────────────────────────────────────────

def report(**context) -> None:
    batch = Path(context["ti"].xcom_pull(task_ids="claim_feedback"))
    result_file = batch / "finetune_result.json"
    result = json.loads(result_file.read_text()) if result_file.exists() else {}
    logger.info("fine-tune result: %s", result)
    if result.get("promoted"):
        logger.info("v%s is now Production; the backend reloads it within ~30 s", result.get("version"))
    elif result.get("version"):
        logger.info("v%s is in Staging (not better than the champion)", result.get("version"))


# ─────────────────────────────────────────────────────────────────────────────
# Task 5: Cleanup
# ─────────────────────────────────────────────────────────────────────────────

def cleanup(**context) -> None:
    batch = Path(context["ti"].xcom_pull(task_ids="claim_feedback"))
    if batch.exists():
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(batch), str(ARCHIVE_DIR / batch.name))
        logger.info("archived %s -> %s", batch, ARCHIVE_DIR / batch.name)
    if IN_PROGRESS_FLAG.exists():
        IN_PROGRESS_FLAG.unlink()
    logger.info("cleanup complete")


# ── DAG definition ─────────────────────────────────────────────────────────────
default_args = {
    "owner":            "melanoma_mlops",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          0,
}

with DAG(
    dag_id="melanoma_retraining_pipeline",
    description="Fine-tunes the Production model on feedback + replay; promotes only if better on the ISIC eval set.",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=timedelta(minutes=30),
    catchup=False,
    max_active_runs=1,
    tags=["melanoma", "retraining", "mlops"],
) as dag:
    t1 = PythonOperator(task_id="check_trigger", python_callable=check_trigger)
    t2 = PythonOperator(task_id="claim_feedback", python_callable=claim_feedback)
    t3 = PythonOperator(task_id="finetune", python_callable=finetune, execution_timeout=timedelta(hours=3))
    t4 = PythonOperator(task_id="report", python_callable=report)
    t5 = PythonOperator(task_id="cleanup", python_callable=cleanup)
    t1 >> t2 >> t3 >> t4 >> t5
