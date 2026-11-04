"""
data_pipeline_dag.py
--------------------
Phase 1 & 2 | Executed: Local (inside Airflow Docker container)

Airflow DAG: melanoma_data_pipeline

Orchestrates the full data engineering pipeline in 4 sequential tasks:
    1. validate_images       → check all raw images are valid
    2. split_dataset         → stratified train/val/test split
    3. preprocess_images     → resize and save to processed folder
    4. compute_baseline_stats → compute stats for drift detection

The config file path is passed as a DAG-level param so the same DAG
can be triggered for both v1 and v2 data by simply changing the config.

Usage:
    - Trigger via Airflow UI at localhost:8080
    - Pass config path as: {"config_path": "/path/to/pipeline_config_v1.yaml"}
    - Or set the default config path below for convenience
"""

import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Import pipeline data_pipleline ───────────────────────────────────────────────────
# data_pipeline are mounted into the container at /opt/airflow/data_pipeline
import sys
sys.path.insert(0, "/opt/airflow/data_pipeline")

import validate
import split
import preprocess
import baseline_stats

# ── Logging ───────────────────────────────────────────────────────────────────

# ── Default config path (override via DAG params in UI) ──────────────────────
DEFAULT_CONFIG_PATH = "/opt/airflow/configs/pipeline_config_v1.yaml"


def load_config(config_path: str) -> dict:
    """
    Load and parse a YAML config file.

    Args:
        config_path (str): Absolute path to the YAML config file.

    Returns:
        dict: Parsed config as a dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


# ── Task wrapper functions ─────────────────────────────────────────────────────
# Each wrapper loads the config and calls the corresponding script's run()
# Airflow PythonOperator requires callables with **context kwargs

def task_validate(**context):
    """Airflow task wrapper for validate.py"""
    config_path = context["params"].get("config_path", DEFAULT_CONFIG_PATH)
    config = load_config(config_path)
    validate.run(config)


def task_split(**context):
    """Airflow task wrapper for split.py"""
    config_path = context["params"].get("config_path", DEFAULT_CONFIG_PATH)
    config = load_config(config_path)
    split.run(config)


def task_preprocess(**context):
    """Airflow task wrapper for preprocess.py"""
    config_path = context["params"].get("config_path", DEFAULT_CONFIG_PATH)
    config = load_config(config_path)
    preprocess.run(config)


def task_baseline_stats(**context):
    """Airflow task wrapper for baseline_stats.py"""
    config_path = context["params"].get("config_path", DEFAULT_CONFIG_PATH)
    config = load_config(config_path)
    baseline_stats.run(config)


# ── DAG default arguments ──────────────────────────────────────────────────────
default_args = {
    "owner": "melanoma_mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2)
}

# ── DAG definition ─────────────────────────────────────────────────────────────
with DAG(
    dag_id="melanoma_data_pipeline",
    description="Data engineering pipeline: validate → split → preprocess → baseline stats",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,        # manual trigger only
    catchup=False,
    tags=["melanoma", "data-engineering", "mlops"],
    params={
        # Override this in the Airflow UI trigger form to switch between v1 and v2
        "config_path": DEFAULT_CONFIG_PATH
    }
) as dag:

    # ── Task 1: Validate ───────────────────────────────────────────────────
    t1_validate = PythonOperator(
        task_id="validate_images",
        python_callable=task_validate,
    )

    # ── Task 2: Split ──────────────────────────────────────────────────────
    t2_split = PythonOperator(
        task_id="split_dataset",
        python_callable=task_split,
    )

    # ── Task 3: Preprocess ─────────────────────────────────────────────────
    t3_preprocess = PythonOperator(
        task_id="preprocess_images",
        python_callable=task_preprocess,
    )

    # ── Task 4: Baseline stats ─────────────────────────────────────────────
    t4_baseline_stats = PythonOperator(
        task_id="compute_baseline_stats",
        python_callable=task_baseline_stats,
    )

    # ── Task dependencies (sequential pipeline) ────────────────────────────
    t1_validate >> t2_split >> t3_preprocess >> t4_baseline_stats