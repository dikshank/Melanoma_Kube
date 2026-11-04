
FROM apache/airflow:2.8.1-python3.10

# First we set it to user as root so we can install OS level dependencies
# USER root  

# Install system dependencies needed for torch
# Usually torch does not need this cause it's prebuilt and even not while running
# Maybe Remove it and see if there is any issue
# RUN apt-get update && apt-get install -y \
#     gcc \
#     && rm -rf /var/lib/apt/lists/*

# airflow is designed to run as user airflow
USER airflow
# We are not setting a working directory because the airflow image already has a working directory set to /opt/airflow, which is where the airflow service will be run from. This is important because the airflow service will need to access the source code in order to run.

# This is now a Python-level installation, rather than an operating-system package installation.
# Install ML dependencies
# torch CPU-only to keep image size manageable
# --no-cache-dir tells pip not to keep a local cache of downloaded packages.
# Since I don't have a GPU, why bother installing the whole torch with CUDA overheads
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --trusted-host download.pytorch.org \
    --trusted-host download-r2.pytorch.org \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org

# Since we are going to run training in airflow dag, which which airflow container must have the required libraries
RUN pip install --no-cache-dir \
    mlflow==2.12.1 \
    scikit-learn \
    pillow \
    pandas \
    pyyaml \
    psycopg2-binary 

COPY . .
# we will use postgre because sqlite3.OperationalError: database gets locked
# This hit during a scheduler heartbeat write while your trigger_training task was mid-execution. SQLite allows only one writer at a time for the whole database file — there's no real concurrency control beyond a blocking lock. When your scheduler tried to write a heartbeat at the same moment something else (another task, the webserver, the scheduler's own internal polling) was writing, one of them got locked out, timed out, and errored. That cascaded: the heartbeat failure triggered a SIGTERM to your task, which killed a legitimate training job mid-flight, and Airflow then flagged it as a "zombie" task.
# This isn't a one-off fluke — it's structural. SQLite is fine for SequentialExecutor on a single dev machine running one task at a time, but the moment you have a scheduler heartbeating concurrently with running tasks (which is always true in real usage, even with just one DAG), you're exposed to this.
# Yes, move to PostgreSQL. Concretely, that gets you:
# Real row-level locking and MVCC — multiple concurrent readers/writers without blocking each other
# Required for LocalExecutor or CeleryExecutor anyway, if you ever want parallel task execution (SQLite forces you into SequentialExecutor, which runs one task at a time, full stop)
# This is Apache's own official recommendation — SQLite is documented as being for local testing only