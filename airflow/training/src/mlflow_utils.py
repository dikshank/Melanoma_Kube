'''
    MLOPS_PROJECT/
│
├── mlruns/                         ← MLflow storage
│   │
│   └── <experiment_id>/
│       │
│       └── <run_id>/
│           │
│           ├── params/             ← batch size, LR, epochs, etc.
│           ├── metrics/            ← val_recall, F1, AUC, etc.
│           ├── tags/               ← git hash, data version, etc.
│           │
│           └── artifacts/          ← actual files
│               ├── confusion_matrix.json
│               ├── classification_report.txt
│               ├── data/
│               │   └── baseline_stats.json
│               │
│               ├── model/
│               │   └── model.pth
│               │
│               └── pytorch_model/
│                   └── ...
│

'''


import logging
import subprocess
from pathlib import Path

import mlflow
import mlflow.pytorch



def get_git_commit_hash() -> str:
    """
    Retrieve the short Git commit hash corresponding to the current
    version of the codebase.

    This is used for reproducibility and experiment traceability.
    Instead of storing the complete source code with every MLflow run,
    we store the Git commit hash as metadata. The commit hash uniquely
    identifies the committed state of the Git repository at the time
    the experiment was run.

    The command executed is:

        git rev-parse --short HEAD

    where:
        - `git rev-parse` is a Git command used to inspect repository
          information.
        - `--short` returns an abbreviated version of the commit hash.
        - `HEAD` refers to the commit currently checked out.

    For example, if the current commit is:

        a83f21c9e7b4...

    the function returns:

        "a83f21c"

    This hash can then be stored as an MLflow tag, allowing us to trace
    a trained model or experiment back to the exact committed version
    of the code that produced it.

    Important:
        This function does not create a new hash from the files in the
        current directory. Git has already created the commit hash when
        the changes were committed. The function simply retrieves the
        hash of the current HEAD commit.

        If there are uncommitted changes in the working directory, this
        function will still return the hash of the last commit. Therefore,
        for reliable reproducibility, code changes should be committed
        before running the experiment.

    Returns:
        str:
            The short Git commit hash if the current directory is inside
            a valid Git repository and the command succeeds.

            Returns "unknown" if:
            - the directory is not inside a Git repository,
            - Git is unavailable,
            - the command fails, or
            - an unexpected exception occurs.
    """

    try:

        # subprocess.run() is a Python function that lets your Python program execute a command that you would normally run in the terminal
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:  # The command executed successfully
            return result.stdout.strip()  # .strip() removes whitespace from the beginning and end of a string
        return "unknown"
    except Exception:
        return "unknown"


def setup_mlflow(config: dict) -> None:
    import os

    if os.environ.get("MLFLOW_RUN_ID"):  # "Does the operating system have an environment variable called MLFLOW_RUN_ID?"
        # Running via mlflow run — only ensure experiment exists, touch nothing else
        client = mlflow.tracking.MlflowClient()  # Think of MlflowClient as a Python interface through which you can directly communicate with MLflow's tracking/registry system, for example client.get_experiment_by_name(...) or client.create_experiment(...)
        experiment_name = config["mlflow"]["experiment_name"]
        if not client.get_experiment_by_name(experiment_name): # "Does an experiment called melanoma_experiment exist?" If it doesn't exist, it creates.
            client.create_experiment(experiment_name)
        return   

    # Running via python train.py directly
    mlruns_path = Path(config["mlflow"]["tracking_uri"]) 
    mlruns_path.mkdir(parents=True, exist_ok=True) # This creates the mlruns directory if it doesn't exist. exist_ok=True, If mlruns already exists, don't throw an error.
    mlflow.set_tracking_uri(str(mlruns_path)) # "Use this location as the tracking backend." str(mlruns_path) becomes: "./mlruns" because that is what we
 
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)



def log_config_params(config: dict) -> None:
    """
    Log all training configuration parameters to MLflow.

    Args:
        config (dict): Parsed training config YAML.
    """
    # ── Flatten config into MLflow params ─────────────────────────────────
    # mlflow.log_params() is an MLflow function used to save the configuration/hyperparameters of your current experiment run.

    mlflow.log_params({
        "model_name": config["model"]["name"],
        "img_size": config["model"]["img_size"],
        "freeze_base": config["model"]["freeze_base"],
        "unfreeze_epoch": config["model"]["unfreeze_epoch"],
        "num_unfreeze": config["model"]["num_layers_to_unfreeze"],
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "lr_head": config["training"]["lr_head"],
        "lr_finetune": config["training"]["lr_finetune"],
        "weight_decay": config["training"]["weight_decay"],
        "early_stop": config["training"]["early_stopping_patience"],
        "data_version": config["data"]["version"],
        "debug_mode": config["debug"]["enabled"],
        "num_classes": 2
    })


# log_params() → experiment configuration
# set_tags()   → metadata used to identify/categorize the run



def log_tags(config: dict, param_counts: dict) -> None:
    """
    Log metadata tags for the current MLflow run.

    Tags are descriptive metadata used to identify, organize, filter,
    and trace experiments. They do not affect model training or model
    behavior; they provide context about how and where the run was
    produced.

    The tags recorded here allow us to answer questions such as:

        - Which exact Git code version produced this run?
        - Which dataset version was used?
        - What type of experiment was performed?
        - Which model architecture was used?
        - How large is the model?
        - How many parameters were actually trainable?
        - Was the run executed in debug mode?

    Args:
        config (dict):
            Parsed training configuration YAML.

        param_counts (dict):
            Dictionary containing the total and trainable parameter
            counts of the model.
    """
    # Store descriptive metadata on the current MLflow run.
    # These tags are useful for experiment traceability and filtering
    # but do not affect the training process or model itself.
    mlflow.set_tags({

        # Identifies the exact Git commit corresponding to the
        # committed code version used for this experiment.
        "git_commit_hash": get_git_commit_hash(),

        # Identifies which version of the dataset was used.
        # This is important because the same code can produce
        # different models when trained on different data versions.
        "data_version": config["data"]["version"],

        # Identifies the type of experiment, for example:
        # baseline, teacher, or distillation.
        "experiment_type": config["mlflow"]["experiment_type"],

        # Identifies the model architecture used in this run.
        "model_name": config["model"]["name"],

        # Total number of parameters in the model, including
        # parameters that may currently be frozen.
        "total_params": param_counts["total"],

        # Number of parameters that are currently trainable
        # (i.e., parameters participating in gradient updates).
        "trainable_params": param_counts["trainable"],

        # Store whether debug mode was enabled for this run.
        # Converted to string because MLflow tags are metadata strings.
        "debug_mode": str(config["debug"]["enabled"])
    })

    # Confirm through the application logs that the tags were
    # successfully sent to MLflow.

    # ── Retrieve MLflow run metadata ────────────────────────────────────────
    # Tags can be accessed later through either the MLflow UI or the
    # MlflowClient API.
    #
    # In the MLflow UI, select an experiment and open a specific run to
    # inspect its Parameters, Metrics, Tags, and Artifacts.
    #
    # Programmatically, MlflowClient.get_run(run_id) can be used to retrieve
    # the run information, including run.data.tags.
    #
    # Example:
    #
    #     client = mlflow.tracking.MlflowClient()
    #     run = client.get_run(run_id)
    #     print(run.data.tags)
    #
    # This allows us to trace a trained model back to metadata such as
    # the Git commit, dataset version, experiment type, and model details.


def log_epoch_metrics(
    epoch: int,
    train_loss: float,
    train_recall: float,
    train_f1: float,
    val_recall: float,
    val_f1: float,
    val_accuracy: float,
    val_auc: float
) -> None:
    """
    Log model performance metrics for the current training epoch to MLflow.

    Unlike MLflow parameters, which describe the configuration used for
    an experiment (e.g. learning rate, batch size, number of epochs),
    metrics represent the observed performance of the model during
    training and validation.

    This function is expected to be called once after every epoch.
    Therefore, MLflow maintains a history of the metrics across epochs,
    allowing us to inspect or visualize how the model's performance
    changes throughout training.

    Args:
        epoch (int):
            Current training epoch. This is passed to MLflow as the
            `step` so that each metric value is associated with the
            corresponding epoch.

        train_loss (float):
            Average loss computed on the training data during this epoch.

        train_recall (float):
            Recall achieved by the model on the training data during
            this epoch.

        train_f1 (float):
            F1 score achieved by the model on the training data during
            this epoch.

        val_recall (float):
            Recall achieved on the validation set during this epoch.
            This is particularly important in this project because
            validation recall is used later when comparing a new model
            against the current Production model.

        val_f1 (float):
            F1 score achieved on the validation set during this epoch.

        val_accuracy (float):
            Classification accuracy on the validation set during
            this epoch.

        val_auc (float):
            Area Under the ROC Curve (AUC) measured on the validation
            set during this epoch.

    MLflow stores these values as a time series using the epoch as the
    step. For example:

        Epoch 1 -> val_recall = 0.82
        Epoch 2 -> val_recall = 0.86
        Epoch 3 -> val_recall = 0.89

    This allows the MLflow UI to plot metrics such as validation recall,
    validation F1, and loss against the training epoch and helps us
    analyze learning behavior, convergence, and potential overfitting.
    """

    # Log all metrics produced during the current epoch to the active
    # MLflow run.
    mlflow.log_metrics(
        {
            # Training performance
            "train_loss": train_loss,
            "train_recall": train_recall,
            "train_f1": train_f1,

            # Validation performance
            "val_recall": val_recall,
            "val_f1": val_f1,
            "val_accuracy": val_accuracy,
            "val_auc": val_auc
        },

        # Associate these metric values with the current epoch.
        # This creates a metric history rather than storing only one
        # final value.
        step=epoch
    )

    # ── Retrieve metric history from MLflow ─────────────────────────────────
    # MlflowClient provides programmatic access to information stored for
    # previous MLflow runs.
    #
    # Unlike run.data.metrics, which gives the latest recorded value of a
    # metric, get_metric_history() retrieves all values recorded for that
    # metric across its different steps.
    #
    # Since log_epoch_metrics() stores metrics using:
    #
    #     step=epoch
    #
    # each returned metric value is associated with the epoch at which it
    # was recorded.
    #
    # For example, if val_recall was logged for 5 epochs:
    #
    #     Epoch 1 → 0.82
    #     Epoch 2 → 0.85
    #     Epoch 3 → 0.89
    #     Epoch 4 → 0.91
    #     Epoch 5 → 0.90
    #
    # get_metric_history() allows us to retrieve this complete history
    # instead of only the final value.
    #
    # This is useful for analyzing training curves, finding the best
    # validation performance, comparing experiments, and implementing
    # logic that depends on historical metric values.

    # from mlflow.tracking import MlflowClient

    # Create a client that can communicate with the configured MLflow
    # tracking backend.
    # client = MlflowClient()

    # Retrieve the complete history of the validation-recall metric
    # for a particular MLflow run.
    # history = client.get_metric_history(
        # run_id="YOUR_RUN_ID",
        # key="val_recall"
    # )

    # Each item in the returned history contains information about one
    # recorded value, including its training step (epoch), metric value,
    # and timestamp.
    # for metric in history:
        # print(metric.step, metric.value)
    # the step comes back automatically when you retrieve the metric history — but it is not automatically the epoch unless you logged it that way.\




def log_threshold(threshold: float) -> None:
    """
    Log the best classification threshold selected during validation-set tuning.

    The classification threshold is the probability cutoff used to convert
    the model's predicted probability into a binary class prediction.

    For example, if the threshold is 0.40:

        probability >= 0.40  -> positive class
        probability <  0.40  -> negative class

    The threshold is not a fixed configuration value in this project.
    It is determined after evaluating the model on the validation set,
    where different threshold values are tested to find the one that
    provides the desired validation performance.

    Args:
        threshold (float):
            The best threshold selected during validation-set tuning.
            For example, 0.3728.

    Returns:
        None:
            The function only records the threshold; it does not return it.
    """

    # Store the selected threshold as an MLflow parameter for this run.
    #
    # round(threshold, 4) limits the stored value to four decimal places.
    # For example:
    #
    #     0.37284619  ->  0.3728
    #
    # This makes the value easier to read while retaining sufficient
    # precision for a classification threshold.
    #
    # "best_threshold" becomes the parameter name visible in MLflow.
    mlflow.log_param(
        "best_threshold",
        round(threshold, 4)
    )

    # Also write the selected threshold to the application's log.
    #
    # %.4f formats the floating-point value to exactly four decimal places.
    # This is separate from MLflow logging:
    #
    #     mlflow.log_param(...) -> stores the value in MLflow
    #     logger.info(...)      -> writes the value to the application log





def log_evaluation_artifacts(
    artifacts_dir: Path,
    baseline_stats_path: Path = None
) -> None:
    """
    Log evaluation-related artifacts to the current MLflow run.

    Artifacts are files generated or used during an ML experiment.
    Unlike parameters and metrics, which are individual values,
    artifacts are complete files such as JSON, TXT, images, or model
    checkpoints.

    In this function, the evaluation artifacts are:
        - confusion matrix JSON
        - classification report TXT

    The baseline statistics JSON is also logged if a valid path is
    provided.

    Note:
        The model checkpoint (model.pth) is also an MLflow artifact,
        but it is handled separately by log_model() because the model
        is additionally logged using MLflow's PyTorch model interface
        and registered in the MLflow Model Registry.

    Args:
        artifacts_dir (Path):
            Directory containing the evaluation artifact files.

        baseline_stats_path (Path):
            Optional path to the baseline statistics JSON file.
            Defaults to None if no baseline statistics file is provided.

    Returns:
        None:
            This function only logs files to MLflow and does not return
            anything.
    """

    # ── Log evaluation artifacts ──────────────────────────────────────────
    #
    # artifacts_dir contains the files generated during model evaluation.
    #
    # For example:
    #
    #     artifacts/
    #     ├── confusion_matrix.json
    #     └── classification_report.txt
    #
    # iterating over artifacts_dir allows us to process every item
    # present in this directory.
    for artifact_file in artifacts_dir.iterdir():   # because evaluate.py is saving the metrices in artifacts_dir which is defined in the config

        # Check that the current item is an actual file.
        # The directory could potentially contain subdirectories as well.
        # We only want to pass files to mlflow.log_artifact().
        if artifact_file.is_file():

            # Upload the evaluation file to the current MLflow run.
            #
            # str(artifact_file) converts the pathlib.Path object into
            # a string path that MLflow can use.
            #
            # For example:
            #
            #     artifacts/confusion_matrix.json
            #
            # will be stored as an artifact associated with this run.
            mlflow.log_artifact(str(artifact_file))

            # Write a message to the normal application log indicating
            # which artifact was logged.
            #
            # artifact_file.name returns only the filename, e.g.
            # "confusion_matrix.json", rather than the complete path.

    # ── Log baseline statistics ───────────────────────────────────────────
    #
    # Baseline statistics are optional, so first check whether a path
    # was provided.
    #
    # baseline_stats_path = None
    #     → no baseline file was provided
    #
    # baseline_stats_path = Path(...)
    #     → a baseline file was provided
    #
    # We also check that the specified path actually exists before
    # attempting to upload it to MLflow.
    
    # Baseline stats are created in airflow dag, from the script baseline_stats.json
    # We log baseline_stats.json in MLflow to preserve the statistical fingerprint of the training dataset alongside the model run. This provides data traceability and gives the monitoring system a baseline against which incoming production data can be compared for drift. It doesn't record the individual training samples themselves.
    if (
        baseline_stats_path
        and Path(baseline_stats_path).exists()
    ):

        # Upload the baseline statistics file to MLflow.
        #
        # artifact_path="data" places the file inside a "data"
        # directory within the MLflow artifact storage.
        #
        # Conceptually, the MLflow run will contain:
        #
        #     data/
        #     └── baseline_stats.json
        #
        # This is only an MLflow artifact directory; it does not mean
        # that the original local file is moved into a "data" folder.
        mlflow.log_artifact(
            str(baseline_stats_path),
            artifact_path="data"
        )

        # Write a message to the application log confirming that the
        # baseline statistics file was logged.


def get_current_production_recall() -> float:
    """
    Get the val_recall of the current Production model from MLflow registry.

    Searches all runs that have a registered model version in Production stage
    and returns the best val_recall found.

    Returns:
        float: val_recall of current Production model, or 0.0 if none exists.
    """
    client = mlflow.tracking.MlflowClient()

    try:
        # Get all versions of the registered model in Production stage
        production_versions = client.get_latest_versions(
            name="melanoma_classifier",
            stages=["Production"]
        )

        if not production_versions:
            return 0.0

        # Get the run associated with the current Production model
        production_run_id = production_versions[0].run_id  # get_latest_versions(...) returns a list, even if there's only one Production version, so production_versions is a list
        client.get_run(production_run_id)

        # Extract val_recall from that run's metrics
        # MLflow stores per-step metrics — get the last value
        # Retrieve the best validation recall achieved by the current Production model.
        # Validation recall is used for model selection/promotion, while test recall is reserved for final evaluation and is not used to choose the Production model.
        recall_history = client.get_metric_history(production_run_id, "val_recall") # log_epoch_metrics saves the matrices and here we are retrieving it so we can compare the model which is in train with the model in product because MLFlow also has to update the production model

        if not recall_history:
            return 0.0

        best_recall = max(m.value for m in recall_history)   # Since we updated the validation metrices every epoch, we will have a full history, so we take the max out of it 
        return best_recall

    except Exception as e:
        return 0.0


def get_current_production_f1() -> float:
    """
    Get the val_f1 of the current Production model from MLflow registry.

    Returns:
        float: val_f1 of current Production model, or 0.0 if none exists.
    """
    client = mlflow.tracking.MlflowClient()

    try:
        production_versions = client.get_latest_versions(
            name="melanoma_classifier",
            stages=["Production"]
        )

        if not production_versions:
            return 0.0

        production_run_id = production_versions[0].run_id
        f1_history = client.get_metric_history(production_run_id, "val_f1")   # this history that we are retrieving was created by log_epoch_metrics

        if not f1_history:
            return 0.0

        best_f1 = max(m.value for m in f1_history)
        return best_f1

    except Exception as e:
        return 0.0


def promote_to_production(model_version: int) -> None:
    """
    Promote a registered model version to Production stage.
    Archive any existing Production model.

    Args:
        model_version (int): The version number to promote.
    """
    client = mlflow.tracking.MlflowClient()

    # ── Archive current Production model first ────────────────────────────
    try:
        current_production = client.get_latest_versions(
            name="melanoma_classifier",
            stages=["Production"]
        )
        for version in current_production:
            client.transition_model_version_stage(
                name="melanoma_classifier",
                version=version.version,
                stage="Archived"
            )
 
    except Exception as e:
        raise
    # ── Promote new model to Production ──────────────────────────────────
    client.transition_model_version_stage(
        name="melanoma_classifier",
        version=model_version,
        stage="Production"
    )



def log_model(
    model,
    model_name: str,
    best_checkpoint_path: Path,
    new_val_recall: float,
    new_val_f1: float = 0.0
) -> None:
    """
    Log model to MLflow registry and auto-promote to Production
    if it outperforms the current Production model (Champion/Challenger pattern).

    Two-stage comparison logic:
        Stage 1 — Recall comparison (primary metric, patient safety):
            If new_recall > current_recall + RECALL_TOLERANCE → promote
            If new_recall < current_recall - RECALL_TOLERANCE → keep current

        Stage 2 — F1 tiebreaker (when recall is tied within tolerance):
            If recalls are tied (within ±RECALL_TOLERANCE):
                If new_f1 > current_f1 → promote (better balance)
                Else → keep current (stability wins)

        This ensures:
            - Recall is always the primary safety metric
            - When models are equally safe, we pick the more precise one
            - Avoids promoting a model that flags everything as malignant

    Args:
        model: Trained PyTorch model.
        model_name (str): Architecture name.
        best_checkpoint_path (Path): Path to saved .pth checkpoint.
        new_val_recall (float): Best val_recall achieved in this training run.
        new_val_f1 (float): Best val_f1 achieved in this training run.
    """
        # ── Sanity check: reject degenerate models ────────────────────────────
    if new_val_f1 < 0.0:

        return
        
    RECALL_TOLERANCE = 0.04   # recalls within 4% are considered tied
    # ── Log raw .pth as artifact in this run ──────────────────────────────
    mlflow.log_artifact(str(best_checkpoint_path), artifact_path="model")

    # ── Register model in MLflow registry ────────────────────────────────
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="pytorch_model",
        registered_model_name="melanoma_classifier"
    )

    # Get the version number just created
    client = mlflow.tracking.MlflowClient()
    latest_versions = client.get_latest_versions(
        name="melanoma_classifier",
        stages=["None"]
    )
    new_version = latest_versions[0].version if latest_versions else 1


    # ── Champion / Challenger comparison ─────────────────────────────────
    current_production_recall = get_current_production_recall()
    current_production_f1 = get_current_production_f1()

    recall_diff = new_val_recall - current_production_recall

    if recall_diff > RECALL_TOLERANCE:
        # Stage 1: New model has strictly better recall → promote
        promote_to_production(new_version)
        mlflow.set_tag("promotion_result", "promoted_recall_improvement")

    elif recall_diff < -RECALL_TOLERANCE:

        client.transition_model_version_stage(
            name="melanoma_classifier",
            version=new_version,
            stage="Staging"
        )
        mlflow.set_tag("promotion_result", "kept_recall_worse")

    else:
        # Stage 2: Recalls are tied — use F1 as tiebreaker
        if new_val_f1 > current_production_f1:

            promote_to_production(new_version)
            mlflow.set_tag("promotion_result", "promoted_f1_tiebreaker")
        else:

            client.transition_model_version_stage(
                name="melanoma_classifier",
                version=new_version,
                stage="Staging"
            )
            mlflow.set_tag("promotion_result", "kept_f1_not_better")






# Path information:
'''
Anything we log goes to mlflow.set_tracking_uri(str(mlruns_path)), this path that is set here
and for this project we have "mlruns_path = Path(config["mlflow"]["tracking_uri"])"
So all the .log_metrices, .set_tags, .log_params will be stored in that directory
while artifacts ( .log_artifacts or .pytorch.log_model) are stored in the artifact location associated with that MLflow run. 
In this project, that ultimately uses the mlruns storage.
mlflow.log_artifact() needs the path to an existing local file that MLflow can read.
for example,     mlflow.log_artifact(str(best_checkpoint_path), artifact_path="model")
or               mlflow.log_artifact(str(artifact_file))
mlflow.pytroch.log_model() takes the actual pytorch model
for example,     mlflow.pytorch.log_model(pytorch_model=model,
                                          artifact_path="pytorch_model",
                                          registered_model_name="melanoma_classifier")


This is how it looks like 
mlruns/
└── <experiment_id>/
    └── <run_id>/
        ├── params/
        ├── metrics/
        ├── tags/
        └── artifacts/
            ├── baseline_stats.json
            ├── ...
            ├── model/
            │   └── melanoma_classifier_best.pth   ← log_artifact()
            │
            └── pytorch_model/
                ├── MLmodel
                ├── ...
                └── model files                 ← log_model()



Note that we have 'melanoma_classifier" as registered model name, which is basically just 
that MLflow creates a registered model named melanoma_classifier. Each time a new model is 
registered under that name, MLflow creates a new model version. Each version is associated 
with the model artifact produced by its corresponding run.

MLflow Run
└── artifacts/
    └── pytorch_model/
        └── actual model files
                ▲
                │
                │ referenced by
                │
MLflow Model Registry
└── melanoma_classifier
    └── Version 1

    
                                or more like 

                        RUN 1                         RUN 2
                    │                             │
                    └─ pytorch_model/             └─ pytorch_model/
                        │                             │
                        │                             │
                        ▼                             ▼
                    Model A                       Model B
                        ▲                             ▲
                        │                             │
                        │                             │
                    Version 1                     Version 2
                        \                             /
                        \                           /
                            └── melanoma_classifier ──┘
'''

# How to retrieve:
'''
1. Set up a client, and setting up a client needs to know the tracking_uri
        mlflow.set_tracking_uri(str(mlruns_path))  # 
        client = MlflowClient()

2. Find the run using run_id
        run = client.get_run(run_id)

3. Run can access these 
        run.data.params
        run.data.metrics
        run.data.tags

        i.e. 
        MLflow run
            ├── params  ← run.data.params
            ├── metrics ← run.data.metrics
            └── tags    ← run.data.tags

4. For artifacts, you use the artifact APIs
        client.list_artifacts(run_id)

        you might see:
            confusion_matrix.json
            classification_report.txt
            model/

        then,
        client.list_artifacts(run_id, path="model") might show:
            model.pth

5. Download the artifact when you need it:
        local_path = client.download_artifacts(run_id,"model/model.pth")
        and then torch.load(local_path)

        Now note that before saving the model.pth, pytorch does this torch.save(model.state_dict(), checkpoint_path)
        which basically means it has state dicts NOT the architecture, so if you want to use it
        you will have to do something like :
                    model = YourModel(...) # Recreate the same model Architecture
                    model.load_state_dict(torch.load(path))
                    model.eval()


6. Pytorch model loading is different
            model = mlflow.pytorch.load_model(model_uri), here model_uri means-> "runs:/<run_id>/<artifact_path>" and <artifact_path> means a folder name inside artifact folder
            model.eval() # this works directly


Note that both pytorch model and model.pth can be used as checkpoint, the training conditions must be identical to what was used it the original training
'''

# But what about model in production?
'''
The Production model does not live in a separate Production/ folder inside mlruns.
"Production" is a property/state of the registered model version, not a physical directory.
This is why we can do get_latest_versions(
    name="melanoma_classifier",
    stages=["Production"]
)
when we want to retrieve the model in production.

Model Registry
└── melanoma_classifier
    ├── Version 1
    ├── Version 2
    └── Version 3
             │
             └── Production

Models can be retrieved using Model registery pointers by:
1. Method 1:
        model_uri = "models:/melanoma_classifier/<version_number?"
        model = mlflow.pytorch.load_model(model_uri)

2. Method 2:
        model_uri = "models:/melanoma_classifier/Production"
        model = mlflow.pytorch.load_model(model_uri)
'''

