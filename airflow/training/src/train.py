"""
train.py
--------
Phase 3 & 4 | Executed: Local (on-prem, CPU)

Main training script for melanoma classification.

Usage:
    python train.py --config configs/config_v1_mobilenet.yaml
    python train.py --config configs/config_v1_efficientnet.yaml

Training strategy:
    Phase 1 (epochs 1 to unfreeze_epoch):
        - Base CNN frozen
        - Only classifier head trained
        - Higher learning rate (lr_head)

    Phase 2 (epochs unfreeze_epoch+1 to end):
        - Last N layers of base unfrozen
        - Full fine-tuning at lower learning rate (lr_finetune)

Debug mode:
    Set debug.enabled: true in config
    Loads only 20 images, runs 2 epochs
    Verifies entire pipeline end-to-end in ~2 minutes
"""


'''
I will explain one thing here that in the previous version which is available on "https://github.com/dikshank/MLOPS_PROJECT/blob/master/training/src/train.py"
that in that version model checkpointing directory was a bit messy as each model's config.yaml contained this:
output:
  model_dir: "training/models/v1"
  artifacts_dir: "training/artifacts/v1_mobilenet"

For artifacts there is a clear seperation but for "model_dir" there is not, so if we train multiple models the model_dir(which is for checkpoints) would have looked like this:

training/
└── models/
    └── v1/
        ├── efficientnet_b0_best.pth
        └── mobilenet_v3_small_best.pth

        but now I have changed the yaml so first change is "model_dir" was a bit misleading name so new name is "checkpoint_dir" and it will look like this:

training/
└── checkpoints/
    └── v1/
        ├── efficientnet_b0/
        │   └── best.pth
        └── mobilenet_v3_small/
            └── best.pth

'''


# ── Local imports ─────────────────────────────────────────────────────────────
import mlflow
from mlflow_utils import (
    setup_mlflow,
    log_config_params,
    log_tags,
    log_epoch_metrics,
    log_threshold,
    log_evaluation_artifacts,
    log_model
)
from evaluate import evaluate
from model import get_model, unfreeze_last_layers, count_parameters
from dataset import MelanomaDataset
import argparse
import logging
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, str(Path(__file__).parent)) # This code explicitly adds the directory containing the current train.py file to Python’s module search path (sys.path). Path(__file__).parent gets the directory where train.py is located, and sys.path.insert(0, ...) places that directory at the beginning of the search path, allowing Python to reliably find local modules such as model.py, dataset.py, evaluate.py, and mlflow_utils.py when they are imported. In your setup, Python normally adds the script’s directory automatically when train.py is run directly, so this line is mainly a defensive measure to make local imports more reliable.



# This function is supposed to read the .yaml of the model that we training, so as a path we will give the model confing yaml path
def load_config(config_path: str) -> dict:
    """
    Load and parse a YAML training config file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)  # since yaml looks alot like a dictionary, it is easy to convert the yaml file to a nest_dictionary

    return config



'''
Important concept about dataloder or anything that can be iterated

This is how the dataloader looks like (not exactly as it is in pytorch but get the idea, we care about pausing the for loop before we call next(), so we can do our thing with processing the batch))
class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def __iter__(self):
        # Decide which dataset indices to use
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        # Create batches of indices
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            # Ask Dataset for each sample
            samples = [
                self.dataset[idx]
                for idx in batch_indices
            ]

            # Combine individual samples into tensors
            batch = default_collate(samples)

            yield batch


def numbers():
    for i in range(3):
        yield i

if we call numbers the we will have an generator, 
it = numbers(), it is a generator 
next(it) will make the for loop execute and we will get 1
next(it) will get us 2
next(it) will get us 3


now
for x in something:

this syntax first calls iterator = something.__iter__() or iter(something), then for loop repeadly calls next(iterator)
so that "something" must have an __iter__() implemented, which usually returns a generator and on that generator we call next()   

note that generator object IS an iterator too, and iterators have __next__() and __iter__() defined in them, no need to overthing this now.

'''

def build_dataloaders(config: dict) -> tuple:
    """
    Build train, val, and test DataLoaders from config.

    Args:
        config (dict): Parsed training config.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # since I have a explicit debug mode, I have to keep that into acccount here as well
    debug = config["debug"]["enabled"]
    debug_size = config["debug"]["num_images"]
    
    img_size = config["model"]["img_size"]
    batch_size = config["debug"]["batch_size"] if debug else config["training"]["batch_size"]
    manifest_dir = Path(config["data"]["manifest_dir"])   # This is where the manifest_dir (the csv files) are located, you could argue that since the data is consitant for all model we might have a different config file for this, but I think this gives more flexibility so I keep it here


    train_ds = MelanomaDataset(
        manifest_path = manifest_dir / "train_manifest.csv",
        img_size=img_size,
        split="train",
        debug=debug,
        debug_size=debug_size)
    
    val_ds = MelanomaDataset(
        manifest_path=manifest_dir / "val_manifest.csv",
        img_size=img_size,
        split="val",
        debug=debug,
        debug_size=debug_size)
    
    test_ds = MelanomaDataset(
        manifest_path=manifest_dir / "test_manifest.csv",
        img_size=img_size,
        split="test",
        debug=debug,
        debug_size=debug_size)
    
    
    # num_workers=0 for Windows compatibility (avoids multiprocessing issues), or we can have it as a argument but that is for later
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=False)
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False)
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str) -> tuple:
    """
    Run one full training epoch.

    Args:
        model (nn.Module): Model to train.
        dataloader (DataLoader): Training dataloader.
        optimizer: PyTorch optimizer.
        criterion: Loss function.
        device (str): Device to train on.

    Returns:
        tuple: (avg_loss, recall, f1)
    """
    from sklearn.metrics import recall_score, f1_score

    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # ── Forward pass ──────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)  # every batch should have fresh gradient, so we refresh it here
        logits = model(images)
        loss = criterion(logits, labels)

        # ── Backward pass ─────────────────────────────────────────────────
        loss.backward()   # That is obvious
        optimizer.step()  # update the parameters here

        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = running_loss / len(dataloader)
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)

    return avg_loss, float(recall), float(f1)

'''
images + labels
      ↓
    model
      ↓
    logits
      ↓
criterion(logits, labels)
      ↓
     loss
      ↓
loss.backward()
      ↓
calculate gradients
      ↓
optimizer.step()
      ↓
UPDATE weights
'''

def run_training(config: dict) -> None:
    """
    Full training pipeline: build model, train, evaluate, log to MLflow.

    Args:
        config (dict): Parsed training config.
    """
    # ── Device ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Debug mode override ───────────────────────────────────────────────
    if config["debug"]["enabled"]:
        config["training"]["epochs"] = config["debug"]["epochs"]

    # ── Directories ───────────────────────────────────────────────────────
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    artifacts_dir = Path(config["output"]["artifacts_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{config['model']['name']}_best.pth"

    setup_mlflow(config)
    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):

        # with thing():
            ## code
        # thing = thing()
        # enter the context
        # thing.__enter__()
        # try:
            ## code inside with
            # ...
        # finally:
            ## leave the context
            # thing.__exit__(...)

        model = get_model(
            model_name=config["model"]["name"],  # name of the model
            num_classes=2,
            freeze_base=config["model"]["freeze_base"]  # do we want to freeze base
        )
        model = model.to(device)
    
        param_counts = count_parameters(model)
        log_config_params(config)
        log_tags(config, param_counts)
    
        # ── Dataloaders ───────────────────────────────────────────────────
        train_loader, val_loader, test_loader = build_dataloaders(config)

        # ── Loss: weighted to penalize missing malignant (recall focus) ───
        # Weight malignant class higher since false negatives are more costly
        class_weights = torch.tensor(
            config["training"]["class_weights"], dtype=torch.float
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimiser is the gradient update rule
        # ── Optimizer: head-only phase ────────────────────────────────────
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["lr_head"],
            weight_decay=config["training"]["weight_decay"]
        )

        # ── Training loop ─────────────────────────────────────────────────
        best_val_recall = 0.0
        best_val_f1 = 0.0
        best_threshold = 0.5
        patience_counter = 0
        patience = config["training"]["early_stopping_patience"]
        unfreeze_epoch = config["model"]["unfreeze_epoch"]
        total_epochs = config["training"]["epochs"]
        phase_switched = False



        for epoch in range(1, total_epochs + 1):

           # ── For first few epoches I will train the classifier head and then I will unfreeze the frozen later i.e. Switch to fine-tuning phase ───────────────────────────────
            if epoch == unfreeze_epoch and not phase_switched:
                model = unfreeze_last_layers(
                    model,
                    config["model"]["name"],
                    config["model"]["num_layers_to_unfreeze"]
                )
                # Lower learning rate for fine-tuning
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=config["training"]["lr_finetune"],
                    weight_decay=config["training"]["weight_decay"]
                )
                phase_switched = True

            # ── Train one epoch ───────────────────────────────────────────
            train_loss, train_recall, train_f1 = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )

            
            # ── Validate ──────────────────────────────────────────────────
            val_metrics, val_threshold = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                split="val",
                artifacts_dir=artifacts_dir,
                threshold=None  # auto-tune on val
            )

            # ── Log to MLflow ─────────────────────────────────────────────
            log_epoch_metrics(
                epoch=epoch,
                train_loss=train_loss,
                train_recall=train_recall,
                train_f1=train_f1,
                val_recall=val_metrics["recall"],
                val_f1=val_metrics["f1"],
                val_accuracy=val_metrics["accuracy"],
                val_auc=val_metrics["auc"]
            )

            # ── Save best model ───────────────────────────────────────────
            if val_metrics["recall"] > best_val_recall:
                # Stage 1: recall improved — always save
                best_val_recall = val_metrics["recall"]
                best_val_f1 = val_metrics["f1"]
                best_threshold = val_threshold
                torch.save(model.state_dict(), checkpoint_path)
                patience_counter = 0

            elif val_metrics["recall"] == best_val_recall and val_metrics["f1"] > best_val_f1:
                # Stage 2: recall tied but F1 improved — save better balanced model
                best_val_f1 = val_metrics["f1"]
                best_threshold = val_threshold
                torch.save(model.state_dict(), checkpoint_path)
                patience_counter = 0

            else:
                patience_counter += 1

            # ── Early stopping ────────────────────────────────────────────
            if patience_counter >= patience:
                break

        # ── Final evaluation on test set ──────────────────────────────────
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        test_metrics, _ = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            split="test",
            artifacts_dir=artifacts_dir,
            threshold=best_threshold   # use val-tuned threshold on test
        )

        # ── Log final test metrics ────────────────────────────────────────
        mlflow.log_metrics({
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_auc": test_metrics["auc"]
        })

        # ── Log threshold ─────────────────────────────────────────────────
        log_threshold(best_threshold)

        # ── Log artifacts ─────────────────────────────────────────────────
        log_evaluation_artifacts(
            artifacts_dir=artifacts_dir,
            baseline_stats_path=config["data"].get("baseline_stats_path")
        )

        # ── Log model + auto-promote if best recall so far ───────────────
        log_model(
            model=model,
            model_name=config["model"]["name"],
            best_checkpoint_path=checkpoint_path,
            new_val_recall=best_val_recall,
            new_val_f1=best_val_f1
        )















def main():
    parser = argparse.ArgumentParser(
        description="Train melanoma classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML"
    )
    args = parser.parse_args()   # args = parser.parse_args() parses all the command-line arguments provided by the user and stores them in the args object. Each argument can then be accessed using the name defined for that argument. For example, because we defined the argument as --config, its value is available as args.config.#
    config = load_config(args.config)
    run_training(config)

# So it does not run if we just do import train
if __name__ == "__main__":
    main()
