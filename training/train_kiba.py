#!/usr/bin/env python3

"""
=============================================================
KIBA Training Script

Causal Multi-Modal Transformer Framework
for Drug–Target Interaction Prediction

Dataset:
---------
KIBA

Task:
------
Regression (KIBA score prediction)

Outputs:
--------
checkpoints/kiba/
outputs/kiba/

=============================================================
"""

# ============================================================
# Imports
# ============================================================

import os
import sys
import time
import yaml
import json
import random
import logging
from pathlib import Path

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torch.cuda.amp import (
    GradScaler,
    autocast
)

from torch.utils.tensorboard import SummaryWriter

from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ============================================================
# Project Root
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# ============================================================
# Model Imports
# ============================================================

from models.full_model import build_model

from datasets.dti_dataset import load_kiba_split

from datasets.collate import (
    dti_collate_fn,
    move_batch_to_device
)

from datasets.protein_dataset import (
    ProteinRepository,
    ProteinEmbeddingCache
)

# ============================================================
# Constants
# ============================================================

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Config Loader
# ============================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# Logger
# ============================================================

def setup_logger(output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "training.log"

    logger = logging.getLogger("kiba")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


# ============================================================
# Device
# ============================================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Protein Repository
# ============================================================

def load_protein_repository():

    repo_path = (
        ROOT_DIR /
        "datasets" /
        "processed" /
        "protein_repository.pkl"
    )

    if not repo_path.exists():
        raise FileNotFoundError(
            "Run preprocess.py first (protein_repository missing)"
        )

    return ProteinRepository.load(repo_path)


# ============================================================
# ESM Cache
# ============================================================

def load_esm_cache():

    cache_path = (
        ROOT_DIR /
        "datasets" /
        "processed" /
        "esm_embeddings.pt"
    )

    if not cache_path.exists():
        raise FileNotFoundError(
            "ESM cache missing. Run precompute_esm_embeddings.py"
        )

    return ProteinEmbeddingCache.load(cache_path)


# ============================================================
# DataLoader Builder
# ============================================================

def build_dataloaders(config):

    base = ROOT_DIR / "datasets" / "processed"

    train = load_kiba_split(base, "train")
    val   = load_kiba_split(base, "val")
    test  = load_kiba_split(base, "test")

    batch_size = config["training"]["batch_size"]
    workers = config["training"]["num_workers"]

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=dti_collate_fn
    )

    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=dti_collate_fn
    )

    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=dti_collate_fn
    )

    return train_loader, val_loader, test_loader


# ============================================================
# Model Initialization
# ============================================================

def initialize_model(config, device):

    repo = load_protein_repository()
    esm = load_esm_cache()

    model = build_model(
        config,
        protein_repository=repo,
        embedding_cache=esm
    )

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model


# ============================================================
# Output Directories
# ============================================================

def create_dirs():

    checkpoint_dir = ROOT_DIR / "checkpoints" / "kiba"
    output_dir = ROOT_DIR / "outputs" / "kiba"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir, output_dir


# ============================================================
# TensorBoard
# ============================================================

def create_writer():

    log_dir = ROOT_DIR / "outputs" / "kiba" / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)

    return SummaryWriter(log_dir=log_dir)

# ============================================================
# Loss Function
# ============================================================

def build_loss_function(config):

    loss_type = config["training"].get("loss", "mse").lower()

    if loss_type == "huber":
        return nn.HuberLoss()

    if loss_type == "smooth_l1":
        return nn.SmoothL1Loss()

    return nn.MSELoss()


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(model, config):

    lr = config["training"]["lr"]

    weight_decay = config["training"].get("weight_decay", 1e-5)

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )


# ============================================================
# Scheduler
# ============================================================

def build_scheduler(optimizer, config):

    name = config["training"].get("scheduler", "cosine").lower()

    if name == "cosine":

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"]
        )

    if name == "plateau":

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )

    return None


# ============================================================
# Metrics
# ============================================================

def concordance_index(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = 0
    h_sum = 0.0

    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):

            if y_true[i] == y_true[j]:
                continue

            n += 1

            if (y_pred[i] == y_pred[j]):
                h_sum += 0.5

            elif (
                (y_true[i] > y_true[j] and y_pred[i] > y_pred[j])
                or
                (y_true[i] < y_true[j] and y_pred[i] < y_pred[j])
            ):
                h_sum += 1

    return float(h_sum / n) if n > 0 else 0.0


# ============================================================
# Regression Metrics
# ============================================================

def compute_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {}

    metrics["mse"] = mean_squared_error(y_true, y_pred)
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["r2"] = r2_score(y_true, y_pred)

    try:
        metrics["pearson"] = float(pearsonr(y_true, y_pred)[0])
    except:
        metrics["pearson"] = 0.0

    try:
        metrics["spearman"] = float(spearmanr(y_true, y_pred)[0])
    except:
        metrics["spearman"] = 0.0

    metrics["ci"] = concordance_index(y_true, y_pred)

    return metrics


# ============================================================
# Epoch Statistics
# ============================================================

class EpochStats:

    def __init__(self):
        self.losses = []
        self.y_true = []
        self.y_pred = []

    def update(self, loss, y, pred):

        self.losses.append(loss)

        self.y_true.extend(
            y.detach().cpu().numpy().flatten().tolist()
        )

        self.y_pred.extend(
            pred.detach().cpu().numpy().flatten().tolist()
        )

    def summary(self):

        metrics = compute_metrics(self.y_true, self.y_pred)

        metrics["loss"] = float(np.mean(self.losses))

        return metrics


# ============================================================
# Validation Loop
# ============================================================

@torch.no_grad()
def validate_epoch(model, loader, criterion, device):

    model.eval()

    stats = EpochStats()

    for batch in tqdm(loader, desc="Validation"):

        batch = move_batch_to_device(batch, device)

        outputs = model(
            batch["graph_batch"],
            batch["protein_ids"]
        )

        pred = outputs["prediction"]

        loss = criterion(pred, batch["labels"])

        stats.update(loss.item(), batch["labels"], pred)

    return stats.summary()


# ============================================================
# Logging Helpers
# ============================================================

def log_metrics(logger, metrics, prefix=""):

    logger.info(
        f"{prefix} "
        f"Loss={metrics['loss']:.4f} | "
        f"RMSE={metrics['rmse']:.4f} | "
        f"MAE={metrics['mae']:.4f} | "
        f"R2={metrics['r2']:.4f} | "
        f"Pearson={metrics['pearson']:.4f} | "
        f"Spearman={metrics['spearman']:.4f} | "
        f"CI={metrics['ci']:.4f}"
    )


# ============================================================
# TensorBoard Logging
# ============================================================

def tb_log(writer, metrics, epoch, split):

    for k, v in metrics.items():
        writer.add_scalar(f"{split}/{k}", v, epoch)


# ============================================================
# AMP Utilities
# ============================================================

def create_scaler():
    return GradScaler()


# ============================================================
# LR Utility
# ============================================================

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


# ============================================================
# Scheduler Step
# ============================================================

def step_scheduler(scheduler, metric=None):

    if scheduler is None:
        return

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()

# ============================================================
# Training Epoch
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    config
):

    model.train()

    stats = EpochStats()

    grad_clip = config["training"].get("gradient_clip", 1.0)

    for batch in tqdm(loader, desc="Training"):

        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        # ----------------------------------------------------
        # Forward (AMP)
        # ----------------------------------------------------
        with autocast():

            outputs = model(
                batch["graph_batch"],
                batch["protein_ids"]
            )

            pred = outputs["prediction"]

            loss = criterion(pred, batch["labels"])

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip
        )

        scaler.step(optimizer)
        scaler.update()

        # ----------------------------------------------------
        # Update Stats
        # ----------------------------------------------------
        stats.update(
            loss.item(),
            batch["labels"],
            pred
        )

    return stats.summary()


# ============================================================
# Epoch Runner
# ============================================================

def run_epoch(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scaler,
    scheduler,
    device,
    config,
    epoch,
    logger,
    writer
):

    logger.info("\n" + "=" * 70)
    logger.info(f"Epoch {epoch+1}")
    logger.info("=" * 70)

    # ---------------- TRAIN ----------------
    train_metrics = train_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        scaler,
        device,
        config
    )

    log_metrics(logger, train_metrics, "[TRAIN]")

    tb_log(writer, train_metrics, epoch, "train")

    # ---------------- VALID ----------------
    val_metrics = validate_epoch(
        model,
        val_loader,
        criterion,
        device
    )

    log_metrics(logger, val_metrics, "[VAL]")

    tb_log(writer, val_metrics, epoch, "val")

    # ---------------- LR ----------------
    lr = get_lr(optimizer)
    writer.add_scalar("lr", lr, epoch)

    logger.info(f"LR: {lr:.6e}")

    # ---------------- Scheduler ----------------
    step_scheduler(scheduler, val_metrics["ci"])

    return train_metrics, val_metrics


# ============================================================
# Best Model Check
# ============================================================

def is_best_model(current, best):

    if best is None:
        return True

    return current["ci"] > best["ci"]


# ============================================================
# Epoch Summary
# ============================================================

def print_epoch_summary(epoch, train_metrics, val_metrics):

    print("\n" + "=" * 70)
    print(f"EPOCH {epoch+1} SUMMARY")
    print("=" * 70)

    print("\nTRAIN:")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nVAL:")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    print("=" * 70)


# ============================================================
# Model Utility
# ============================================================

def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, logger):

    logger.info("\n" + "=" * 70)
    logger.info("MODEL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Trainable Params: {count_parameters(model):,}")
    logger.info("=" * 70)


# ============================================================
# Training State
# ============================================================

def init_state():

    return {
        "best_metrics": None,
        "best_epoch": -1,
        "no_improve": 0
    }

# ============================================================
# Checkpoint Saving
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    metrics,
    path
):

    if isinstance(model, nn.DataParallel):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    ckpt = {
        "epoch": epoch,
        "model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics
    }

    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()

    if scaler is not None:
        ckpt["scaler_state"] = scaler.state_dict()

    torch.save(ckpt, path)


# ============================================================
# Checkpoint Loading
# ============================================================

def load_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    path,
    device
):

    ckpt = torch.load(path, map_location=device)

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt["model_state"])

    optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])

    start_epoch = ckpt["epoch"] + 1
    best_metrics = ckpt.get("metrics", None)

    return start_epoch, best_metrics


# ============================================================
# Find Latest Checkpoint
# ============================================================

def find_latest_checkpoint(checkpoint_dir):

    files = sorted(checkpoint_dir.glob("epoch_*.pth"))

    if len(files) == 0:
        return None

    return files[-1]


# ============================================================
# Save Best Model
# ============================================================

def save_best_model(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    metrics,
    checkpoint_dir
):

    path = checkpoint_dir / "best_model.pth"

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        metrics,
        path
    )

    return path


# ============================================================
# Save Epoch Checkpoint
# ============================================================

def save_epoch_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    metrics,
    checkpoint_dir
):

    path = checkpoint_dir / f"epoch_{epoch:03d}.pth"

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        metrics,
        path
    )


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:

    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best = None
        self.stop = False

    def step(self, score):

        if self.best is None:
            self.best = score
            return False

        if score > self.best:
            self.best = score
            self.counter = 0
            return False

        self.counter += 1

        if self.counter >= self.patience:
            self.stop = True

        return self.stop


# ============================================================
# Training History
# ============================================================

class History:

    def __init__(self):
        self.records = []

    def add(self, epoch, train_metrics, val_metrics):

        self.records.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics
        })

    def best_epoch(self):

        if not self.records:
            return None

        return max(self.records, key=lambda x: x["val"]["ci"])["epoch"]

    def best_ci(self):

        if not self.records:
            return 0.0

        return max(r["val"]["ci"] for r in self.records)


# ============================================================
# Export CSV
# ============================================================

def export_history_csv(history, out_dir):

    import pandas as pd

    rows = []

    for r in history.records:

        row = {"epoch": r["epoch"]}

        for k, v in r["train"].items():
            row[f"train_{k}"] = v

        for k, v in r["val"].items():
            row[f"val_{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)

    path = out_dir / "history.csv"
    df.to_csv(path, index=False)

    return path


# ============================================================
# Export JSON Report
# ============================================================

def save_report(history, out_dir):

    import json

    report = {
        "best_epoch": history.best_epoch(),
        "best_ci": history.best_ci(),
        "total_epochs": len(history.records)
    }

    path = out_dir / "report.json"

    with open(path, "w") as f:
        json.dump(report, f, indent=4)

    return path


# ============================================================
# Experiment Metadata
# ============================================================

def save_metadata(config, out_dir):

    import json
    from datetime import datetime

    meta = {
        "time": datetime.now().isoformat(),
        "dataset": "KIBA",
        "task": "Regression",
        "metric": "CI",
        "config": config
    }

    path = out_dir / "metadata.json"

    with open(path, "w") as f:
        json.dump(meta, f, indent=4)

    return path

# ============================================================
# Training Manager
# ============================================================

class TrainingManager:

    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        scaler,
        train_loader,
        val_loader,
        device,
        logger,
        writer,
        checkpoint_dir,
        output_dir
    ):

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device
        self.logger = logger
        self.writer = writer

        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir

        self.criterion = build_loss_function(config)

        self.history = History()

        self.early_stopping = EarlyStopping(
            patience=config["training"].get("early_stopping_patience", 15)
        )

        self.start_epoch = 0
        self.best_metrics = None
        self.best_epoch = -1

    # ========================================================
    # Resume Training
    # ========================================================

    def resume_if_available(self):

        if not self.config["training"].get("resume", False):
            return

        ckpt = find_latest_checkpoint(self.checkpoint_dir)

        if ckpt is None:
            self.logger.info("No checkpoint found for resume.")
            return

        self.logger.info(f"Resuming from checkpoint: {ckpt}")

        self.start_epoch, self.best_metrics = load_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            ckpt,
            self.device
        )

        self.logger.info(f"Resumed at epoch {self.start_epoch}")

    # ========================================================
    # Save Epoch Checkpoint
    # ========================================================

    def save_epoch(self, epoch, metrics):

        save_epoch_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            epoch,
            metrics,
            self.checkpoint_dir
        )

    # ========================================================
    # Best Model Update
    # ========================================================

    def update_best_model(self, epoch, metrics):

        improved = (
            self.best_metrics is None
            or metrics["ci"] > self.best_metrics["ci"]
        )

        if not improved:
            return False

        self.best_metrics = metrics
        self.best_epoch = epoch

        path = save_best_model(
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            epoch,
            metrics,
            self.checkpoint_dir
        )

        self.logger.info(f"Best model saved at {path}")

        return True

    # ========================================================
    # Early Stopping
    # ========================================================

    def check_early_stopping(self, metrics):

        return self.early_stopping.step(metrics["ci"])

    # ========================================================
    # Training Loop
    # ========================================================

    def train(self):

        epochs = self.config["training"]["epochs"]

        self.resume_if_available()

        print_model_summary(self.model, self.logger)

        self.logger.info("\nStarting KIBA training...\n")

        for epoch in range(self.start_epoch, epochs):

            train_metrics, val_metrics = run_epoch(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                scaler=self.scaler,
                scheduler=self.scheduler,
                device=self.device,
                config=self.config,
                epoch=epoch,
                logger=self.logger,
                writer=self.writer
            )

            self.history.add(epoch, train_metrics, val_metrics)

            print_epoch_summary(epoch, train_metrics, val_metrics)

            self.save_epoch(epoch, val_metrics)

            if self.update_best_model(epoch, val_metrics):
                self.logger.info("Best model updated.")

            if self.check_early_stopping(val_metrics):
                self.logger.info("Early stopping triggered.")
                break

        self.finish_training()

    # ========================================================
    # Finalization
    # ========================================================

    def finish_training(self):

        self.writer.flush()
        self.writer.close()

        csv_path = export_history_csv(self.history, self.output_dir)
        json_path = save_report(self.history, self.output_dir)
        meta_path = save_metadata(self.config, self.output_dir)

        self.logger.info("\nTraining Complete")
        self.logger.info(f"Best Epoch: {self.best_epoch}")
        self.logger.info(f"Best CI: {self.best_metrics['ci']:.5f}" if self.best_metrics else "No best model found")

        self.logger.info(f"CSV saved: {csv_path}")
        self.logger.info(f"Report saved: {json_path}")
        self.logger.info(f"Metadata saved: {meta_path}")

# ============================================================
# Load Best Model
# ============================================================

def load_best_model(
    model,
    checkpoint_dir,
    device
):

    best_model = (
        checkpoint_dir /
        "best_model.pth"
    )

    if not best_model.exists():
        raise FileNotFoundError(
            f"Best model not found: {best_model}"
        )

    checkpoint = torch.load(
        best_model,
        map_location=device
    )

    if isinstance(model, nn.DataParallel):

        model.module.load_state_dict(
            checkpoint["model_state"]
        )

    else:

        model.load_state_dict(
            checkpoint["model_state"]
        )

    return checkpoint


# ============================================================
# Test Evaluation
# ============================================================

@torch.no_grad()
def evaluate_test_set(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    stats = EpochStats()

    for batch in tqdm(
        loader,
        desc="Testing"
    ):

        batch = move_batch_to_device(
            batch,
            device
        )

        outputs = model(
            batch["graph_batch"],
            batch["protein_ids"]
        )

        pred = outputs["prediction"]

        loss = criterion(
            pred,
            batch["labels"]
        )

        stats.update(
            loss.item(),
            batch["labels"],
            pred
        )

    return stats.summary()


# ============================================================
# Collect Predictions
# ============================================================

@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device
):

    model.eval()

    y_true = []
    y_pred = []

    for batch in tqdm(
        loader,
        desc="Collect Predictions"
    ):

        batch = move_batch_to_device(
            batch,
            device
        )

        outputs = model(
            batch["graph_batch"],
            batch["protein_ids"]
        )

        pred = outputs["prediction"]

        y_true.extend(
            batch["labels"]
            .cpu()
            .numpy()
            .flatten()
            .tolist()
        )

        y_pred.extend(
            pred.cpu()
            .numpy()
            .flatten()
            .tolist()
        )

    return (
        np.asarray(y_true),
        np.asarray(y_pred)
    )


# ============================================================
# Save Predictions
# ============================================================

def save_predictions_csv(
    y_true,
    y_pred,
    output_dir
):

    import pandas as pd

    df = pd.DataFrame({

        "true_kiba":
            y_true,

        "predicted_kiba":
            y_pred,

        "error":
            y_pred - y_true
    })

    path = (
        output_dir /
        "test_predictions.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    return path


# ============================================================
# Scatter Plot Export
# ============================================================

def export_scatter_data(
    y_true,
    y_pred,
    output_dir
):

    import pandas as pd

    df = pd.DataFrame({

        "true":
            y_true,

        "predicted":
            y_pred
    })

    path = (
        output_dir /
        "scatter_data.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    return path


# ============================================================
# Test Workflow
# ============================================================

def run_test_workflow(
    model,
    test_loader,
    checkpoint_dir,
    output_dir,
    device,
    logger,
    config
):

    logger.info(
        "\nRunning KIBA Test Evaluation..."
    )

    load_best_model(
        model,
        checkpoint_dir,
        device
    )

    criterion = build_loss_function(
        config
    )

    metrics = evaluate_test_set(
        model,
        test_loader,
        criterion,
        device
    )

    log_metrics(
        logger,
        metrics,
        prefix="[TEST]"
    )

    y_true, y_pred = collect_predictions(
        model,
        test_loader,
        device
    )

    pred_file = save_predictions_csv(
        y_true,
        y_pred,
        output_dir
    )

    scatter_file = export_scatter_data(
        y_true,
        y_pred,
        output_dir
    )

    logger.info(
        f"Predictions saved: {pred_file}"
    )

    logger.info(
        f"Scatter data saved: {scatter_file}"
    )

    logger.info(
        f"Final CI: {metrics['ci']:.5f}"
    )

    logger.info(
        f"Final RMSE: {metrics['rmse']:.5f}"
    )

    logger.info(
        f"Final Pearson: {metrics['pearson']:.5f}"
    )

    return metrics


# ============================================================
# Runtime Report
# ============================================================

def print_runtime_report(
    seconds
):

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

    print("\n")
    print("=" * 80)
    print("TRAINING RUNTIME")
    print("=" * 80)

    print(
        f"{h:02d}:{m:02d}:{s:02d}"
    )

    print("=" * 80)


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed_everything()

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    config = load_config(
        ROOT_DIR /
        "configs" /
        "kiba.yaml"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    checkpoint_dir, output_dir = (
        create_dirs()
    )

    # --------------------------------------------------------
    # Logger
    # --------------------------------------------------------

    logger = setup_logger(
        output_dir
    )

    logger.info(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # TensorBoard
    # --------------------------------------------------------

    writer = create_writer()

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    save_metadata(
        config,
        output_dir
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader, val_loader, test_loader = (
        build_dataloaders(
            config
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = initialize_model(
        config,
        device
    )

    optimizer = build_optimizer(
        model,
        config
    )

    scheduler = build_scheduler(
        optimizer,
        config
    )

    scaler = create_scaler()

    # --------------------------------------------------------
    # Training Manager
    # --------------------------------------------------------

    manager = TrainingManager(

        config=config,

        model=model,

        optimizer=optimizer,

        scheduler=scheduler,

        scaler=scaler,

        train_loader=train_loader,

        val_loader=val_loader,

        device=device,

        logger=logger,

        writer=writer,

        checkpoint_dir=checkpoint_dir,

        output_dir=output_dir
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    start_time = time.time()

    manager.train()

    elapsed = (
        time.time() -
        start_time
    )

    print_runtime_report(
        elapsed
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_metrics = run_test_workflow(

        model=model,

        test_loader=test_loader,

        checkpoint_dir=checkpoint_dir,

        output_dir=output_dir,

        device=device,

        logger=logger,

        config=config
    )

    logger.info(
        "\nKIBA Pipeline Complete"
    )

    logger.info(
        f"Final Test CI: "
        f"{test_metrics['ci']:.5f}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()