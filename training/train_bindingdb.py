#!/usr/bin/env python3

"""
=============================================================
BindingDB Training Script

Causal Multi-Modal Transformer Framework
with Uncertainty-Aware Explainability
for Drug–Target Interaction Prediction

Dataset:
---------
BindingDB

Task:
------
Binary Classification

Output:
--------
checkpoints/
outputs/

Author:
Roshan Kotkondawar
Kunal Nicose
=============================================================
"""

# ============================================================
# Imports
# ============================================================

import os
import sys
import yaml
import time
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

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from torch.utils.tensorboard import (
    SummaryWriter
)

# ============================================================
# Project Root
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.append(
    str(ROOT_DIR)
)

# ============================================================
# Repository Imports
# ============================================================

from models.full_model import (
    build_model
)

from datasets.dti_dataset import (
    load_bindingdb_split
)

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
# Seed Everything
# ============================================================

def seed_everything(
    seed=SEED
):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# Load YAML Config
# ============================================================

def load_config(
    config_path
):

    with open(
        config_path,
        "r"
    ) as f:

        config = yaml.safe_load(f)

    return config


# ============================================================
# Logging
# ============================================================

def setup_logger(
    output_dir
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_file = (
        output_dir
        / "training.log"
    )

    logging.basicConfig(

        level=logging.INFO,

        format=
        (
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),

        handlers=[

            logging.FileHandler(
                log_file
            ),

            logging.StreamHandler()
        ]
    )

    return logging.getLogger(
        "bindingdb"
    )


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    return device


# ============================================================
# Protein Repository
# ============================================================

def load_protein_repository():

    repository_file = (

        ROOT_DIR
        /
        "datasets"
        /
        "processed"
        /
        "protein_repository.pkl"
    )

    if not repository_file.exists():

        raise FileNotFoundError(

            "\nprotein_repository.pkl "
            "not found.\n"
            "Run preprocess.py first."
        )

    repository = (
        ProteinRepository.load(
            repository_file
        )
    )

    return repository


# ============================================================
# ESM Embedding Cache
# ============================================================

def load_esm_cache():

    cache_file = (

        ROOT_DIR
        /
        "datasets"
        /
        "processed"
        /
        "esm_embeddings.pt"
    )

    if not cache_file.exists():

        raise FileNotFoundError(

            "\nesm_embeddings.pt "
            "not found.\n"
            "Run:\n"
            "python scripts/"
            "precompute_esm_embeddings.py"
        )

    cache = (
        ProteinEmbeddingCache.load(
            cache_file
        )
    )

    return cache


# ============================================================
# Data Loaders
# ============================================================

def build_dataloaders(
    config
):

    processed_dir = (

        ROOT_DIR
        /
        "datasets"
        /
        "processed"
    )

    train_dataset = (
        load_bindingdb_split(
            processed_dir,
            split="train"
        )
    )

    val_dataset = (
        load_bindingdb_split(
            processed_dir,
            split="val"
        )
    )

    test_dataset = (
        load_bindingdb_split(
            processed_dir,
            split="test"
        )
    )

    batch_size = (
        config["training"][
            "batch_size"
        ]
    )

    num_workers = (
        config["training"][
            "num_workers"
        ]
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=True,

        collate_fn=
        dti_collate_fn
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=True,

        collate_fn=
        dti_collate_fn
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=True,

        collate_fn=
        dti_collate_fn
    )

    return (

        train_loader,

        val_loader,

        test_loader
    )


# ============================================================
# Build Model
# ============================================================

def initialize_model(
    config,
    device
):

    repository = (
        load_protein_repository()
    )

    esm_cache = (
        load_esm_cache()
    )

    model = build_model(

        config,

        protein_repository=
        repository,

        embedding_cache=
        esm_cache
    )

    model = model.to(
        device
    )

    if (
        torch.cuda.device_count()
        > 1
    ):

        model = (
            nn.DataParallel(
                model
            )
        )

    return model


# ============================================================
# Output Directories
# ============================================================

def create_output_dirs():

    checkpoint_dir = (

        ROOT_DIR
        /
        "checkpoints"
    )

    output_dir = (

        ROOT_DIR
        /
        "outputs"
        /
        "bindingdb"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return (
        checkpoint_dir,
        output_dir
    )


# ============================================================
# TensorBoard
# ============================================================

def create_tensorboard_writer():

    log_dir = (

        ROOT_DIR
        /
        "outputs"
        /
        "bindingdb"
        /
        "tensorboard"
    )

    writer = SummaryWriter(
        log_dir=log_dir
    )

    return writer

# ============================================================
# Loss Function
# ============================================================

def build_loss_function():

    criterion = nn.BCEWithLogitsLoss()

    return criterion


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(
    model,
    config
):

    lr = config["training"]["lr"]

    weight_decay = (
        config["training"]
        .get(
            "weight_decay",
            1e-5
        )
    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=lr,

        weight_decay=weight_decay
    )

    return optimizer


# ============================================================
# Scheduler
# ============================================================

def build_scheduler(
    optimizer,
    config
):

    scheduler_name = (

        config["training"]
        .get(
            "scheduler",
            "cosine"
        )
    )

    if scheduler_name == "cosine":

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(

                optimizer,

                T_max=
                config["training"][
                    "epochs"
                ]
            )
        )

    elif scheduler_name == "plateau":

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(

                optimizer,

                mode="max",

                factor=0.5,

                patience=5,

                verbose=True
            )
        )

    else:

        scheduler = None

    return scheduler


# ============================================================
# Probability Conversion
# ============================================================

def logits_to_probabilities(
    logits
):

    return torch.sigmoid(
        logits
    )


# ============================================================
# Prediction Conversion
# ============================================================

def probabilities_to_labels(
    probabilities,
    threshold=0.5
):

    return (
        probabilities
        >= threshold
    ).astype(
        np.int32
    )


# ============================================================
# Classification Metrics
# ============================================================

def compute_metrics(
    y_true,
    y_prob
):
    """
    Binary classification metrics.
    """

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )

    y_pred = (
        y_prob >= 0.5
    ).astype(
        np.int32
    )

    metrics = {}

    try:

        metrics["roc_auc"] = (
            roc_auc_score(
                y_true,
                y_prob
            )
        )

    except Exception:

        metrics["roc_auc"] = 0.0

    try:

        metrics["pr_auc"] = (
            average_precision_score(
                y_true,
                y_prob
            )
        )

    except Exception:

        metrics["pr_auc"] = 0.0

    metrics["accuracy"] = (
        accuracy_score(
            y_true,
            y_pred
        )
    )

    metrics["precision"] = (
        precision_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    metrics["recall"] = (
        recall_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    metrics["f1"] = (
        f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    return metrics


# ============================================================
# Epoch Statistics
# ============================================================

class EpochStatistics:

    def __init__(self):

        self.reset()

    def reset(self):

        self.losses = []

        self.targets = []

        self.probabilities = []

    def update(
        self,
        loss,
        targets,
        probabilities
    ):

        self.losses.append(
            float(loss)
        )

        self.targets.extend(

            targets
            .detach()
            .cpu()
            .numpy()
            .flatten()
            .tolist()
        )

        self.probabilities.extend(

            probabilities
            .detach()
            .cpu()
            .numpy()
            .flatten()
            .tolist()
        )

    def summary(self):

        metrics = compute_metrics(

            self.targets,

            self.probabilities
        )

        metrics["loss"] = float(
            np.mean(
                self.losses
            )
        )

        return metrics


# ============================================================
# Validation Loop
# ============================================================

@torch.no_grad()
def validate_epoch(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    stats = EpochStatistics()

    progress = tqdm(

        dataloader,

        desc="Validation",

        leave=False
    )

    for batch in progress:

        batch = (
            move_batch_to_device(
                batch,
                device
            )
        )

        graph_batch = (
            batch[
                "graph_batch"
            ]
        )

        protein_ids = (
            batch[
                "protein_ids"
            ]
        )

        labels = (
            batch[
                "labels"
            ]
        )

        outputs = model(

            graph_batch,

            protein_ids
        )

        logits = (
            outputs[
                "prediction"
            ]
        )

        loss = criterion(

            logits,

            labels
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

        stats.update(

            loss.item(),

            labels,

            probabilities
        )

    return stats.summary()


# ============================================================
# Metric Logging
# ============================================================

def log_metrics(
    logger,
    metrics,
    prefix=""
):

    logger.info(
        (
            f"{prefix}"
            f"Loss={metrics['loss']:.5f} | "
            f"AUC={metrics['roc_auc']:.5f} | "
            f"PRAUC={metrics['pr_auc']:.5f} | "
            f"ACC={metrics['accuracy']:.5f} | "
            f"PREC={metrics['precision']:.5f} | "
            f"REC={metrics['recall']:.5f} | "
            f"F1={metrics['f1']:.5f}"
        )
    )


# ============================================================
# TensorBoard Logging
# ============================================================

def tensorboard_log_metrics(
    writer,
    metrics,
    epoch,
    phase
):

    for key, value in metrics.items():

        writer.add_scalar(

            f"{phase}/{key}",

            value,

            epoch
        )


# ============================================================
# AMP Utilities
# ============================================================

def create_grad_scaler():

    scaler = GradScaler()

    return scaler


# ============================================================
# Learning Rate Helper
# ============================================================

def get_learning_rate(
    optimizer
):

    return optimizer.param_groups[
        0
    ]["lr"]


# ============================================================
# Scheduler Step
# ============================================================

def scheduler_step(
    scheduler,
    metric=None
):

    if scheduler is None:

        return

    if isinstance(

        scheduler,

        torch.optim.lr_scheduler.ReduceLROnPlateau
    ):

        scheduler.step(
            metric
        )

    else:

        scheduler.step()

# ============================================================
# Train One Epoch
# ============================================================

def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    scaler,
    device,
    config
):
    """
    Single training epoch.
    """

    model.train()

    stats = EpochStatistics()

    grad_clip = (
        config["training"]
        .get(
            "gradient_clip",
            1.0
        )
    )

    progress = tqdm(

        dataloader,

        desc="Training",

        leave=False
    )

    for batch_idx, batch in enumerate(
        progress
    ):

        batch = (
            move_batch_to_device(
                batch,
                device
            )
        )

        graph_batch = (
            batch[
                "graph_batch"
            ]
        )

        protein_ids = (
            batch[
                "protein_ids"
            ]
        )

        labels = (
            batch[
                "labels"
            ]
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward Pass
        # ----------------------------------------------------

        with autocast():

            outputs = model(

                graph_batch,

                protein_ids
            )

            logits = (
                outputs[
                    "prediction"
                ]
            )

            loss = criterion(

                logits,

                labels
            )

        # ----------------------------------------------------
        # Backward Pass
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()

        # ----------------------------------------------------
        # Gradient Clipping
        # ----------------------------------------------------

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            grad_clip
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        stats.update(

            loss.item(),

            labels,

            probabilities
        )

        # ----------------------------------------------------
        # Progress Bar
        # ----------------------------------------------------

        progress.set_postfix({

            "loss":

                f"{loss.item():.4f}",

            "lr":

                f"{get_learning_rate(optimizer):.2e}"
        })

    metrics = stats.summary()

    return metrics


# ============================================================
# Train + Validate Epoch
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
    """
    Complete epoch.
    """

    logger.info(
        "\n"
        + "=" * 70
    )

    logger.info(
        f"Epoch "
        f"{epoch + 1}"
    )

    logger.info(
        "=" * 70
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_metrics = train_epoch(

        model=model,

        dataloader=train_loader,

        optimizer=optimizer,

        criterion=criterion,

        scaler=scaler,

        device=device,

        config=config
    )

    log_metrics(

        logger,

        train_metrics,

        prefix="[TRAIN] "
    )

    tensorboard_log_metrics(

        writer,

        train_metrics,

        epoch,

        "train"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_metrics = validate_epoch(

        model=model,

        dataloader=val_loader,

        criterion=criterion,

        device=device
    )

    log_metrics(

        logger,

        val_metrics,

        prefix="[VALID] "
    )

    tensorboard_log_metrics(

        writer,

        val_metrics,

        epoch,

        "validation"
    )

    # --------------------------------------------------------
    # Learning Rate
    # --------------------------------------------------------

    current_lr = (
        get_learning_rate(
            optimizer
        )
    )

    writer.add_scalar(

        "learning_rate",

        current_lr,

        epoch
    )

    logger.info(

        f"Learning Rate: "
        f"{current_lr:.6e}"
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler_step(

        scheduler,

        val_metrics[
            "roc_auc"
        ]
    )

    return (

        train_metrics,

        val_metrics
    )


# ============================================================
# Metric Comparison
# ============================================================

def is_best_model(
    current_metrics,
    best_metrics
):
    """
    Determine whether current
    validation performance is best.
    """

    if best_metrics is None:

        return True

    current_auc = (
        current_metrics[
            "roc_auc"
        ]
    )

    best_auc = (
        best_metrics[
            "roc_auc"
        ]
    )

    return current_auc > best_auc


# ============================================================
# Epoch Summary
# ============================================================

def print_epoch_summary(
    epoch,
    train_metrics,
    val_metrics
):

    print("\n")

    print("=" * 70)

    print(
        f"Epoch "
        f"{epoch + 1} Summary"
    )

    print("=" * 70)

    print(
        "\nTraining"
    )

    for key, value in train_metrics.items():

        print(
            f"{key}: "
            f"{value:.5f}"
        )

    print(
        "\nValidation"
    )

    for key, value in val_metrics.items():

        print(
            f"{key}: "
            f"{value:.5f}"
        )

    print("=" * 70)


# ============================================================
# Parameter Count
# ============================================================

def count_parameters(
    model
):

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad
    )


# ============================================================
# Model Summary
# ============================================================

def print_model_summary(
    model,
    logger
):

    if isinstance(
        model,
        nn.DataParallel
    ):

        actual_model = (
            model.module
        )

    else:

        actual_model = model

    total_params = (
        count_parameters(
            actual_model
        )
    )

    logger.info(
        "\n"
        + "=" * 70
    )

    logger.info(
        "MODEL SUMMARY"
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        f"Trainable Parameters: "
        f"{total_params:,}"
    )

    logger.info(
        "=" * 70
    )


# ============================================================
# Training State
# ============================================================

def initialize_training_state():

    state = {

        "best_epoch":
            -1,

        "best_metrics":
            None,

        "epochs_without_improvement":
            0
    }

    return state

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
    checkpoint_path
):
    """
    Save training checkpoint.
    """

    if isinstance(
        model,
        nn.DataParallel
    ):

        model_state = (
            model.module.state_dict()
        )

    else:

        model_state = (
            model.state_dict()
        )

    checkpoint = {

        "epoch":
            epoch,

        "model_state_dict":
            model_state,

        "optimizer_state_dict":
            optimizer.state_dict(),

        "metrics":
            metrics
    }

    if scheduler is not None:

        checkpoint[
            "scheduler_state_dict"
        ] = (
            scheduler.state_dict()
        )

    if scaler is not None:

        checkpoint[
            "scaler_state_dict"
        ] = (
            scaler.state_dict()
        )

    torch.save(
        checkpoint,
        checkpoint_path
    )


# ============================================================
# Checkpoint Loading
# ============================================================

def load_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    checkpoint_path,
    device
):
    """
    Resume training.
    """

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device
    )

    if isinstance(
        model,
        nn.DataParallel
    ):

        model.module.load_state_dict(

            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(

            checkpoint[
                "model_state_dict"
            ]
        )

    optimizer.load_state_dict(

        checkpoint[
            "optimizer_state_dict"
        ]
    )

    if (

        scheduler is not None

        and

        "scheduler_state_dict"
        in checkpoint
    ):

        scheduler.load_state_dict(

            checkpoint[
                "scheduler_state_dict"
            ]
        )

    if (

        scaler is not None

        and

        "scaler_state_dict"
        in checkpoint
    ):

        scaler.load_state_dict(

            checkpoint[
                "scaler_state_dict"
            ]
        )

    start_epoch = (
        checkpoint["epoch"]
        + 1
    )

    metrics = (
        checkpoint.get(
            "metrics",
            {}
        )
    )

    return (

        start_epoch,

        metrics
    )


# ============================================================
# Latest Checkpoint
# ============================================================

def find_latest_checkpoint(
    checkpoint_dir
):
    """
    Locate latest checkpoint.
    """

    checkpoints = sorted(

        checkpoint_dir.glob(
            "epoch_*.pth"
        )
    )

    if len(checkpoints) == 0:

        return None

    return checkpoints[-1]


# ============================================================
# Best Model Save
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
    """
    Save best model.
    """

    best_file = (
        checkpoint_dir
        / "best_model.pth"
    )

    save_checkpoint(

        model=model,

        optimizer=optimizer,

        scheduler=scheduler,

        scaler=scaler,

        epoch=epoch,

        metrics=metrics,

        checkpoint_path=
        best_file
    )

    return best_file


# ============================================================
# Epoch Checkpoint Save
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
    """
    Save epoch checkpoint.
    """

    checkpoint_file = (

        checkpoint_dir
        /
        f"epoch_{epoch:03d}.pth"
    )

    save_checkpoint(

        model,

        optimizer,

        scheduler,

        scaler,

        epoch,

        metrics,

        checkpoint_file
    )


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:

    def __init__(
        self,
        patience=15
    ):

        self.patience = patience

        self.counter = 0

        self.best_score = None

        self.should_stop = False

    def step(
        self,
        current_score
    ):

        if self.best_score is None:

            self.best_score = (
                current_score
            )

            return False

        if current_score > self.best_score:

            self.best_score = (
                current_score
            )

            self.counter = 0

            return False

        self.counter += 1

        if (
            self.counter
            >=
            self.patience
        ):

            self.should_stop = True

        return self.should_stop


# ============================================================
# Training History
# ============================================================

class TrainingHistory:

    def __init__(self):

        self.history = []

    def add_epoch(
        self,
        epoch,
        train_metrics,
        val_metrics
    ):

        record = {

            "epoch":
                epoch,

            "train":
                train_metrics,

            "validation":
                val_metrics
        }

        self.history.append(
            record
        )

    def best_epoch(
        self
    ):

        if len(
            self.history
        ) == 0:

            return None

        best = max(

            self.history,

            key=lambda x:

            x["validation"][
                "roc_auc"
            ]
        )

        return best["epoch"]

    def best_auc(
        self
    ):

        if len(
            self.history
        ) == 0:

            return 0.0

        return max(

            x["validation"][
                "roc_auc"
            ]

            for x in self.history
        )


# ============================================================
# History Export CSV
# ============================================================

def export_history_csv(
    history,
    output_dir
):
    """
    Export training history.
    """

    import pandas as pd

    rows = []

    for record in history.history:

        row = {

            "epoch":
                record["epoch"]
        }

        for key, value in (

            record["train"]
            .items()
        ):

            row[
                f"train_{key}"
            ] = value

        for key, value in (

            record["validation"]
            .items()
        ):

            row[
                f"val_{key}"
            ] = value

        rows.append(
            row
        )

    df = pd.DataFrame(
        rows
    )

    csv_file = (

        output_dir
        /
        "training_history.csv"
    )

    df.to_csv(
        csv_file,
        index=False
    )

    return csv_file


# ============================================================
# Training Report JSON
# ============================================================

def save_training_report(
    history,
    output_dir
):
    """
    Save summary report.
    """

    import json

    report = {

        "best_epoch":
            history.best_epoch(),

        "best_auc":
            float(
                history.best_auc()
            ),

        "total_epochs":
            len(
                history.history
            )
    }

    report_file = (

        output_dir
        /
        "training_report.json"
    )

    with open(
        report_file,
        "w"
    ) as f:

        json.dump(

            report,

            f,

            indent=4
        )

    return report_file


# ============================================================
# Experiment Metadata
# ============================================================

def save_experiment_metadata(
    config,
    output_dir
):
    """
    Save experiment config.
    """

    import json
    from datetime import datetime

    metadata = {

        "timestamp":

            datetime.now()
            .isoformat(),

        "config":

            config
    }

    metadata_file = (

        output_dir
        /
        "experiment_metadata.json"
    )

    with open(
        metadata_file,
        "w"
    ) as f:

        json.dump(

            metadata,

            f,

            indent=4
        )

    return metadata_file

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

        self.criterion = (
            build_loss_function()
        )

        self.history = (
            TrainingHistory()
        )

        self.early_stopping = (
            EarlyStopping(
                patience=
                config["training"].get(
                    "early_stopping_patience",
                    15
                )
            )
        )

        self.start_epoch = 0

        self.best_metrics = None

        self.best_epoch = -1

    # ========================================================
    # Resume Training
    # ========================================================

    def resume_if_available(
        self
    ):

        resume = (
            self.config["training"]
            .get(
                "resume",
                False
            )
        )

        if not resume:

            return

        checkpoint = (
            find_latest_checkpoint(
                self.checkpoint_dir
            )
        )

        if checkpoint is None:

            self.logger.info(
                "No checkpoint found."
            )

            return

        self.logger.info(
            f"Resuming from:\n"
            f"{checkpoint}"
        )

        (
            self.start_epoch,
            metrics
        ) = load_checkpoint(

            self.model,

            self.optimizer,

            self.scheduler,

            self.scaler,

            checkpoint,

            self.device
        )

        self.logger.info(
            f"Starting at epoch "
            f"{self.start_epoch}"
        )

        if metrics:

            self.best_metrics = metrics

    # ========================================================
    # Run Training
    # ========================================================

    def train(
        self
    ):

        epochs = (
            self.config["training"][
                "epochs"
            ]
        )

        self.resume_if_available()

        print_model_summary(

            self.model,

            self.logger
        )

        self.logger.info(
            "\nStarting training..."
        )

        for epoch in range(

            self.start_epoch,

            epochs
        ):

            (
                train_metrics,

                val_metrics

            ) = run_epoch(

                model=self.model,

                train_loader=
                self.train_loader,

                val_loader=
                self.val_loader,

                optimizer=
                self.optimizer,

                criterion=
                self.criterion,

                scaler=
                self.scaler,

                scheduler=
                self.scheduler,

                device=
                self.device,

                config=
                self.config,

                epoch=epoch,

                logger=
                self.logger,

                writer=
                self.writer
            )

            self.history.add_epoch(

                epoch,

                train_metrics,

                val_metrics
            )

            print_epoch_summary(

                epoch,

                train_metrics,

                val_metrics
            )

            self.save_epoch(

                epoch,

                val_metrics
            )

            if self.handle_best_model(

                epoch,

                val_metrics
            ):

                self.logger.info(
                    "\nBest model updated."
                )

            if self.check_early_stop(

                val_metrics
            ):

                self.logger.info(
                    "\nEarly stopping triggered."
                )

                break

        self.training_complete()

    # ========================================================
    # Save Epoch Checkpoint
    # ========================================================

    def save_epoch(
        self,
        epoch,
        metrics
    ):

        save_epoch_checkpoint(

            model=self.model,

            optimizer=
            self.optimizer,

            scheduler=
            self.scheduler,

            scaler=
            self.scaler,

            epoch=epoch,

            metrics=metrics,

            checkpoint_dir=
            self.checkpoint_dir
        )

    # ========================================================
    # Best Model Tracking
    # ========================================================

    def handle_best_model(
        self,
        epoch,
        metrics
    ):

        improved = (
            is_best_model(

                metrics,

                self.best_metrics
            )
        )

        if not improved:

            return False

        self.best_metrics = metrics

        self.best_epoch = epoch

        best_file = (
            save_best_model(

                model=self.model,

                optimizer=
                self.optimizer,

                scheduler=
                self.scheduler,

                scaler=
                self.scaler,

                epoch=epoch,

                metrics=metrics,

                checkpoint_dir=
                self.checkpoint_dir
            )
        )

        self.logger.info(
            f"Best model saved:\n"
            f"{best_file}"
        )

        return True

    # ========================================================
    # Early Stopping
    # ========================================================

    def check_early_stop(
        self,
        metrics
    ):

        score = (
            metrics["roc_auc"]
        )

        stop = (
            self.early_stopping.step(
                score
            )
        )

        return stop

    # ========================================================
    # Training Complete
    # ========================================================

    def training_complete(
        self
    ):

        self.writer.flush()

        self.writer.close()

        csv_file = (
            export_history_csv(

                self.history,

                self.output_dir
            )
        )

        report_file = (
            save_training_report(

                self.history,

                self.output_dir
            )
        )

        self.logger.info(
            "\nTraining Complete"
        )

        self.logger.info(
            f"Best Epoch: "
            f"{self.best_epoch}"
        )

        if self.best_metrics:

            self.logger.info(
                f"Best ROC-AUC: "
                f"{self.best_metrics['roc_auc']:.5f}"
            )

            self.logger.info(
                f"Best PR-AUC: "
                f"{self.best_metrics['pr_auc']:.5f}"
            )

            self.logger.info(
                f"Best F1: "
                f"{self.best_metrics['f1']:.5f}"
            )

        self.logger.info(
            f"CSV Saved:\n"
            f"{csv_file}"
        )

        self.logger.info(
            f"Report Saved:\n"
            f"{report_file}"
        )


# ============================================================
# Final Training Summary
# ============================================================

def print_training_summary(
    history
):

    print("\n")

    print("=" * 80)

    print(
        "TRAINING SUMMARY"
    )

    print("=" * 80)

    print(
        f"Epochs Completed: "
        f"{len(history.history)}"
    )

    print(
        f"Best Epoch: "
        f"{history.best_epoch()}"
    )

    print(
        f"Best ROC-AUC: "
        f"{history.best_auc():.5f}"
    )

    print("=" * 80)


# ============================================================
# Save Training Configuration
# ============================================================

def archive_training_configuration(
    config,
    output_dir
):

    save_experiment_metadata(

        config,

        output_dir
    )


# ============================================================
# Load Best Model
# ============================================================

def load_best_model(
    model,
    checkpoint_dir,
    device
):

    best_model_file = (
        checkpoint_dir
        / "best_model.pth"
    )

    if not best_model_file.exists():

        raise FileNotFoundError(
            "\nBest model not found."
        )

    checkpoint = torch.load(

        best_model_file,

        map_location=device
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    if isinstance(
        model,
        nn.DataParallel
    ):

        model.module.load_state_dict(
            state_dict
        )

    else:

        model.load_state_dict(
            state_dict
        )

    return checkpoint


# ============================================================
# Test Evaluation
# ============================================================

@torch.no_grad()
def evaluate_test_set(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    stats = EpochStatistics()

    progress = tqdm(

        dataloader,

        desc="Testing",

        leave=False
    )

    for batch in progress:

        batch = move_batch_to_device(
            batch,
            device
        )

        graph_batch = batch[
            "graph_batch"
        ]

        protein_ids = batch[
            "protein_ids"
        ]

        labels = batch[
            "labels"
        ]

        outputs = model(

            graph_batch,

            protein_ids
        )

        logits = outputs[
            "prediction"
        ]

        loss = criterion(

            logits,

            labels
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

        stats.update(

            loss.item(),

            labels,

            probabilities
        )

    return stats.summary()


# ============================================================
# Raw Predictions
# ============================================================

@torch.no_grad()
def collect_predictions(
    model,
    dataloader,
    device
):

    model.eval()

    y_true = []

    y_prob = []

    progress = tqdm(

        dataloader,

        desc="Predictions",

        leave=False
    )

    for batch in progress:

        batch = move_batch_to_device(
            batch,
            device
        )

        outputs = model(

            batch["graph_batch"],

            batch["protein_ids"]
        )

        probabilities = (
            torch.sigmoid(
                outputs[
                    "prediction"
                ]
            )
        )

        y_true.extend(

            batch["labels"]

            .cpu()

            .numpy()

            .flatten()
        )

        y_prob.extend(

            probabilities

            .cpu()

            .numpy()

            .flatten()
        )

    return (

        np.asarray(y_true),

        np.asarray(y_prob)
    )


# ============================================================
# Export Predictions
# ============================================================

def save_predictions_csv(
    y_true,
    y_prob,
    output_dir
):

    import pandas as pd

    df = pd.DataFrame({

        "true_label":
            y_true,

        "probability":
            y_prob,

        "prediction":
            (
                y_prob >= 0.5
            ).astype(int)
    })

    file_path = (
        output_dir
        / "test_predictions.csv"
    )

    df.to_csv(
        file_path,
        index=False
    )

    return file_path


# ============================================================
# ROC Curve Data
# ============================================================

def export_roc_data(
    y_true,
    y_prob,
    output_dir
):

    from sklearn.metrics import (
        roc_curve
    )

    import pandas as pd

    fpr, tpr, thresholds = (
        roc_curve(
            y_true,
            y_prob
        )
    )

    df = pd.DataFrame({

        "fpr": fpr,

        "tpr": tpr,

        "threshold":
            thresholds
    })

    file_path = (
        output_dir
        / "roc_curve.csv"
    )

    df.to_csv(
        file_path,
        index=False
    )

    return file_path


# ============================================================
# PR Curve Data
# ============================================================

def export_pr_data(
    y_true,
    y_prob,
    output_dir
):

    from sklearn.metrics import (
        precision_recall_curve
    )

    import pandas as pd

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            y_prob
        )
    )

    df = pd.DataFrame({

        "precision":
            precision[:-1],

        "recall":
            recall[:-1],

        "threshold":
            thresholds
    })

    file_path = (
        output_dir
        / "pr_curve.csv"
    )

    df.to_csv(
        file_path,
        index=False
    )

    return file_path


# ============================================================
# Confusion Matrix
# ============================================================

def export_confusion_matrix(
    y_true,
    y_prob,
    output_dir
):

    from sklearn.metrics import (
        confusion_matrix
    )

    import pandas as pd

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    df = pd.DataFrame(
        cm
    )

    file_path = (
        output_dir
        / "confusion_matrix.csv"
    )

    df.to_csv(
        file_path,
        index=False
    )

    return file_path


# ============================================================
# Test Workflow
# ============================================================

def run_test_workflow(
    model,
    test_loader,
    checkpoint_dir,
    output_dir,
    device,
    logger
):

    logger.info(
        "\nRunning Test Evaluation..."
    )

    load_best_model(

        model,

        checkpoint_dir,

        device
    )

    criterion = (
        build_loss_function()
    )

    metrics = (
        evaluate_test_set(

            model,

            test_loader,

            criterion,

            device
        )
    )

    log_metrics(

        logger,

        metrics,

        prefix="[TEST] "
    )

    y_true, y_prob = (
        collect_predictions(

            model,

            test_loader,

            device
        )
    )

    save_predictions_csv(

        y_true,

        y_prob,

        output_dir
    )

    export_roc_data(

        y_true,

        y_prob,

        output_dir
    )

    export_pr_data(

        y_true,

        y_prob,

        output_dir
    )

    export_confusion_matrix(

        y_true,

        y_prob,

        output_dir
    )

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    seed_everything()

    config = load_config(

        ROOT_DIR
        / "configs"
        / "bindingdb.yaml"
    )

    device = get_device()

    checkpoint_dir, output_dir = (
        create_output_dirs()
    )

    logger = setup_logger(
        output_dir
    )

    writer = (
        create_tensorboard_writer()
    )

    logger.info(
        f"Device: {device}"
    )

    archive_training_configuration(

        config,

        output_dir
    )

    (
        train_loader,

        val_loader,

        test_loader

    ) = build_dataloaders(
        config
    )

    model = initialize_model(

        config,

        device
    )

    optimizer = (
        build_optimizer(
            model,
            config
        )
    )

    scheduler = (
        build_scheduler(
            optimizer,
            config
        )
    )

    scaler = (
        create_grad_scaler()
    )

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

        checkpoint_dir=
        checkpoint_dir,

        output_dir=
        output_dir
    )

    start_time = time.time()

    manager.train()

    elapsed = (
        time.time()
        - start_time
    )

    logger.info(
        f"\nTraining Time: "
        f"{elapsed / 60:.2f} min"
    )

    test_metrics = (
        run_test_workflow(

            model,

            test_loader,

            checkpoint_dir,

            output_dir,

            device,

            logger
        )
    )

    logger.info(
        "\nTraining Pipeline Complete."
    )

    logger.info(
        f"Final Test ROC-AUC: "
        f"{test_metrics['roc_auc']:.5f}"
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()