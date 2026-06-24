#!/usr/bin/env python3

"""
=============================================================
Unified Evaluation Script

Supports:
---------
BindingDB
DAVIS
KIBA

Usage
-----
python evaluation/test.py --dataset bindingdb
python evaluation/test.py --dataset davis
python evaluation/test.py --dataset kiba

=============================================================
"""

import os
import sys
import json
import yaml
import argparse

from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm

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
# Imports
# ============================================================

from models.full_model import (
    build_model
)

from datasets.dti_dataset import (

    load_bindingdb_split,

    load_davis_split,

    load_kiba_split
)

from datasets.collate import (

    dti_collate_fn,

    move_batch_to_device
)

from datasets.protein_dataset import (

    ProteinRepository,

    ProteinEmbeddingCache
)

from evaluation.metrics import (

    classification_metrics,

    regression_metrics
)

from torch.utils.data import DataLoader

# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ============================================================
# Config Loader
# ============================================================

def load_config(
    dataset_name
):

    config_path = (

        ROOT_DIR

        / "configs"

        / f"{dataset_name}.yaml"
    )

    with open(
        config_path,
        "r"
    ) as f:

        return yaml.safe_load(f)


# ============================================================
# Protein Repository
# ============================================================

def load_protein_repository():

    return ProteinRepository.load(

        ROOT_DIR

        / "datasets"

        / "processed"

        / "protein_repository.pkl"
    )


# ============================================================
# ESM Cache
# ============================================================

def load_esm_cache():

    return ProteinEmbeddingCache.load(

        ROOT_DIR

        / "datasets"

        / "processed"

        / "esm_embeddings.pt"
    )


# ============================================================
# Build Model
# ============================================================

def load_model(
    dataset_name,
    config
):

    repository = (
        load_protein_repository()
    )

    cache = (
        load_esm_cache()
    )

    model = build_model(

        config,

        protein_repository=
        repository,

        embedding_cache=
        cache
    )

    checkpoint_path = (

        ROOT_DIR

        / "checkpoints"

        / dataset_name

        / "best_model.pth"
    )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=DEVICE
    )

    state_key = None

    if "model_state" in checkpoint:
        state_key = "model_state"

    elif "model_state_dict" in checkpoint:
        state_key = "model_state_dict"

    else:
        raise RuntimeError(
            "Model state not found."
        )

    model.load_state_dict(
        checkpoint[state_key]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# Dataset Loader
# ============================================================

def build_test_loader(
    dataset_name,
    config
):

    processed_dir = (

        ROOT_DIR

        / "datasets"

        / "processed"
    )

    if dataset_name == "bindingdb":

        dataset = (
            load_bindingdb_split(
                processed_dir,
                "test"
            )
        )

    elif dataset_name == "davis":

        dataset = (
            load_davis_split(
                processed_dir,
                "test"
            )
        )

    elif dataset_name == "kiba":

        dataset = (
            load_kiba_split(
                processed_dir,
                "test"
            )
        )

    else:

        raise ValueError(
            dataset_name
        )

    loader = DataLoader(

        dataset,

        batch_size=
        config["training"][
            "batch_size"
        ],

        shuffle=False,

        collate_fn=
        dti_collate_fn,

        num_workers=
        config["training"][
            "num_workers"
        ]
    )

    return loader


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def run_inference(
    model,
    loader,
    dataset_name
):

    targets = []
    outputs = []

    for batch in tqdm(
        loader,
        desc="Inference"
    ):

        batch = (
            move_batch_to_device(
                batch,
                DEVICE
            )
        )

        result = model(

            batch[
                "graph_batch"
            ],

            batch[
                "protein_ids"
            ]
        )

        prediction = result[
            "prediction"
        ]

        if dataset_name == "bindingdb":

            prediction = torch.sigmoid(
                prediction
            )

        targets.extend(

            batch[
                "labels"
            ]

            .cpu()

            .numpy()

            .flatten()
        )

        outputs.extend(

            prediction

            .cpu()

            .numpy()

            .flatten()
        )

    return (

        np.asarray(targets),

        np.asarray(outputs)
    )


# ============================================================
# Compute Metrics
# ============================================================

def evaluate(
    dataset_name,
    targets,
    outputs
):

    if dataset_name == "bindingdb":

        return classification_metrics(
            targets,
            outputs
        )

    return regression_metrics(
        targets,
        outputs
    )


# ============================================================
# Export Predictions
# ============================================================

def export_predictions(
    dataset_name,
    targets,
    outputs
):

    output_dir = (

        ROOT_DIR

        / "outputs"

        / dataset_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame({

        "target":
            targets,

        "prediction":
            outputs
    })

    csv_path = (
        output_dir
        / "evaluation_predictions.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    return csv_path


# ============================================================
# Export Report
# ============================================================

def export_report(
    dataset_name,
    metrics
):

    output_dir = (

        ROOT_DIR

        / "outputs"

        / dataset_name
    )

    report_path = (
        output_dir
        / "evaluation_report.json"
    )

    with open(
        report_path,
        "w"
    ) as f:

        json.dump(

            metrics,

            f,

            indent=4
        )

    return report_path


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--dataset",

        required=True,

        choices=[

            "bindingdb",

            "davis",

            "kiba"
        ]
    )

    args = parser.parse_args()

    dataset_name = (
        args.dataset.lower()
    )

    print(
        f"\nEvaluating {dataset_name}"
    )

    config = load_config(
        dataset_name
    )

    model = load_model(
        dataset_name,
        config
    )

    loader = build_test_loader(
        dataset_name,
        config
    )

    targets, outputs = run_inference(

        model,

        loader,

        dataset_name
    )

    metrics = evaluate(

        dataset_name,

        targets,

        outputs
    )

    print("\nResults")

    for k, v in metrics.items():

        print(
            f"{k}: {v:.6f}"
        )

    pred_file = export_predictions(

        dataset_name,

        targets,

        outputs
    )

    report_file = export_report(

        dataset_name,

        metrics
    )

    print(
        f"\nPredictions: {pred_file}"
    )

    print(
        f"Report: {report_file}"
    )

    print(
        "\nEvaluation Complete."
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()