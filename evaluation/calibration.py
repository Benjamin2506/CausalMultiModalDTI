#!/usr/bin/env python3

"""
=============================================================
Calibration Analysis

Causal Multi-Modal DTI Framework

Features
--------
Expected Calibration Error (ECE)
Maximum Calibration Error (MCE)
Brier Score
Reliability Diagrams
Temperature Scaling

=============================================================
"""

from __future__ import annotations

import json
import numpy as np

from pathlib import Path

from sklearn.metrics import (
    brier_score_loss
)

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# ============================================================
# ECE
# ============================================================

def expected_calibration_error(
    y_true,
    y_prob,
    n_bins=15
):
    """
    Expected Calibration Error.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(
        0,
        1,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        mask = (
            (y_prob >= bins[i])
            &
            (y_prob < bins[i + 1])
        )

        if mask.sum() == 0:
            continue

        accuracy = np.mean(
            y_true[mask]
        )

        confidence = np.mean(
            y_prob[mask]
        )

        weight = (
            mask.sum()
            /
            len(y_true)
        )

        ece += (
            np.abs(
                accuracy
                - confidence
            )
            * weight
        )

    return float(ece)


# ============================================================
# MCE
# ============================================================

def maximum_calibration_error(
    y_true,
    y_prob,
    n_bins=15
):

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(
        0,
        1,
        n_bins + 1
    )

    max_error = 0.0

    for i in range(n_bins):

        mask = (
            (y_prob >= bins[i])
            &
            (y_prob < bins[i + 1])
        )

        if mask.sum() == 0:
            continue

        acc = np.mean(
            y_true[mask]
        )

        conf = np.mean(
            y_prob[mask]
        )

        error = abs(
            acc - conf
        )

        max_error = max(
            max_error,
            error
        )

    return float(max_error)


# ============================================================
# Brier Score
# ============================================================

def compute_brier_score(
    y_true,
    y_prob
):

    return float(
        brier_score_loss(
            y_true,
            y_prob
        )
    )


# ============================================================
# Reliability Diagram Data
# ============================================================

def reliability_data(
    y_true,
    y_prob,
    n_bins=15
):

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(
        0,
        1,
        n_bins + 1
    )

    accuracies = []
    confidences = []
    counts = []

    for i in range(n_bins):

        mask = (
            (y_prob >= bins[i])
            &
            (y_prob < bins[i + 1])
        )

        if mask.sum() == 0:

            accuracies.append(0)
            confidences.append(0)
            counts.append(0)

            continue

        accuracies.append(
            np.mean(
                y_true[mask]
            )
        )

        confidences.append(
            np.mean(
                y_prob[mask]
            )
        )

        counts.append(
            int(mask.sum())
        )

    return {

        "accuracies":
            accuracies,

        "confidences":
            confidences,

        "counts":
            counts
    }


# ============================================================
# Reliability Diagram
# ============================================================

def plot_reliability_diagram(
    y_true,
    y_prob,
    save_path,
    n_bins=15
):

    data = reliability_data(
        y_true,
        y_prob,
        n_bins
    )

    acc = data["accuracies"]
    conf = data["confidences"]

    plt.figure(
        figsize=(7, 7)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=2
    )

    plt.bar(
        conf,
        acc,
        width=0.05
    )

    plt.xlabel(
        "Confidence"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Reliability Diagram"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()


# ============================================================
# Confidence Histogram
# ============================================================

def plot_confidence_histogram(
    y_prob,
    save_path,
    bins=20
):

    plt.figure(
        figsize=(7, 5)
    )

    plt.hist(
        y_prob,
        bins=bins
    )

    plt.xlabel(
        "Confidence"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Confidence Histogram"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()


# ============================================================
# Temperature Scaling
# ============================================================

class TemperatureScaler(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.temperature = nn.Parameter(
            torch.ones(1)
        )

    def forward(
        self,
        logits
    ):

        return (
            logits
            /
            self.temperature
        )

    def predict_proba(
        self,
        logits
    ):

        scaled = self.forward(
            logits
        )

        return torch.sigmoid(
            scaled
        )


# ============================================================
# Calibration Metrics
# ============================================================

def calibration_report(
    y_true,
    y_prob
):

    report = {

        "ece":
            expected_calibration_error(
                y_true,
                y_prob
            ),

        "mce":
            maximum_calibration_error(
                y_true,
                y_prob
            ),

        "brier":
            compute_brier_score(
                y_true,
                y_prob
            )
    }

    return report


# ============================================================
# Save Report
# ============================================================

def save_calibration_report(
    report,
    output_path
):

    output_path = Path(
        output_path
    )

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )


# ============================================================
# Full Calibration Pipeline
# ============================================================

def run_calibration_analysis(
    y_true,
    y_prob,
    output_dir
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report = calibration_report(
        y_true,
        y_prob
    )

    save_calibration_report(

        report,

        output_dir
        /
        "calibration_report.json"
    )

    plot_reliability_diagram(

        y_true,

        y_prob,

        output_dir
        /
        "reliability_diagram.png"
    )

    plot_confidence_histogram(

        y_prob,

        output_dir
        /
        "confidence_histogram.png"
    )

    return report


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":

    np.random.seed(42)

    y_true = np.random.randint(
        0,
        2,
        1000
    )

    y_prob = np.random.rand(
        1000
    )

    report = run_calibration_analysis(

        y_true,

        y_prob,

        "calibration_test"
    )

    print(
        json.dumps(
            report,
            indent=4
        )
    )