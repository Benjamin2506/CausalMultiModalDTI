#!/usr/bin/env python3

"""
=============================================================
Calibration Curve Visualization

Causal Multi-Modal DTI Framework

Outputs
--------
reliability_diagram.png
calibration_curve.png
confidence_histogram.png
calibration_metrics.json
calibration_bins.csv

=============================================================
"""

from __future__ import annotations

import json
import argparse
import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# ============================================================
# Load Predictions
# ============================================================

def load_predictions(prediction_file):
    """
    Expected CSV:

    target,prediction

    Binary classification probabilities.
    """

    df = pd.read_csv(prediction_file)

    if "target" not in df.columns:
        raise ValueError(
            "Column 'target' not found."
        )

    if "prediction" not in df.columns:
        raise ValueError(
            "Column 'prediction' not found."
        )

    y_true = df["target"].values
    y_prob = df["prediction"].values

    return y_true, y_prob


# ============================================================
# ECE
# ============================================================

def expected_calibration_error(
    y_true,
    y_prob,
    n_bins=15
):

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
            abs(
                accuracy
                -
                confidence
            )
            *
            weight
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

    bins = np.linspace(
        0,
        1,
        n_bins + 1
    )

    mce = 0.0

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

        error = abs(
            accuracy
            -
            confidence
        )

        mce = max(
            mce,
            error
        )

    return float(mce)


# ============================================================
# Calibration Metrics
# ============================================================

def compute_calibration_metrics(
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

        "brier_score":
            float(
                brier_score_loss(
                    y_true,
                    y_prob
                )
            )
    }

    return report


# ============================================================
# Reliability Data
# ============================================================

def build_reliability_table(
    y_true,
    y_prob,
    n_bins=15
):

    bins = np.linspace(
        0,
        1,
        n_bins + 1
    )

    rows = []

    for i in range(n_bins):

        mask = (
            (y_prob >= bins[i])
            &
            (y_prob < bins[i + 1])
        )

        if mask.sum() == 0:

            rows.append({
                "bin": i,
                "count": 0,
                "accuracy": 0,
                "confidence": 0
            })

            continue

        rows.append({

            "bin":
                i,

            "count":
                int(mask.sum()),

            "accuracy":
                float(
                    np.mean(
                        y_true[mask]
                    )
                ),

            "confidence":
                float(
                    np.mean(
                        y_prob[mask]
                    )
                )
        })

    return pd.DataFrame(rows)


# ============================================================
# Reliability Diagram
# ============================================================

def plot_reliability_diagram(
    table,
    output_file
):

    plt.figure(
        figsize=(8, 8)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=2,
        label="Perfect Calibration"
    )

    plt.plot(
        table["confidence"],
        table["accuracy"],
        marker="o",
        linewidth=2,
        label="Model"
    )

    plt.xlabel(
        "Confidence",
        fontsize=14
    )

    plt.ylabel(
        "Accuracy",
        fontsize=14
    )

    plt.title(
        "Reliability Diagram",
        fontsize=16
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Calibration Curve
# ============================================================

def plot_calibration_curve(
    y_true,
    y_prob,
    output_file
):

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=15
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=2
    )

    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        linewidth=2
    )

    plt.xlabel(
        "Mean Predicted Probability"
    )

    plt.ylabel(
        "Fraction of Positives"
    )

    plt.title(
        "Calibration Curve"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Confidence Histogram
# ============================================================

def plot_confidence_histogram(
    y_prob,
    output_file
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        y_prob,
        bins=20
    )

    plt.xlabel(
        "Prediction Confidence"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Confidence Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Save Metrics
# ============================================================

def save_metrics(
    metrics,
    output_file
):

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


# ============================================================
# Full Pipeline
# ============================================================

def generate_calibration_figures(
    prediction_file,
    output_dir
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    y_true, y_prob = load_predictions(
        prediction_file
    )

    metrics = compute_calibration_metrics(
        y_true,
        y_prob
    )

    table = build_reliability_table(
        y_true,
        y_prob
    )

    # ------------------------------------------
    # Figures
    # ------------------------------------------

    plot_reliability_diagram(

        table,

        output_dir
        /
        "reliability_diagram.png"
    )

    plot_calibration_curve(

        y_true,

        y_prob,

        output_dir
        /
        "calibration_curve.png"
    )

    plot_confidence_histogram(

        y_prob,

        output_dir
        /
        "confidence_histogram.png"
    )

    # ------------------------------------------
    # Exports
    # ------------------------------------------

    table.to_csv(

        output_dir
        /
        "calibration_bins.csv",

        index=False
    )

    save_metrics(

        metrics,

        output_dir
        /
        "calibration_metrics.json"
    )

    print(
        f"Calibration figures saved to {output_dir}"
    )

    return metrics


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/calibration"
    )

    args = parser.parse_args()

    metrics = generate_calibration_figures(

        args.predictions,

        args.output_dir
    )

    print(
        json.dumps(
            metrics,
            indent=4
        )
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()