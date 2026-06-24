#!/usr/bin/env python3

"""
=============================================================
Docking Validation Module

Causal Multi-Modal DTI Framework

Compares:
-----------
Model Predictions
vs
AutoDock Vina Binding Affinities

Outputs:
-----------
validation_report.json
validation_results.csv
scatter_plot.png

=============================================================
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from pathlib import Path

from scipy.stats import (
    pearsonr,
    spearmanr,
    kendalltau
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

import matplotlib.pyplot as plt

# ============================================================
# Load Docking Results
# ============================================================

def load_docking_results(
    docking_summary_file
):

    docking_summary_file = Path(
        docking_summary_file
    )

    with open(
        docking_summary_file,
        "r"
    ) as f:

        data = json.load(f)

    return pd.DataFrame(data)


# ============================================================
# Load Prediction Results
# ============================================================

def load_prediction_results(
    prediction_file
):

    prediction_file = Path(
        prediction_file
    )

    return pd.read_csv(
        prediction_file
    )


# ============================================================
# Merge Results
# ============================================================

def merge_results(
    prediction_df,
    docking_df
):
    """
    Expected common column:
    pair
    """

    merged = prediction_df.merge(

        docking_df,

        on="pair",

        how="inner"
    )

    return merged


# ============================================================
# Correlation Analysis
# ============================================================

def compute_correlations(
    predictions,
    affinities
):

    predictions = np.asarray(
        predictions
    )

    affinities = np.asarray(
        affinities
    )

    # More negative affinity = stronger binding

    affinities = -affinities

    report = {}

    try:

        report["pearson"] = float(

            pearsonr(

                predictions,

                affinities

            )[0]
        )

    except:

        report["pearson"] = 0.0

    try:

        report["spearman"] = float(

            spearmanr(

                predictions,

                affinities

            )[0]
        )

    except:

        report["spearman"] = 0.0

    try:

        report["kendall_tau"] = float(

            kendalltau(

                predictions,

                affinities

            )[0]
        )

    except:

        report["kendall_tau"] = 0.0

    return report


# ============================================================
# Error Metrics
# ============================================================

def compute_errors(
    predictions,
    affinities
):

    predictions = np.asarray(
        predictions
    )

    affinities = np.asarray(
        affinities
    )

    affinities = -affinities

    rmse = np.sqrt(

        mean_squared_error(

            affinities,

            predictions
        )
    )

    mae = mean_absolute_error(

        affinities,

        predictions
    )

    return {

        "rmse":
            float(rmse),

        "mae":
            float(mae)
    }


# ============================================================
# Top-K Hit Rate
# ============================================================

def top_k_hit_rate(
    prediction_df,
    docking_df,
    k=20
):

    pred_top = set(

        prediction_df

        .sort_values(
            "prediction",
            ascending=False
        )

        .head(k)

        ["pair"]
    )

    dock_top = set(

        docking_df

        .sort_values(
            "affinity",
            ascending=True
        )

        .head(k)

        ["pair"]
    )

    hits = len(
        pred_top.intersection(
            dock_top
        )
    )

    return float(
        hits / k
    )


# ============================================================
# Enrichment Factor
# ============================================================

def enrichment_factor(
    prediction_df,
    docking_df,
    k=20
):

    hit_rate = top_k_hit_rate(

        prediction_df,

        docking_df,

        k
    )

    random_rate = (

        k

        /

        len(docking_df)
    )

    return float(
        hit_rate
        /
        max(
            random_rate,
            1e-8
        )
    )


# ============================================================
# Statistical Significance
# ============================================================

def significance_test(
    predictions,
    affinities
):

    from scipy.stats import ttest_rel

    predictions = np.asarray(
        predictions
    )

    affinities = np.asarray(
        affinities
    )

    affinities = -affinities

    statistic, p_value = ttest_rel(

        predictions,

        affinities
    )

    return {

        "t_statistic":
            float(statistic),

        "p_value":
            float(p_value)
    }


# ============================================================
# Scatter Plot
# ============================================================

def generate_scatter_plot(
    predictions,
    affinities,
    output_file
):

    affinities = -np.asarray(
        affinities
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.scatter(
        predictions,
        affinities
    )

    plt.xlabel(
        "Predicted Score"
    )

    plt.ylabel(
        "Docking Affinity"
    )

    plt.title(
        "Prediction vs Docking"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()


# ============================================================
# Ranking Table
# ============================================================

def create_ranking_table(
    merged_df
):

    table = merged_df.copy()

    table["prediction_rank"] = (

        table["prediction"]

        .rank(
            ascending=False
        )
    )

    table["affinity_rank"] = (

        table["affinity"]

        .rank(
            ascending=True
        )
    )

    return table


# ============================================================
# Export Validation Results
# ============================================================

def export_validation_table(
    table,
    output_file
):

    table.to_csv(
        output_file,
        index=False
    )


# ============================================================
# Validation Pipeline
# ============================================================

def validate_docking_results(
    prediction_file,
    docking_summary_file,
    output_dir,
    top_k=20
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_df = load_prediction_results(
        prediction_file
    )

    docking_df = load_docking_results(
        docking_summary_file
    )

    merged = merge_results(

        prediction_df,

        docking_df
    )

    predictions = merged[
        "prediction"
    ].values

    affinities = merged[
        "affinity"
    ].values

    report = {}

    report.update(

        compute_correlations(

            predictions,

            affinities
        )
    )

    report.update(

        compute_errors(

            predictions,

            affinities
        )
    )

    report["top_k_hit_rate"] = (

        top_k_hit_rate(

            prediction_df,

            docking_df,

            top_k
        )
    )

    report["enrichment_factor"] = (

        enrichment_factor(

            prediction_df,

            docking_df,

            top_k
        )
    )

    report.update(

        significance_test(

            predictions,

            affinities
        )
    )

    generate_scatter_plot(

        predictions,

        affinities,

        output_dir
        /
        "scatter_plot.png"
    )

    ranking_table = (
        create_ranking_table(
            merged
        )
    )

    export_validation_table(

        ranking_table,

        output_dir
        /
        "validation_results.csv"
    )

    with open(

        output_dir
        /
        "validation_report.json",

        "w"
    ) as f:

        json.dump(

            report,

            f,

            indent=4
        )

    return report


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":

    print(
        "Docking Validation Module Ready."
    )