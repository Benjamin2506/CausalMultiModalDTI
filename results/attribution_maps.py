#!/usr/bin/env python3

"""
=============================================================
Attribution Maps Visualization

Causal Multi-Modal DTI Framework

Outputs
--------
drug_attribution.png
protein_attribution.png
combined_attribution.png
drug_attributions.csv
protein_attributions.csv
attribution_summary.json

=============================================================
"""

from __future__ import annotations

import json
import argparse
import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt

# ============================================================
# Load Attribution Scores
# ============================================================

def load_scores(score_file):
    """
    Supported:
        .npy
        .csv
    """

    score_file = Path(score_file)

    if score_file.suffix == ".npy":

        scores = np.load(score_file)

    elif score_file.suffix == ".csv":

        scores = pd.read_csv(
            score_file,
            header=None
        ).values.squeeze()

    else:

        raise ValueError(
            f"Unsupported file format: {score_file}"
        )

    return np.asarray(scores).flatten()


# ============================================================
# Normalize
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(scores)

    scores = scores.astype(
        np.float32
    )

    scores = np.abs(scores)

    scores -= scores.min()

    scores /= (
        scores.max()
        + 1e-12
    )

    return scores


# ============================================================
# Export CSV
# ============================================================

def export_scores(scores, output_file):

    pd.DataFrame({

        "importance": scores

    }).to_csv(
        output_file,
        index=False
    )


# ============================================================
# Statistics
# ============================================================

def compute_statistics(scores):

    scores = np.asarray(scores)

    return {

        "count":
            int(len(scores)),

        "mean":
            float(np.mean(scores)),

        "std":
            float(np.std(scores)),

        "min":
            float(np.min(scores)),

        "max":
            float(np.max(scores)),

        "median":
            float(np.median(scores)),

        "top_1_index":
            int(np.argmax(scores)),

        "top_1_score":
            float(np.max(scores))
    }


# ============================================================
# Save Summary
# ============================================================

def save_summary(summary, output_file):

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )


# ============================================================
# Drug Attribution Plot
# ============================================================

def plot_drug_attributions(
    scores,
    output_file
):

    plt.figure(
        figsize=(12, 4)
    )

    plt.bar(
        np.arange(len(scores)),
        scores
    )

    plt.xlabel(
        "Drug Atom Index",
        fontsize=14
    )

    plt.ylabel(
        "Importance",
        fontsize=14
    )

    plt.title(
        "Drug Atom Attribution",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Protein Attribution Plot
# ============================================================

def plot_protein_attributions(
    scores,
    output_file
):

    plt.figure(
        figsize=(14, 4)
    )

    plt.plot(
        scores,
        linewidth=2
    )

    plt.xlabel(
        "Protein Residue Index",
        fontsize=14
    )

    plt.ylabel(
        "Importance",
        fontsize=14
    )

    plt.title(
        "Protein Residue Attribution",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Combined Attribution Plot
# ============================================================

def plot_combined_attributions(
    drug_scores,
    protein_scores,
    output_file
):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8)
    )

    axes[0].bar(
        np.arange(len(drug_scores)),
        drug_scores
    )

    axes[0].set_title(
        "Drug Atom Attribution"
    )

    axes[0].set_ylabel(
        "Importance"
    )

    axes[1].plot(
        protein_scores,
        linewidth=2
    )

    axes[1].set_title(
        "Protein Residue Attribution"
    )

    axes[1].set_xlabel(
        "Residue Index"
    )

    axes[1].set_ylabel(
        "Importance"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Heatmap Attribution
# ============================================================

def plot_attribution_heatmap(
    matrix,
    output_file,
    title="Cross-Modal Attribution"
):

    plt.figure(
        figsize=(12, 8)
    )

    plt.imshow(
        matrix,
        aspect="auto"
    )

    plt.colorbar()

    plt.title(
        title,
        fontsize=16
    )

    plt.xlabel(
        "Protein Residues"
    )

    plt.ylabel(
        "Drug Atoms"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Cross-Modal Attribution Matrix
# ============================================================

def build_cross_modal_matrix(
    drug_scores,
    protein_scores
):
    """
    Outer product attribution matrix.
    """

    return np.outer(
        drug_scores,
        protein_scores
    )


# ============================================================
# Uncertainty Weighting
# ============================================================

def apply_uncertainty_weighting(
    attributions,
    uncertainty
):

    uncertainty = np.asarray(
        uncertainty
    )

    uncertainty = normalize_scores(
        uncertainty
    )

    return (
        attributions
        *
        (
            1.0
            -
            uncertainty
        )
    )


# ============================================================
# Full Pipeline
# ============================================================

def generate_attribution_figures(
    drug_file,
    protein_file,
    output_dir
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    drug_scores = normalize_scores(
        load_scores(
            drug_file
        )
    )

    protein_scores = normalize_scores(
        load_scores(
            protein_file
        )
    )

    # ---------------------------------------
    # Figures
    # ---------------------------------------

    plot_drug_attributions(

        drug_scores,

        output_dir
        /
        "drug_attribution.png"
    )

    plot_protein_attributions(

        protein_scores,

        output_dir
        /
        "protein_attribution.png"
    )

    plot_combined_attributions(

        drug_scores,

        protein_scores,

        output_dir
        /
        "combined_attribution.png"
    )

    # ---------------------------------------
    # Cross Modal Attribution
    # ---------------------------------------

    matrix = build_cross_modal_matrix(

        drug_scores,

        protein_scores
    )

    plot_attribution_heatmap(

        matrix,

        output_dir
        /
        "cross_modal_attribution.png"
    )

    # ---------------------------------------
    # Export Scores
    # ---------------------------------------

    export_scores(

        drug_scores,

        output_dir
        /
        "drug_attributions.csv"
    )

    export_scores(

        protein_scores,

        output_dir
        /
        "protein_attributions.csv"
    )

    # ---------------------------------------
    # Summary
    # ---------------------------------------

    summary = {

        "drug": compute_statistics(
            drug_scores
        ),

        "protein": compute_statistics(
            protein_scores
        )
    }

    save_summary(

        summary,

        output_dir
        /
        "attribution_summary.json"
    )

    print(
        f"Attribution figures saved to {output_dir}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--drug_scores",
        required=True
    )

    parser.add_argument(
        "--protein_scores",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/attributions"
    )

    args = parser.parse_args()

    generate_attribution_figures(

        args.drug_scores,

        args.protein_scores,

        args.output_dir
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()