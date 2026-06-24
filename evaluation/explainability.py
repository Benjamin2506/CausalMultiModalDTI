#!/usr/bin/env python3

"""
=============================================================
Explainability Module

Causal Multi-Modal DTI Framework

Supports
---------
Cross-Modal Attention
Integrated Gradients
Gradient x Input
Attention Rollout
Causal Analysis
Uncertainty-Aware Explanations

=============================================================
"""

from __future__ import annotations

import json
import numpy as np

from pathlib import Path

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

# ============================================================
# Utility
# ============================================================

def normalize_scores(scores):
    """
    Normalize attribution scores to [0,1].
    """

    scores = np.asarray(scores)

    scores = scores - scores.min()

    denom = scores.max() + 1e-12

    scores = scores / denom

    return scores


# ============================================================
# Attention Rollout
# ============================================================

def attention_rollout(
    attention_matrices
):
    """
    Compute transformer attention rollout.

    Input:
        list of attention tensors

    Shape:
        [layers, heads, tokens, tokens]
    """

    rollout = None

    for attn in attention_matrices:

        attn = attn.mean(axis=0)

        identity = np.eye(
            attn.shape[-1]
        )

        attn = attn + identity

        attn = attn / (
            attn.sum(
                axis=-1,
                keepdims=True
            )
            + 1e-12
        )

        if rollout is None:

            rollout = attn

        else:

            rollout = rollout @ attn

    return rollout


# ============================================================
# Gradient x Input
# ============================================================

def gradient_x_input(
    inputs,
    gradients
):

    return (
        inputs * gradients
    )


# ============================================================
# Integrated Gradients
# ============================================================

def integrated_gradients(
    model,
    inputs,
    baseline,
    target_index=None,
    steps=50
):
    """
    Generic integrated gradients.
    """

    scaled_inputs = []

    for alpha in np.linspace(
        0,
        1,
        steps
    ):

        scaled_inputs.append(

            baseline

            +

            alpha
            *
            (
                inputs
                -
                baseline
            )
        )

    total_gradients = torch.zeros_like(
        inputs
    )

    for scaled in scaled_inputs:

        scaled.requires_grad_(True)

        output = model(
            scaled
        )

        if isinstance(
            output,
            dict
        ):
            output = output[
                "prediction"
            ]

        if target_index is not None:

            output = output[
                :,
                target_index
            ]

        output.sum().backward()

        total_gradients += (
            scaled.grad
        )

    avg_gradients = (
        total_gradients
        /
        steps
    )

    integrated = (
        inputs
        -
        baseline
    ) * avg_gradients

    return integrated.detach()


# ============================================================
# Drug Atom Attribution
# ============================================================

def drug_atom_importance(
    atom_embeddings,
    atom_gradients
):
    """
    Atom-level importance.
    """

    scores = (
        atom_embeddings
        *
        atom_gradients
    )

    scores = scores.sum(
        axis=-1
    )

    scores = normalize_scores(
        scores
    )

    return scores


# ============================================================
# Protein Residue Attribution
# ============================================================

def protein_residue_importance(
    residue_embeddings,
    residue_gradients
):

    scores = (
        residue_embeddings
        *
        residue_gradients
    )

    scores = scores.sum(
        axis=-1
    )

    scores = normalize_scores(
        scores
    )

    return scores


# ============================================================
# Cross Modal Attention
# ============================================================

def extract_cross_modal_attention(
    attention_tensor
):
    """
    Shape:

    [heads,
     drug_tokens,
     protein_tokens]
    """

    attention = (
        attention_tensor.mean(
            axis=0
        )
    )

    return attention


# ============================================================
# Uncertainty Weighting
# ============================================================

def uncertainty_weighted_scores(
    importance_scores,
    uncertainty_scores
):
    """
    Penalize uncertain explanations.
    """

    uncertainty_scores = normalize_scores(
        uncertainty_scores
    )

    weighted = (
        importance_scores
        *
        (
            1.0
            -
            uncertainty_scores
        )
    )

    return normalize_scores(
        weighted
    )


# ============================================================
# Causal Effect Analysis
# ============================================================

def estimate_causal_effect(
    factual_prediction,
    counterfactual_prediction
):
    """
    Simple treatment effect.
    """

    return float(
        factual_prediction
        -
        counterfactual_prediction
    )


# ============================================================
# Attention Heatmap
# ============================================================

def plot_attention_heatmap(
    attention,
    save_path,
    title="Cross-Modal Attention"
):

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        attention,
        aspect="auto"
    )

    plt.colorbar()

    plt.title(title)

    plt.xlabel(
        "Protein Residues"
    )

    plt.ylabel(
        "Drug Atoms"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()


# ============================================================
# Attribution Plot
# ============================================================

def plot_attributions(
    scores,
    save_path,
    title="Attribution Scores"
):

    plt.figure(
        figsize=(10, 4)
    )

    plt.plot(scores)

    plt.title(title)

    plt.xlabel(
        "Position"
    )

    plt.ylabel(
        "Importance"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()


# ============================================================
# Save Attribution
# ============================================================

def save_attributions(
    scores,
    output_path
):

    scores = np.asarray(
        scores
    )

    np.savetxt(
        output_path,
        scores,
        delimiter=","
    )


# ============================================================
# Export Explainability Report
# ============================================================

def export_explainability_report(
    report,
    output_file
):

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )


# ============================================================
# Full Explainability Pipeline
# ============================================================

def run_explainability_pipeline(
    attention_matrix,
    atom_scores,
    residue_scores,
    output_dir
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_attention_heatmap(

        attention_matrix,

        output_dir
        /
        "attention_heatmap.png"
    )

    plot_attributions(

        atom_scores,

        output_dir
        /
        "drug_atom_importance.png",

        title=
        "Drug Atom Importance"
    )

    plot_attributions(

        residue_scores,

        output_dir
        /
        "protein_residue_importance.png",

        title=
        "Protein Residue Importance"
    )

    save_attributions(

        atom_scores,

        output_dir
        /
        "atom_scores.csv"
    )

    save_attributions(

        residue_scores,

        output_dir
        /
        "residue_scores.csv"
    )

    report = {

        "num_drug_atoms":
            int(
                len(atom_scores)
            ),

        "num_residues":
            int(
                len(residue_scores)
            ),

        "max_atom_score":
            float(
                np.max(
                    atom_scores
                )
            ),

        "max_residue_score":
            float(
                np.max(
                    residue_scores
                )
            )
    }

    export_explainability_report(

        report,

        output_dir
        /
        "explainability_report.json"
    )

    return report


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":

    attention = np.random.rand(
        30,
        200
    )

    atom_scores = np.random.rand(
        30
    )

    residue_scores = np.random.rand(
        200
    )

    report = run_explainability_pipeline(

        attention,

        atom_scores,

        residue_scores,

        "explainability_test"
    )

    print(
        json.dumps(
            report,
            indent=4
        )
    )