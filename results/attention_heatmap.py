#!/usr/bin/env python3

"""
=============================================================
Attention Heatmap Visualization

Causal Multi-Modal DTI Framework

Outputs
--------
attention_heatmap.png
head_attention_grid.png
attention_rollout.png
attention_matrix.csv

=============================================================
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt

# ============================================================
# Load Attention
# ============================================================

def load_attention_matrix(
    attention_file
):
    """
    Supported:

    .npy
    .csv

    Expected Shape:

    [heads, drug_tokens, protein_tokens]

    or

    [drug_tokens, protein_tokens]
    """

    attention_file = Path(
        attention_file
    )

    if attention_file.suffix == ".npy":

        attention = np.load(
            attention_file
        )

    elif attention_file.suffix == ".csv":

        attention = pd.read_csv(
            attention_file,
            header=None
        ).values

    else:

        raise ValueError(
            f"Unsupported format: {attention_file}"
        )

    return attention


# ============================================================
# Normalize
# ============================================================

def normalize_attention(
    attention
):

    attention = attention.astype(
        np.float32
    )

    attention -= attention.min()

    attention /= (
        attention.max()
        + 1e-12
    )

    return attention


# ============================================================
# Average Heads
# ============================================================

def average_heads(
    attention
):

    if attention.ndim == 2:

        return attention

    return attention.mean(
        axis=0
    )


# ============================================================
# Attention Rollout
# ============================================================

def attention_rollout(
    attentions
):
    """
    Input:

    [layers, heads, tokens, tokens]
    """

    rollout = None

    for layer_attention in attentions:

        attn = layer_attention.mean(
            axis=0
        )

        identity = np.eye(
            attn.shape[-1]
        )

        attn = (
            attn + identity
        )

        attn = (
            attn
            /
            attn.sum(
                axis=-1,
                keepdims=True
            )
        )

        if rollout is None:

            rollout = attn

        else:

            rollout = rollout @ attn

    return rollout


# ============================================================
# Export Matrix
# ============================================================

def export_matrix(
    matrix,
    output_file
):

    pd.DataFrame(
        matrix
    ).to_csv(
        output_file,
        index=False
    )


# ============================================================
# Single Heatmap
# ============================================================

def plot_attention_heatmap(
    attention,
    output_file,
    title="Cross-Modal Attention"
):

    plt.figure(
        figsize=(12, 8)
    )

    plt.imshow(
        attention,
        aspect="auto"
    )

    plt.colorbar()

    plt.title(
        title,
        fontsize=16
    )

    plt.xlabel(
        "Protein Residues",
        fontsize=14
    )

    plt.ylabel(
        "Drug Atoms",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Multi-Head Visualization
# ============================================================

def plot_attention_heads(
    attention,
    output_file
):

    if attention.ndim != 3:

        raise ValueError(
            "Expected [heads,tokens,tokens]"
        )

    heads = attention.shape[0]

    cols = min(
        4,
        heads
    )

    rows = int(
        np.ceil(
            heads / cols
        )
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, 4 * rows)
    )

    axes = np.array(
        axes
    ).reshape(-1)

    for idx in range(heads):

        ax = axes[idx]

        im = ax.imshow(
            attention[idx],
            aspect="auto"
        )

        ax.set_title(
            f"Head {idx+1}"
        )

    for idx in range(
        heads,
        len(axes)
    ):
        axes[idx].axis("off")

    fig.colorbar(
        im,
        ax=axes.tolist()
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Attention Statistics
# ============================================================

def compute_attention_statistics(
    attention
):

    return {

        "mean":
            float(
                np.mean(attention)
            ),

        "std":
            float(
                np.std(attention)
            ),

        "max":
            float(
                np.max(attention)
            ),

        "min":
            float(
                np.min(attention)
            ),

        "sparsity":
            float(
                np.mean(
                    attention < 0.01
                )
            )
    }


# ============================================================
# Save Statistics
# ============================================================

def save_statistics(
    stats,
    output_file
):

    import json

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            stats,
            f,
            indent=4
        )


# ============================================================
# Full Pipeline
# ============================================================

def generate_attention_figures(
    attention_file,
    output_dir
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    attention = load_attention_matrix(
        attention_file
    )

    attention = normalize_attention(
        attention
    )

    stats = compute_attention_statistics(
        attention
    )

    save_statistics(

        stats,

        output_dir
        /
        "attention_statistics.json"
    )

    # -----------------------------------------
    # Mean Attention
    # -----------------------------------------

    mean_attention = average_heads(
        attention
    )

    plot_attention_heatmap(

        mean_attention,

        output_dir
        /
        "attention_heatmap.png"
    )

    export_matrix(

        mean_attention,

        output_dir
        /
        "attention_matrix.csv"
    )

    # -----------------------------------------
    # Multi-head
    # -----------------------------------------

    if attention.ndim == 3:

        plot_attention_heads(

            attention,

            output_dir
            /
            "head_attention_grid.png"
        )

    print(
        f"Saved figures to: {output_dir}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--attention",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/attention"
    )

    args = parser.parse_args()

    generate_attention_figures(

        args.attention,

        args.output_dir
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()