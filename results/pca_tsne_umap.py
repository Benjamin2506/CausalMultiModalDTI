#!/usr/bin/env python3

"""
=============================================================
PCA / t-SNE / UMAP Visualization

Causal Multi-Modal DTI Framework

Outputs
--------
pca.png
tsne.png
umap.png

=============================================================
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import umap

# ============================================================
# Load Embeddings
# ============================================================

def load_embeddings(
    embedding_file
):
    """
    Supported formats:

    .npy
    .csv
    """

    embedding_file = Path(
        embedding_file
    )

    if embedding_file.suffix == ".npy":

        embeddings = np.load(
            embedding_file
        )

        labels = None

    elif embedding_file.suffix == ".csv":

        df = pd.read_csv(
            embedding_file
        )

        if "label" in df.columns:

            labels = df["label"].values

            embeddings = df.drop(
                columns=["label"]
            ).values

        else:

            labels = None

            embeddings = df.values

    else:

        raise ValueError(
            f"Unsupported format: {embedding_file}"
        )

    return embeddings, labels


# ============================================================
# Standardization
# ============================================================

def normalize_embeddings(
    embeddings
):

    scaler = StandardScaler()

    return scaler.fit_transform(
        embeddings
    )


# ============================================================
# PCA
# ============================================================

def compute_pca(
    embeddings,
    n_components=2
):

    pca = PCA(
        n_components=n_components,
        random_state=42
    )

    transformed = pca.fit_transform(
        embeddings
    )

    variance = (
        pca.explained_variance_ratio_
    )

    return transformed, variance


# ============================================================
# t-SNE
# ============================================================

def compute_tsne(
    embeddings,
    perplexity=30
):

    tsne = TSNE(

        n_components=2,

        perplexity=perplexity,

        learning_rate="auto",

        init="pca",

        random_state=42
    )

    return tsne.fit_transform(
        embeddings
    )


# ============================================================
# UMAP
# ============================================================

def compute_umap(
    embeddings,
    n_neighbors=15,
    min_dist=0.1
):

    reducer = umap.UMAP(

        n_components=2,

        n_neighbors=n_neighbors,

        min_dist=min_dist,

        metric="euclidean",

        random_state=42
    )

    return reducer.fit_transform(
        embeddings
    )


# ============================================================
# Plot Helper
# ============================================================

def create_scatter_plot(
    coordinates,
    labels,
    title,
    output_file
):

    plt.figure(
        figsize=(10, 8)
    )

    if labels is None:

        plt.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=12
        )

    else:

        scatter = plt.scatter(

            coordinates[:, 0],

            coordinates[:, 1],

            c=labels,

            s=12
        )

        plt.colorbar(
            scatter
        )

    plt.title(
        title,
        fontsize=14
    )

    plt.xlabel(
        "Component 1"
    )

    plt.ylabel(
        "Component 2"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PCA Figure
# ============================================================

def generate_pca_figure(
    embeddings,
    labels,
    output_dir
):

    coords, variance = compute_pca(
        embeddings
    )

    output_file = (
        output_dir
        / "pca.png"
    )

    title = (
        f"PCA "
        f"(Explained Variance: "
        f"{variance[0]:.2f}, "
        f"{variance[1]:.2f})"
    )

    create_scatter_plot(

        coords,

        labels,

        title,

        output_file
    )

    return output_file


# ============================================================
# t-SNE Figure
# ============================================================

def generate_tsne_figure(
    embeddings,
    labels,
    output_dir
):

    coords = compute_tsne(
        embeddings
    )

    output_file = (
        output_dir
        / "tsne.png"
    )

    create_scatter_plot(

        coords,

        labels,

        "t-SNE Projection",

        output_file
    )

    return output_file


# ============================================================
# UMAP Figure
# ============================================================

def generate_umap_figure(
    embeddings,
    labels,
    output_dir
):

    coords = compute_umap(
        embeddings
    )

    output_file = (
        output_dir
        / "umap.png"
    )

    create_scatter_plot(

        coords,

        labels,

        "UMAP Projection",

        output_file
    )

    return output_file


# ============================================================
# Export Coordinates
# ============================================================

def export_coordinates(
    coordinates,
    output_file
):

    df = pd.DataFrame({

        "x": coordinates[:, 0],

        "y": coordinates[:, 1]
    })

    df.to_csv(
        output_file,
        index=False
    )


# ============================================================
# Full Pipeline
# ============================================================

def run_visualization_pipeline(
    embedding_file,
    output_dir
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    embeddings, labels = (
        load_embeddings(
            embedding_file
        )
    )

    embeddings = normalize_embeddings(
        embeddings
    )

    # PCA
    pca_coords, _ = compute_pca(
        embeddings
    )

    generate_pca_figure(
        embeddings,
        labels,
        output_dir
    )

    export_coordinates(

        pca_coords,

        output_dir
        /
        "pca_coordinates.csv"
    )

    # TSNE
    tsne_coords = compute_tsne(
        embeddings
    )

    generate_tsne_figure(
        embeddings,
        labels,
        output_dir
    )

    export_coordinates(

        tsne_coords,

        output_dir
        /
        "tsne_coordinates.csv"
    )

    # UMAP
    umap_coords = compute_umap(
        embeddings
    )

    generate_umap_figure(
        embeddings,
        labels,
        output_dir
    )

    export_coordinates(

        umap_coords,

        output_dir
        /
        "umap_coordinates.csv"
    )

    print(
        f"Figures saved to: {output_dir}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/figures"
    )

    args = parser.parse_args()

    run_visualization_pipeline(

        args.embeddings,

        args.output_dir
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()