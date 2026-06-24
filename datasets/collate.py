"""
=============================================================
DTI Collate Functions

Causal Multi-Modal DTI Framework

Purpose
-------
Convert dataset samples into mini-batches
compatible with:

- PyTorch Geometric
- ProteinRepository
- Full Model

Input
-----
[
    {
        "graph": Data(...),
        "protein_id": 42,
        "label": tensor([1])
    },
    ...
]

Output
------
{
    "graph_batch": Batch,
    "protein_ids": Tensor,
    "labels": Tensor
}

Author:
Roshan Kotkondawar
Kunal Nicose
=============================================================
"""

import torch

from torch_geometric.data import Batch


# ============================================================
# Main Collate Function
# ============================================================

def dti_collate_fn(batch):
    """
    Standard DTI collate function.

    Parameters
    ----------
    batch : list

    Returns
    -------
    dict
    """

    graphs = []

    protein_ids = []

    labels = []

    for sample in batch:

        graphs.append(
            sample["graph"]
        )

        protein_ids.append(
            sample["protein_id"]
        )

        labels.append(
            sample["label"]
        )

    # --------------------------------------------------------
    # Graph Batch
    # --------------------------------------------------------

    graph_batch = Batch.from_data_list(
        graphs
    )

    # --------------------------------------------------------
    # Protein IDs
    # --------------------------------------------------------

    protein_ids = torch.tensor(
        protein_ids,
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = torch.stack(
        labels,
        dim=0
    )

    return {

        "graph_batch":
            graph_batch,

        "protein_ids":
            protein_ids,

        "labels":
            labels
    }


# ============================================================
# Device Transfer Utility
# ============================================================

def move_batch_to_device(
    batch,
    device
):
    """
    Move batch to GPU/CPU.
    """

    batch["graph_batch"] = (
        batch["graph_batch"].to(
            device
        )
    )

    batch["protein_ids"] = (
        batch["protein_ids"].to(
            device
        )
    )

    batch["labels"] = (
        batch["labels"].to(
            device
        )
    )

    return batch


# ============================================================
# Classification Batch Validation
# ============================================================

def validate_classification_batch(
    batch
):
    """
    Validate BindingDB batch.
    """

    labels = batch["labels"]

    if labels.ndim > 2:

        raise ValueError(
            "Labels must be [B,1]"
        )

    if torch.isnan(labels).any():

        raise ValueError(
            "NaN labels detected."
        )

    return True


# ============================================================
# Regression Batch Validation
# ============================================================

def validate_regression_batch(
    batch
):
    """
    Validate DAVIS/KIBA batch.
    """

    labels = batch["labels"]

    if torch.isnan(labels).any():

        raise ValueError(
            "NaN labels detected."
        )

    return True


# ============================================================
# Batch Statistics
# ============================================================

def batch_statistics(
    batch
):
    """
    Useful for debugging.
    """

    graph_batch = batch[
        "graph_batch"
    ]

    protein_ids = batch[
        "protein_ids"
    ]

    labels = batch[
        "labels"
    ]

    stats = {

        "batch_size":
            labels.shape[0],

        "graphs":
            graph_batch.num_graphs,

        "nodes":
            graph_batch.num_nodes,

        "edges":
            graph_batch.edge_index.shape[1],

        "unique_proteins":
            len(
                torch.unique(
                    protein_ids
                )
            )
    }

    return stats


# ============================================================
# Pretty Print Statistics
# ============================================================

def print_batch_statistics(
    batch
):
    """
    Debug helper.
    """

    stats = batch_statistics(
        batch
    )

    print("\n")
    print("=" * 60)

    print("BATCH STATISTICS")

    print("=" * 60)

    for key, value in stats.items():

        print(
            f"{key}: {value}"
        )

    print("=" * 60)


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":

    from torch_geometric.data import Data

    sample1 = {

        "graph":
            Data(
                x=torch.randn(5, 9),
                edge_index=torch.tensor(
                    [
                        [0, 1],
                        [1, 0]
                    ]
                )
            ),

        "protein_id":
            10,

        "label":
            torch.tensor(
                [1.0]
            )
    }

    sample2 = {

        "graph":
            Data(
                x=torch.randn(8, 9),
                edge_index=torch.tensor(
                    [
                        [0, 1],
                        [1, 0]
                    ]
                )
            ),

        "protein_id":
            15,

        "label":
            torch.tensor(
                [0.0]
            )
    }

    batch = dti_collate_fn(
        [
            sample1,
            sample2
        ]
    )

    print_batch_statistics(
        batch
    )

    print(
        "\nGraph Batch:",
        batch[
            "graph_batch"
        ]
    )

    print(
        "Protein IDs:",
        batch[
            "protein_ids"
        ]
    )

    print(
        "Labels:",
        batch[
            "labels"
        ]
    )