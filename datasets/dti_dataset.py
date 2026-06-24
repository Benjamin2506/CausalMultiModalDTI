"""
=============================================================
DTI Dataset

Causal Multi-Modal DTI Framework

Supports:
----------
✓ BindingDB
✓ DAVIS
✓ KIBA

Outputs:
---------
graph
protein_id
label

Compatible With:
----------------
PyTorch Geometric
ProteinRepository
Full Model
Training Scripts

Author:
Roshan Kotkondawar
Kunal Nicose
=============================================================
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data


# ============================================================
# DTI Dataset
# ============================================================

class DTIDataset(Dataset):

    def __init__(
        self,
        dataset_file
    ):
        """
        Parameters
        ----------
        dataset_file : str
            .pt file path
        """

        self.dataset_file = Path(
            dataset_file
        )

        if not self.dataset_file.exists():

            raise FileNotFoundError(
                f"Dataset not found:\n"
                f"{self.dataset_file}"
            )

        self.samples = torch.load(
            self.dataset_file,
            map_location="cpu"
        )

        print(
            f"[INFO] Loaded "
            f"{len(self.samples)} samples"
        )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):

        sample = self.samples[idx]

        graph = sample

        # ----------------------------------------
        # Protein ID
        # ----------------------------------------

        protein_id = getattr(
            sample,
            "protein_id",
            None
        )

        if protein_id is None:

            raise RuntimeError(
                "protein_id missing.\n"
                "Re-run preprocess.py "
                "using ProteinRepository."
            )

        # ----------------------------------------
        # Label
        # ----------------------------------------

        label = sample.y

        return {

            "graph":
                graph,

            "protein_id":
                int(protein_id),

            "label":
                label
        }


# ============================================================
# Split Dataset Wrapper
# ============================================================

class DTISplitDataset(Dataset):

    """
    Wrapper for train/val/test splits.
    """

    def __init__(
        self,
        split_file,
        split="train"
    ):

        split_file = Path(
            split_file
        )

        if not split_file.exists():

            raise FileNotFoundError(
                split_file
            )

        data = torch.load(
            split_file,
            map_location="cpu"
        )

        if split not in data:

            raise ValueError(
                f"Split {split} "
                f"not found."
            )

        self.samples = data[
            split
        ]

        self.split = split

        print(
            f"[INFO] Loaded "
            f"{split} split "
            f"({len(self.samples)} samples)"
        )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):

        sample = self.samples[idx]

        protein_id = getattr(
            sample,
            "protein_id",
            None
        )

        if protein_id is None:

            raise RuntimeError(
                "protein_id missing."
            )

        return {

            "graph":
                sample,

            "protein_id":
                int(protein_id),

            "label":
                sample.y
        }


# ============================================================
# Dataset Statistics
# ============================================================

def dataset_statistics(
    dataset
):

    labels = []

    protein_ids = set()

    num_nodes = []

    num_edges = []

    for sample in dataset:

        graph = sample["graph"]

        labels.append(
            sample["label"]
        )

        protein_ids.add(
            sample["protein_id"]
        )

        num_nodes.append(
            graph.x.shape[0]
        )

        num_edges.append(
            graph.edge_index.shape[1]
        )

    stats = {

        "samples":
            len(dataset),

        "unique_proteins":
            len(protein_ids),

        "avg_nodes":
            sum(num_nodes)
            /
            len(num_nodes),

        "avg_edges":
            sum(num_edges)
            /
            len(num_edges)
    }

    return stats


# ============================================================
# BindingDB Loader
# ============================================================

def load_bindingdb_split(
    processed_dir,
    split="train"
):

    split_file = (
        Path(processed_dir)
        /
        "bindingdb_splits.pt"
    )

    return DTISplitDataset(
        split_file,
        split
    )


# ============================================================
# DAVIS Loader
# ============================================================

def load_davis_split(
    processed_dir,
    split="train"
):

    split_file = (
        Path(processed_dir)
        /
        "davis_splits.pt"
    )

    return DTISplitDataset(
        split_file,
        split
    )


# ============================================================
# KIBA Loader
# ============================================================

def load_kiba_split(
    processed_dir,
    split="train"
):

    split_file = (
        Path(processed_dir)
        /
        "kiba_splits.pt"
    )

    return DTISplitDataset(
        split_file,
        split
    )


# ============================================================
# Utility
# ============================================================

def print_dataset_summary(
    dataset,
    name="Dataset"
):

    stats = dataset_statistics(
        dataset
    )

    print("\n")
    print("=" * 60)

    print(name)

    print("=" * 60)

    for k, v in stats.items():

        print(
            f"{k}: {v}"
        )

    print("=" * 60)


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":

    print(
        "\nDTI Dataset Module Loaded"
    )

    print(
        "Ready for training."
    )