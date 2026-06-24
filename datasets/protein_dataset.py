"""
=============================================================
Protein Dataset Manager

Causal Multi-Modal DTI Framework

Purpose
-------
Store and manage protein sequences separately
from PyTorch Geometric graph objects.

Features
--------
1. Protein ID Mapping
2. Sequence Storage
3. ESM Embedding Cache
4. BindingDB Support
5. DAVIS Support
6. KIBA Support
7. Fast Retrieval

Author:
Roshan Kotkondawar
Kunal Nicose
=============================================================
"""

import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset


# ============================================================
# Protein Repository
# ============================================================

class ProteinRepository:
    """
    Central protein sequence storage.
    """

    def __init__(self):

        self.sequence_to_id = {}

        self.id_to_sequence = {}

        self.next_id = 0

    # =========================================================
    # Add Sequence
    # =========================================================

    def add_sequence(
        self,
        sequence
    ):

        sequence = str(sequence)

        if sequence in self.sequence_to_id:

            return self.sequence_to_id[
                sequence
            ]

        sequence_id = self.next_id

        self.sequence_to_id[
            sequence
        ] = sequence_id

        self.id_to_sequence[
            sequence_id
        ] = sequence

        self.next_id += 1

        return sequence_id

    # =========================================================
    # Get Sequence
    # =========================================================

    def get_sequence(
        self,
        sequence_id
    ):

        return self.id_to_sequence[
            int(sequence_id)
        ]

    # =========================================================
    # Length
    # =========================================================

    def __len__(self):

        return len(
            self.id_to_sequence
        )

    # =========================================================
    # Save
    # =========================================================

    def save(
        self,
        filepath
    ):

        data = {

            "sequence_to_id":
                self.sequence_to_id,

            "id_to_sequence":
                self.id_to_sequence,

            "next_id":
                self.next_id
        }

        with open(
            filepath,
            "wb"
        ) as f:

            pickle.dump(
                data,
                f
            )

    # =========================================================
    # Load
    # =========================================================

    @classmethod
    def load(
        cls,
        filepath
    ):

        with open(
            filepath,
            "rb"
        ) as f:

            data = pickle.load(f)

        repo = cls()

        repo.sequence_to_id = data[
            "sequence_to_id"
        ]

        repo.id_to_sequence = data[
            "id_to_sequence"
        ]

        repo.next_id = data[
            "next_id"
        ]

        return repo


# ============================================================
# Protein Dataset
# ============================================================

class ProteinDataset(Dataset):

    """
    Dataset of unique proteins.
    """

    def __init__(
        self,
        protein_repository
    ):

        self.repository = (
            protein_repository
        )

        self.ids = sorted(
            list(
                self.repository.id_to_sequence.keys()
            )
        )

    def __len__(self):

        return len(self.ids)

    def __getitem__(
        self,
        idx
    ):

        protein_id = self.ids[idx]

        sequence = (
            self.repository.get_sequence(
                protein_id
            )
        )

        return {

            "protein_id":
                protein_id,

            "sequence":
                sequence
        }


# ============================================================
# Protein Embedding Cache
# ============================================================

class ProteinEmbeddingCache:

    """
    Store precomputed ESM embeddings.

    Avoids repeated ESM forward passes.
    """

    def __init__(self):

        self.cache = {}

    # =========================================================
    # Add Embedding
    # =========================================================

    def add_embedding(
        self,
        protein_id,
        embedding
    ):

        self.cache[
            int(protein_id)
        ] = (
            embedding
            .detach()
            .cpu()
        )

    # =========================================================
    # Exists
    # =========================================================

    def exists(
        self,
        protein_id
    ):

        return (
            int(protein_id)
            in self.cache
        )

    # =========================================================
    # Retrieve
    # =========================================================

    def get_embedding(
        self,
        protein_id
    ):

        return self.cache[
            int(protein_id)
        ]

    # =========================================================
    # Save
    # =========================================================

    def save(
        self,
        filepath
    ):

        torch.save(
            self.cache,
            filepath
        )

    # =========================================================
    # Load
    # =========================================================

    @classmethod
    def load(
        cls,
        filepath
    ):

        obj = cls()

        obj.cache = torch.load(
            filepath,
            map_location="cpu"
        )

        return obj

    # =========================================================
    # Length
    # =========================================================

    def __len__(self):

        return len(
            self.cache
        )


# ============================================================
# Batch Sequence Retrieval
# ============================================================

def get_sequences_from_ids(
    protein_ids,
    repository
):
    """
    Convert batch of IDs
    into protein sequences.
    """

    sequences = []

    for pid in protein_ids:

        sequence = (
            repository.get_sequence(
                int(pid)
            )
        )

        sequences.append(
            sequence
        )

    return sequences


# ============================================================
# Batch Cached Embeddings
# ============================================================

def get_cached_embeddings(
    protein_ids,
    embedding_cache,
    device="cpu"
):

    embeddings = []

    for pid in protein_ids:

        embedding = (
            embedding_cache.get_embedding(
                int(pid)
            )
        )

        embeddings.append(
            embedding
        )

    embeddings = torch.stack(
        embeddings,
        dim=0
    )

    embeddings = embeddings.to(
        device
    )

    return embeddings


# ============================================================
# Save Repository Metadata
# ============================================================

def save_repository_metadata(
    repository,
    filepath
):

    metadata = {

        "num_unique_proteins":
            len(repository),

        "max_sequence_length":
            max(
                len(seq)
                for seq in
                repository.id_to_sequence.values()
            ),

        "min_sequence_length":
            min(
                len(seq)
                for seq in
                repository.id_to_sequence.values()
            )
    }

    with open(
        filepath,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":

    repo = ProteinRepository()

    id1 = repo.add_sequence(
        "MKWVTFISLLLLFSSAYSRGVFRR"
    )

    id2 = repo.add_sequence(
        "MKAILVVLLYTFATANAD"
    )

    print(
        "Protein IDs:",
        id1,
        id2
    )

    print(
        "Total Proteins:",
        len(repo)
    )

    dataset = ProteinDataset(
        repo
    )

    print(
        dataset[0]
    )

    cache = ProteinEmbeddingCache()

    emb = torch.randn(
        1280
    )

    cache.add_embedding(
        id1,
        emb
    )

    print(
        "Cache Size:",
        len(cache)
    )