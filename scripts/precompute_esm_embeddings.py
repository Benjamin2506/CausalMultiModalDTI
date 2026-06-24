"""
=============================================================
Precompute ESM Protein Embeddings

Causal Multi-Modal DTI Framework

Input
-----
datasets/processed/protein_repository.pkl

Output
------
datasets/processed/esm_embeddings.pt

Supported Models
----------------
esm2_t6_8M_UR50D
esm2_t12_35M_UR50D
esm2_t30_150M_UR50D
esm2_t33_650M_UR50D

Default:
esm2_t33_650M_UR50D

Author:
Roshan Kotkondawar
Kunal Nicose
=============================================================
"""

import argparse
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import esm


# ============================================================
# Protein Dataset
# ============================================================

class ProteinEmbeddingDataset(Dataset):

    def __init__(self, repository):

        self.repository = repository

        self.ids = sorted(
            repository.id_to_sequence.keys()
        )

    def __len__(self):

        return len(self.ids)

    def __getitem__(self, idx):

        protein_id = self.ids[idx]

        sequence = (
            self.repository.id_to_sequence[
                protein_id
            ]
        )

        return protein_id, sequence


# ============================================================
# Collate
# ============================================================

def collate_fn(batch):

    protein_ids = []
    sequences = []

    for pid, seq in batch:

        protein_ids.append(pid)
        sequences.append(seq)

    return protein_ids, sequences


# ============================================================
# ESM Loader
# ============================================================

def load_esm_model(model_name):

    if model_name == "esm2_t6_8M_UR50D":

        model, alphabet = (
            esm.pretrained.esm2_t6_8M_UR50D()
        )

    elif model_name == "esm2_t12_35M_UR50D":

        model, alphabet = (
            esm.pretrained.esm2_t12_35M_UR50D()
        )

    elif model_name == "esm2_t30_150M_UR50D":

        model, alphabet = (
            esm.pretrained.esm2_t30_150M_UR50D()
        )

    elif model_name == "esm2_t33_650M_UR50D":

        model, alphabet = (
            esm.pretrained.esm2_t33_650M_UR50D()
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model, alphabet


# ============================================================
# Mean Pooling
# ============================================================

def mean_pool_embeddings(
    representations,
    tokens,
    padding_idx
):

    pooled = []

    for i in range(
        representations.shape[0]
    ):

        valid = (
            tokens[i]
            != padding_idx
        )

        emb = representations[i][valid]

        pooled.append(
            emb.mean(dim=0)
        )

    return torch.stack(
        pooled,
        dim=0
    )


# ============================================================
# Main
# ============================================================

def main(args):

    repo_file = Path(
        args.repository
    )

    output_file = Path(
        args.output
    )

    if not repo_file.exists():

        raise FileNotFoundError(
            repo_file
        )

    print(
        "\nLoading Protein Repository..."
    )

    with open(
        repo_file,
        "rb"
    ) as f:

        repository = pickle.load(f)

    print(
        f"Proteins: "
        f"{len(repository.id_to_sequence):,}"
    )

    dataset = ProteinEmbeddingDataset(
        repository
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    print(
        f"\nLoading ESM Model: "
        f"{args.model_name}"
    )

    model, alphabet = load_esm_model(
        args.model_name
    )

    batch_converter = (
        alphabet.get_batch_converter()
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    model.eval()

    if (
        torch.cuda.device_count() > 1
        and device == "cuda"
    ):

        print(
            f"Using "
            f"{torch.cuda.device_count()} GPUs"
        )

        model = torch.nn.DataParallel(
            model
        )

    embeddings = {}

    print(
        "\nComputing Embeddings..."
    )

    with torch.no_grad():

        for protein_ids, sequences in tqdm(
            loader
        ):

            batch = []

            for i, seq in enumerate(
                sequences
            ):

                seq = str(seq)[:1024]

                batch.append(
                    (
                        str(
                            protein_ids[i]
                        ),
                        seq
                    )
                )

            _, _, tokens = (
                batch_converter(batch)
            )

            tokens = tokens.to(device)

            with torch.cuda.amp.autocast(
                enabled=(device == "cuda")
            ):

                results = model(
                    tokens,
                    repr_layers=[33],
                    return_contacts=False
                )

            representations = (
                results[
                    "representations"
                ][33]
            )

            pooled = (
                mean_pool_embeddings(
                    representations,
                    tokens,
                    alphabet.padding_idx
                )
            )

            pooled = pooled.cpu()

            for pid, emb in zip(
                protein_ids,
                pooled
            ):

                embeddings[
                    int(pid)
                ] = emb

    print(
        f"\nSaving embeddings:"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        embeddings,
        output_file
    )

    print(
        f"{output_file}"
    )

    print(
        f"\nEmbeddings Saved:"
    )

    print(
        f"{len(embeddings):,}"
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repository",
        type=str,
        default=(
            "datasets/processed/"
            "protein_repository.pkl"
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "datasets/processed/"
            "esm_embeddings.pt"
        )
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="esm2_t33_650M_UR50D"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )

    args = parser.parse_args()

    main(args)