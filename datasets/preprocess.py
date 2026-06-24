#!/usr/bin/env python3

"""
=============================================================
Causal Multi-Modal DTI Framework

Unified Dataset Preprocessing Pipeline

Supported Datasets
------------------
1. BindingDB
2. DAVIS
3. KIBA

Features
--------
✓ ProteinRepository generation
✓ protein_id mapping
✓ PyTorch Geometric graphs
✓ Dataset statistics
✓ Train/Val/Test splits
✓ Metadata generation
✓ ESM cache compatibility

Outputs
-------
bindingdb.pt
davis.pt
kiba.pt

bindingdb_splits.pt
davis_splits.pt
kiba_splits.pt

protein_repository.pkl
protein_repository_metadata.json

Author:
Roshan Kotkondawar
Kunal Nicose
=============================================================
"""

# ============================================================
# Imports
# ============================================================

import os
import re
import json
import pickle
import random
import warnings

from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import (
    train_test_split
)

import torch

from torch_geometric.data import (
    Data
)

from rdkit import Chem

from rdkit.Chem import (
    rdchem
)

from datasets.protein_dataset import (
    ProteinRepository,
    save_repository_metadata
)

warnings.filterwarnings(
    "ignore"
)

# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

# ============================================================
# Paths
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

RAW_DIR = (
    ROOT_DIR
    / "datasets"
    / "raw"
)

PROCESSED_DIR = (
    ROOT_DIR
    / "datasets"
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# Protein Repository
# ============================================================

protein_repository = (
    ProteinRepository()
)

# ============================================================
# Constants
# ============================================================

VALID_AMINO_ACIDS = set(
    list(
        "ACDEFGHIKLMNPQRSTVWY"
    )
)

NODE_FEATURE_DIM = 9

EDGE_FEATURE_DIM = 7

# ============================================================
# Protein Cleaning
# ============================================================

def clean_protein_sequence(
    sequence
):
    """
    Remove invalid residues.
    """

    sequence = str(sequence)

    sequence = sequence.upper()

    cleaned = []

    for aa in sequence:

        if aa in VALID_AMINO_ACIDS:

            cleaned.append(aa)

    return "".join(cleaned)


# ============================================================
# Protein Registration
# ============================================================

def register_protein(
    sequence
):
    """
    Register sequence inside repository.

    Returns
    -------
    protein_id
    """

    sequence = clean_protein_sequence(
        sequence
    )

    if len(sequence) < 20:

        return None

    protein_id = (
        protein_repository
        .add_sequence(sequence)
    )

    return int(protein_id)


# ============================================================
# Atom Features
# ============================================================

def atom_features(
    atom
):
    """
    Atom feature vector.

    Output Dimension = 9
    """

    return [

        atom.GetAtomicNum(),

        atom.GetDegree(),

        atom.GetFormalCharge(),

        int(
            atom.GetHybridization()
        ),

        int(
            atom.GetIsAromatic()
        ),

        atom.GetTotalNumHs(),

        atom.GetImplicitValence(),

        atom.GetExplicitValence(),

        int(
            atom.IsInRing()
        )
    ]


# ============================================================
# Bond Features
# ============================================================

def bond_features(
    bond
):
    """
    Bond feature vector.

    Output Dimension = 7
    """

    return [

        float(
            bond.GetBondType()
            ==
            rdchem.BondType.SINGLE
        ),

        float(
            bond.GetBondType()
            ==
            rdchem.BondType.DOUBLE
        ),

        float(
            bond.GetBondType()
            ==
            rdchem.BondType.TRIPLE
        ),

        float(
            bond.GetBondType()
            ==
            rdchem.BondType.AROMATIC
        ),

        float(
            bond.GetIsConjugated()
        ),

        float(
            bond.IsInRing()
        ),

        float(
            bond.GetStereo()
            !=
            rdchem.BondStereo.STEREONONE
        )
    ]


# ============================================================
# SMILES Validation
# ============================================================

def validate_smiles(
    smiles
):
    """
    Validate molecule.
    """

    try:

        mol = Chem.MolFromSmiles(
            smiles
        )

        return mol is not None

    except Exception:

        return False
    
# ============================================================
# Molecular Graph Construction
# ============================================================

def smiles_to_graph(
    smiles
):
    """
    Convert SMILES into a
    PyTorch Geometric graph.

    Returns
    -------
    torch_geometric.data.Data
    """

    mol = Chem.MolFromSmiles(
        smiles
    )

    if mol is None:

        return None

    try:

        Chem.SanitizeMol(
            mol
        )

    except Exception:

        return None

    # --------------------------------------------------------
    # Node Features
    # --------------------------------------------------------

    node_features = []

    for atom in mol.GetAtoms():

        node_features.append(

            atom_features(
                atom
            )
        )

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    # --------------------------------------------------------
    # Edge Features
    # --------------------------------------------------------

    edge_index = []

    edge_attr = []

    for bond in mol.GetBonds():

        src = bond.GetBeginAtomIdx()

        dst = bond.GetEndAtomIdx()

        features = (
            bond_features(
                bond
            )
        )

        edge_index.append(
            [src, dst]
        )

        edge_index.append(
            [dst, src]
        )

        edge_attr.append(
            features
        )

        edge_attr.append(
            features
        )

    if len(edge_index) == 0:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

        edge_attr = torch.empty(
            (0, EDGE_FEATURE_DIM),
            dtype=torch.float
        )

    else:

        edge_index = torch.tensor(
            edge_index,
            dtype=torch.long
        ).t().contiguous()

        edge_attr = torch.tensor(
            edge_attr,
            dtype=torch.float
        )

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    return graph


# ============================================================
# Affinity Parsing
# ============================================================

def parse_affinity(
    value
):
    """
    Parse affinity values.

    Examples
    --------
    >100
    <10
    ~25
    50 nM

    Returns
    -------
    float or None
    """

    if pd.isna(value):

        return None

    value = str(value)

    value = (
        value
        .replace(">", "")
        .replace("<", "")
        .replace("~", "")
        .replace(",", "")
        .strip()
    )

    matches = re.findall(
        r"[-+]?\d*\.\d+|\d+",
        value
    )

    if len(matches) == 0:

        return None

    try:

        value = float(
            matches[0]
        )

        if value <= 0:

            return None

        return value

    except Exception:

        return None


# ============================================================
# Affinity Conversion
# ============================================================

def affinity_to_pKd(
    affinity_nm
):
    """
    Convert nM affinity
    into pKd.

    pKd = -log10(Kd[M])
    """

    try:

        affinity_nm = float(
            affinity_nm
        )

        if affinity_nm <= 0:

            return None

        kd_molar = (
            affinity_nm * 1e-9
        )

        return float(
            -np.log10(
                kd_molar
            )
        )

    except Exception:

        return None


# ============================================================
# Classification Label
# ============================================================

def affinity_to_label(
    affinity_nm,
    threshold_nm=1000
):
    """
    Binary interaction label.

    Positive:
        affinity <= threshold

    Negative:
        affinity > threshold
    """

    if affinity_nm is None:

        return None

    if affinity_nm <= threshold_nm:

        return 1

    return 0


# ============================================================
# Dataset Statistics
# ============================================================

def dataset_statistics(
    dataset
):
    """
    Compute dataset statistics.
    """

    stats = {}

    stats["num_samples"] = (
        len(dataset)
    )

    node_counts = []

    edge_counts = []

    protein_ids = set()

    labels = []

    for sample in dataset:

        node_counts.append(
            sample.x.shape[0]
        )

        edge_counts.append(
            sample.edge_index.shape[1]
        )

        if hasattr(
            sample,
            "protein_id"
        ):

            protein_ids.add(
                int(
                    sample.protein_id
                )
            )

        if hasattr(
            sample,
            "y"
        ):

            labels.append(
                float(
                    sample.y.item()
                )
            )

    stats["unique_proteins"] = (
        len(protein_ids)
    )

    stats["avg_nodes"] = float(
        np.mean(node_counts)
    )

    stats["avg_edges"] = float(
        np.mean(edge_counts)
    )

    if len(labels) > 0:

        stats["label_mean"] = (
            float(
                np.mean(labels)
            )
        )

        stats["label_std"] = (
            float(
                np.std(labels)
            )
        )

    return stats


# ============================================================
# Pretty Statistics
# ============================================================

def print_dataset_statistics(
    name,
    dataset
):
    """
    Print dataset summary.
    """

    stats = dataset_statistics(
        dataset
    )

    print("\n")
    print("=" * 70)

    print(
        f"{name} Statistics"
    )

    print("=" * 70)

    for key, value in stats.items():

        print(
            f"{key}: {value}"
        )

    print("=" * 70)


# ============================================================
# Metadata Saving
# ============================================================

def save_metadata(
    metadata,
    filename
):
    """
    Save metadata file.
    """

    filepath = (
        PROCESSED_DIR
        / filename
    )

    with open(
        filepath,
        "wb"
    ) as f:

        pickle.dump(
            metadata,
            f
        )

    print(
        f"[INFO] Saved: "
        f"{filepath}"
    )


# ============================================================
# Metadata Loading
# ============================================================

def load_metadata(
    filename
):
    """
    Load metadata.
    """

    filepath = (
        PROCESSED_DIR
        / filename
    )

    with open(
        filepath,
        "rb"
    ) as f:

        metadata = pickle.load(
            f
        )

    return metadata


# ============================================================
# Dataset Split
# ============================================================

def split_dataset(
    dataset,
    train_ratio=0.80,
    val_ratio=0.10,
    test_ratio=0.10
):
    """
    Train / Validation / Test
    """

    assert abs(
        (
            train_ratio
            + val_ratio
            + test_ratio
        )
        - 1.0
    ) < 1e-6

    train_data, temp_data = (
        train_test_split(
            dataset,
            test_size=(
                1.0
                - train_ratio
            ),
            random_state=SEED,
            shuffle=True
        )
    )

    val_fraction = (
        val_ratio
        /
        (
            val_ratio
            + test_ratio
        )
    )

    val_data, test_data = (
        train_test_split(
            temp_data,
            test_size=(
                1.0
                - val_fraction
            ),
            random_state=SEED,
            shuffle=True
        )
    )

    return (
        train_data,
        val_data,
        test_data
    )


# ============================================================
# Split Saving
# ============================================================

def save_dataset_splits(
    dataset,
    dataset_name
):
    """
    Save train/val/test split.
    """

    (
        train_data,
        val_data,
        test_data
    ) = split_dataset(
        dataset
    )

    split_dict = {

        "train":
            train_data,

        "val":
            val_data,

        "test":
            test_data
    }

    output_file = (
        PROCESSED_DIR
        /
        f"{dataset_name}_splits.pt"
    )

    torch.save(
        split_dict,
        output_file
    )

    print(
        f"[SAVED] "
        f"{output_file}"
    )

    print(
        f"Train: {len(train_data):,}"
    )

    print(
        f"Validation: {len(val_data):,}"
    )

    print(
        f"Test: {len(test_data):,}"
    )

    return split_dict


# ============================================================
# Dataset Serialization
# ============================================================

def save_dataset(
    dataset,
    filename
):
    """
    Save processed dataset.
    """

    filepath = (
        PROCESSED_DIR
        / filename
    )

    torch.save(
        dataset,
        filepath
    )

    print(
        f"[SAVED] "
        f"{filepath}"
    )


# ============================================================
# Dataset Verification
# ============================================================

def verify_dataset(
    dataset,
    dataset_name
):
    """
    Verify dataset integrity.
    """

    if len(dataset) == 0:

        raise RuntimeError(
            f"{dataset_name} "
            f"is empty."
        )

    sample = dataset[0]

    if not hasattr(
        sample,
        "protein_id"
    ):

        raise RuntimeError(
            "protein_id missing."
        )

    if sample.x.shape[1] != NODE_FEATURE_DIM:

        raise RuntimeError(
            "Node feature mismatch."
        )

    if sample.edge_attr.shape[1] != EDGE_FEATURE_DIM:

        raise RuntimeError(
            "Edge feature mismatch."
        )

    print(
        f"[VERIFIED] "
        f"{dataset_name}"
    )

# ============================================================
# BindingDB Column Detection
# ============================================================

def detect_bindingdb_columns(
    dataframe
):
    """
    Automatically detect required columns.

    Supports multiple BindingDB releases.
    """

    columns = {
        c.lower(): c
        for c in dataframe.columns
    }

    smiles_candidates = [
        "smiles",
        "ligand smiles",
        "canonical smiles",
        "drug_smiles"
    ]

    sequence_candidates = [
        "target sequence",
        "protein sequence",
        "sequence",
        "target_seq"
    ]

    affinity_candidates = [
        "ki (nm)",
        "ki",
        "kd (nm)",
        "kd",
        "ic50 (nm)",
        "ic50"
    ]

    detected = {}

    # --------------------------------------------------------
    # SMILES
    # --------------------------------------------------------

    for candidate in smiles_candidates:

        if candidate in columns:

            detected["smiles"] = (
                columns[candidate]
            )

            break

    # --------------------------------------------------------
    # Protein Sequence
    # --------------------------------------------------------

    for candidate in sequence_candidates:

        if candidate in columns:

            detected["sequence"] = (
                columns[candidate]
            )

            break

    # --------------------------------------------------------
    # Affinity
    # --------------------------------------------------------

    for candidate in affinity_candidates:

        if candidate in columns:

            detected["affinity"] = (
                columns[candidate]
            )

            break

    required = [
        "smiles",
        "sequence",
        "affinity"
    ]

    missing = [

        x

        for x in required

        if x not in detected
    ]

    if len(missing) > 0:

        raise RuntimeError(

            "BindingDB column detection failed.\n"

            f"Missing: {missing}"
        )

    return detected


# ============================================================
# BindingDB File Discovery
# ============================================================

def locate_bindingdb_file():
    """
    Locate BindingDB file inside raw folder.
    """

    candidates = [

        "BindingDB.tsv",

        "bindingdb.tsv",

        "BindingDB_All.tsv",

        "BindingDB.csv",

        "bindingdb.csv"
    ]

    for candidate in candidates:

        filepath = (
            RAW_DIR
            / candidate
        )

        if filepath.exists():

            return filepath

    raise FileNotFoundError(
        "\nBindingDB file not found.\n"
        "Expected one of:\n"
        f"{candidates}"
    )


# ============================================================
# BindingDB Row Processing
# ============================================================

def process_bindingdb_row(
    row,
    columns
):
    """
    Convert row into graph sample.
    """

    try:

        smiles = str(
            row[
                columns["smiles"]
            ]
        ).strip()

        sequence = str(
            row[
                columns["sequence"]
            ]
        ).strip()

        affinity_raw = (
            row[
                columns["affinity"]
            ]
        )

    except Exception:

        return None

    # --------------------------------------------------------
    # Validate SMILES
    # --------------------------------------------------------

    if not validate_smiles(
        smiles
    ):
        return None

    # --------------------------------------------------------
    # Protein Registration
    # --------------------------------------------------------

    protein_id = (
        register_protein(
            sequence
        )
    )

    if protein_id is None:

        return None

    # --------------------------------------------------------
    # Affinity
    # --------------------------------------------------------

    affinity_nm = (
        parse_affinity(
            affinity_raw
        )
    )

    if affinity_nm is None:

        return None

    label = (
        affinity_to_label(
            affinity_nm
        )
    )

    if label is None:

        return None

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    graph = smiles_to_graph(
        smiles
    )

    if graph is None:

        return None

    graph.protein_id = int(
        protein_id
    )

    graph.y = torch.tensor(
        [float(label)],
        dtype=torch.float
    )

    graph.affinity_nm = float(
        affinity_nm
    )

    return graph


# ============================================================
# BindingDB Metadata
# ============================================================

def build_bindingdb_metadata(
    dataset
):
    """
    Generate dataset metadata.
    """

    positives = 0

    negatives = 0

    affinities = []

    proteins = set()

    for sample in dataset:

        label = int(
            sample.y.item()
        )

        if label == 1:

            positives += 1

        else:

            negatives += 1

        proteins.add(
            int(
                sample.protein_id
            )
        )

        affinities.append(
            sample.affinity_nm
        )

    metadata = {

        "dataset":
            "BindingDB",

        "samples":
            len(dataset),

        "positive_samples":
            positives,

        "negative_samples":
            negatives,

        "positive_ratio":
            positives
            /
            max(
                1,
                len(dataset)
            ),

        "unique_proteins":
            len(proteins),

        "mean_affinity_nm":
            float(
                np.mean(
                    affinities
                )
            ),

        "std_affinity_nm":
            float(
                np.std(
                    affinities
                )
            )
    }

    return metadata


# ============================================================
# BindingDB Processing
# ============================================================

def process_bindingdb():
    """
    Main BindingDB processing pipeline.
    """

    print("\n")
    print("=" * 80)
    print("PROCESSING BINDINGDB")
    print("=" * 80)

    filepath = (
        locate_bindingdb_file()
    )

    print(
        f"Loading: {filepath}"
    )

    if filepath.suffix.lower() == ".csv":

        df = pd.read_csv(
            filepath,
            low_memory=False
        )

    else:

        df = pd.read_csv(
            filepath,
            sep="\t",
            low_memory=False
        )

    print(
        f"Rows: {len(df):,}"
    )

    columns = (
        detect_bindingdb_columns(
            df
        )
    )

    print(
        "\nDetected Columns:"
    )

    for k, v in columns.items():

        print(
            f"{k}: {v}"
        )

    dataset = []

    failures = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="BindingDB"
    ):

        sample = (
            process_bindingdb_row(
                row,
                columns
            )
        )

        if sample is None:

            failures += 1

            continue

        dataset.append(
            sample
        )

    print(
        f"\nValid Samples: "
        f"{len(dataset):,}"
    )

    print(
        f"Rejected Samples: "
        f"{failures:,}"
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_dataset(
        dataset,
        "BindingDB"
    )

    # --------------------------------------------------------
    # Save Dataset
    # --------------------------------------------------------

    save_dataset(
        dataset,
        "bindingdb.pt"
    )

    # --------------------------------------------------------
    # Save Splits
    # --------------------------------------------------------

    save_dataset_splits(
        dataset,
        "bindingdb"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_dataset_statistics(
        "BindingDB",
        dataset
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = (
        build_bindingdb_metadata(
            dataset
        )
    )

    save_metadata(
        metadata,
        "bindingdb_metadata.pkl"
    )

    print(
        "\nBindingDB Complete"
    )

    return dataset

# ============================================================
# DAVIS File Discovery
# ============================================================

def locate_davis_files():
    """
    Locate DAVIS dataset files.

    Expected Structure
    ------------------
    datasets/raw/davis/

        ligands_can.txt
        proteins.txt
        Y
    """

    davis_dir = (
        RAW_DIR
        / "davis"
    )

    if not davis_dir.exists():

        raise FileNotFoundError(
            f"DAVIS directory not found:\n"
            f"{davis_dir}"
        )

    ligands_file = (
        davis_dir
        / "ligands_can.txt"
    )

    proteins_file = (
        davis_dir
        / "proteins.txt"
    )

    affinity_file = (
        davis_dir
        / "Y"
    )

    required = [

        ligands_file,
        proteins_file,
        affinity_file
    ]

    for file in required:

        if not file.exists():

            raise FileNotFoundError(
                file
            )

    return (
        ligands_file,
        proteins_file,
        affinity_file
    )


# ============================================================
# DAVIS Loading
# ============================================================

def load_davis_raw():
    """
    Load DAVIS dataset.
    """

    (
        ligands_file,
        proteins_file,
        affinity_file
    ) = locate_davis_files()

    print(
        "\nLoading DAVIS files..."
    )

    with open(
        ligands_file,
        "r"
    ) as f:

        ligands = json.load(f)

    with open(
        proteins_file,
        "r"
    ) as f:

        proteins = json.load(f)

    affinity_matrix = np.loadtxt(
        affinity_file
    )

    print(
        f"Ligands: "
        f"{len(ligands):,}"
    )

    print(
        f"Proteins: "
        f"{len(proteins):,}"
    )

    print(
        f"Affinity Matrix Shape: "
        f"{affinity_matrix.shape}"
    )

    return (
        ligands,
        proteins,
        affinity_matrix
    )


# ============================================================
# DAVIS Affinity Conversion
# ============================================================

def davis_affinity_to_pkd(
    kd_value
):
    """
    Convert DAVIS Kd values
    to pKd.

    pKd = -log10(Kd[M])
    """

    try:

        kd_value = float(
            kd_value
        )

        if kd_value <= 0:

            return None

        return float(
            -np.log10(
                kd_value * 1e-9
            )
        )

    except Exception:

        return None


# ============================================================
# DAVIS Sample Creation
# ============================================================

def create_davis_sample(
    smiles,
    protein_sequence,
    affinity_value
):
    """
    Create regression sample.
    """

    # --------------------------------------------------------
    # Validate Molecule
    # --------------------------------------------------------

    if not validate_smiles(
        smiles
    ):
        return None

    graph = smiles_to_graph(
        smiles
    )

    if graph is None:

        return None

    # --------------------------------------------------------
    # Protein Registration
    # --------------------------------------------------------

    protein_id = (
        register_protein(
            protein_sequence
        )
    )

    if protein_id is None:

        return None

    # --------------------------------------------------------
    # Affinity
    # --------------------------------------------------------

    pkd = (
        davis_affinity_to_pkd(
            affinity_value
        )
    )

    if pkd is None:

        return None

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    graph.protein_id = int(
        protein_id
    )

    graph.y = torch.tensor(
        [pkd],
        dtype=torch.float
    )

    graph.raw_affinity = float(
        affinity_value
    )

    return graph


# ============================================================
# DAVIS Metadata
# ============================================================

def build_davis_metadata(
    dataset
):
    """
    DAVIS statistics.
    """

    proteins = set()

    affinities = []

    for sample in dataset:

        proteins.add(
            int(
                sample.protein_id
            )
        )

        affinities.append(
            float(
                sample.y.item()
            )
        )

    metadata = {

        "dataset":
            "DAVIS",

        "samples":
            len(dataset),

        "unique_proteins":
            len(proteins),

        "mean_pkd":
            float(
                np.mean(
                    affinities
                )
            ),

        "std_pkd":
            float(
                np.std(
                    affinities
                )
            ),

        "min_pkd":
            float(
                np.min(
                    affinities
                )
            ),

        "max_pkd":
            float(
                np.max(
                    affinities
                )
            )
    }

    return metadata


# ============================================================
# DAVIS Processing
# ============================================================

def process_davis():
    """
    Main DAVIS processing pipeline.
    """

    print("\n")
    print("=" * 80)
    print("PROCESSING DAVIS")
    print("=" * 80)

    (
        ligands,
        proteins,
        affinity_matrix
    ) = load_davis_raw()

    ligand_keys = list(
        ligands.keys()
    )

    protein_keys = list(
        proteins.keys()
    )

    dataset = []

    failures = 0

    for i in tqdm(
        range(
            len(ligand_keys)
        ),
        desc="DAVIS Ligands"
    ):

        smiles = ligands[
            ligand_keys[i]
        ]

        for j in range(
            len(protein_keys)
        ):

            affinity = (
                affinity_matrix[i][j]
            )

            if np.isnan(
                affinity
            ):
                continue

            protein_sequence = (
                proteins[
                    protein_keys[j]
                ]
            )

            sample = (
                create_davis_sample(
                    smiles,
                    protein_sequence,
                    affinity
                )
            )

            if sample is None:

                failures += 1

                continue

            dataset.append(
                sample
            )

    print(
        f"\nValid Samples: "
        f"{len(dataset):,}"
    )

    print(
        f"Rejected Samples: "
        f"{failures:,}"
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_dataset(
        dataset,
        "DAVIS"
    )

    # --------------------------------------------------------
    # Save Dataset
    # --------------------------------------------------------

    save_dataset(
        dataset,
        "davis.pt"
    )

    # --------------------------------------------------------
    # Save Splits
    # --------------------------------------------------------

    save_dataset_splits(
        dataset,
        "davis"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_dataset_statistics(
        "DAVIS",
        dataset
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = (
        build_davis_metadata(
            dataset
        )
    )

    save_metadata(
        metadata,
        "davis_metadata.pkl"
    )

    print(
        "\nDAVIS Complete"
    )

    return dataset

# ============================================================
# KIBA File Discovery
# ============================================================

def locate_kiba_files():
    """
    Locate KIBA dataset files.

    Expected Structure
    ------------------

    datasets/raw/kiba/

        ligands_can.txt
        proteins.txt
        Y
    """

    kiba_dir = (
        RAW_DIR
        / "kiba"
    )

    if not kiba_dir.exists():

        raise FileNotFoundError(
            f"KIBA directory not found:\n"
            f"{kiba_dir}"
        )

    ligands_file = (
        kiba_dir
        / "ligands_can.txt"
    )

    proteins_file = (
        kiba_dir
        / "proteins.txt"
    )

    affinity_file = (
        kiba_dir
        / "Y"
    )

    required = [

        ligands_file,
        proteins_file,
        affinity_file
    ]

    for file in required:

        if not file.exists():

            raise FileNotFoundError(
                file
            )

    return (
        ligands_file,
        proteins_file,
        affinity_file
    )


# ============================================================
# Load KIBA Raw Files
# ============================================================

def load_kiba_raw():
    """
    Load KIBA dataset.
    """

    (
        ligands_file,
        proteins_file,
        affinity_file
    ) = locate_kiba_files()

    print(
        "\nLoading KIBA files..."
    )

    with open(
        ligands_file,
        "r"
    ) as f:

        ligands = json.load(f)

    with open(
        proteins_file,
        "r"
    ) as f:

        proteins = json.load(f)

    affinity_matrix = np.loadtxt(
        affinity_file
    )

    print(
        f"Ligands: "
        f"{len(ligands):,}"
    )

    print(
        f"Proteins: "
        f"{len(proteins):,}"
    )

    print(
        f"Affinity Matrix Shape: "
        f"{affinity_matrix.shape}"
    )

    return (
        ligands,
        proteins,
        affinity_matrix
    )


# ============================================================
# KIBA Score Validation
# ============================================================

def validate_kiba_score(
    score
):
    """
    Validate KIBA score.
    """

    try:

        score = float(score)

        if np.isnan(score):

            return None

        if np.isinf(score):

            return None

        return score

    except Exception:

        return None


# ============================================================
# KIBA Sample Creation
# ============================================================

def create_kiba_sample(
    smiles,
    protein_sequence,
    kiba_score
):
    """
    Create KIBA regression sample.
    """

    # --------------------------------------------------------
    # Validate Molecule
    # --------------------------------------------------------

    if not validate_smiles(
        smiles
    ):
        return None

    graph = smiles_to_graph(
        smiles
    )

    if graph is None:

        return None

    # --------------------------------------------------------
    # Protein Registration
    # --------------------------------------------------------

    protein_id = (
        register_protein(
            protein_sequence
        )
    )

    if protein_id is None:

        return None

    # --------------------------------------------------------
    # KIBA Score
    # --------------------------------------------------------

    kiba_score = (
        validate_kiba_score(
            kiba_score
        )
    )

    if kiba_score is None:

        return None

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    graph.protein_id = int(
        protein_id
    )

    graph.y = torch.tensor(
        [float(kiba_score)],
        dtype=torch.float
    )

    graph.kiba_score = float(
        kiba_score
    )

    return graph


# ============================================================
# KIBA Metadata
# ============================================================

def build_kiba_metadata(
    dataset
):
    """
    Build KIBA metadata.
    """

    proteins = set()

    scores = []

    for sample in dataset:

        proteins.add(
            int(
                sample.protein_id
            )
        )

        scores.append(
            float(
                sample.y.item()
            )
        )

    metadata = {

        "dataset":
            "KIBA",

        "samples":
            len(dataset),

        "unique_proteins":
            len(proteins),

        "mean_kiba":
            float(
                np.mean(scores)
            ),

        "std_kiba":
            float(
                np.std(scores)
            ),

        "min_kiba":
            float(
                np.min(scores)
            ),

        "max_kiba":
            float(
                np.max(scores)
            )
    }

    return metadata


# ============================================================
# Process KIBA
# ============================================================

def process_kiba():
    """
    Main KIBA preprocessing pipeline.
    """

    print("\n")
    print("=" * 80)
    print("PROCESSING KIBA")
    print("=" * 80)

    (
        ligands,
        proteins,
        affinity_matrix
    ) = load_kiba_raw()

    ligand_keys = list(
        ligands.keys()
    )

    protein_keys = list(
        proteins.keys()
    )

    dataset = []

    failures = 0

    for i in tqdm(
        range(
            len(ligand_keys)
        ),
        desc="KIBA Ligands"
    ):

        smiles = ligands[
            ligand_keys[i]
        ]

        for j in range(
            len(protein_keys)
        ):

            score = (
                affinity_matrix[i][j]
            )

            if np.isnan(
                score
            ):
                continue

            protein_sequence = (
                proteins[
                    protein_keys[j]
                ]
            )

            sample = (
                create_kiba_sample(
                    smiles,
                    protein_sequence,
                    score
                )
            )

            if sample is None:

                failures += 1

                continue

            dataset.append(
                sample
            )

    print(
        f"\nValid Samples: "
        f"{len(dataset):,}"
    )

    print(
        f"Rejected Samples: "
        f"{failures:,}"
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_dataset(
        dataset,
        "KIBA"
    )

    # --------------------------------------------------------
    # Save Dataset
    # --------------------------------------------------------

    save_dataset(
        dataset,
        "kiba.pt"
    )

    # --------------------------------------------------------
    # Save Splits
    # --------------------------------------------------------

    save_dataset_splits(
        dataset,
        "kiba"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_dataset_statistics(
        "KIBA",
        dataset
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = (
        build_kiba_metadata(
            dataset
        )
    )

    save_metadata(
        metadata,
        "kiba_metadata.pkl"
    )

    print(
        "\nKIBA Complete"
    )

    return dataset

# ============================================================
# Protein Repository Saving
# ============================================================

def save_protein_repository():
    """
    Save ProteinRepository and metadata.
    """

    print("\n")
    print("=" * 80)
    print("SAVING PROTEIN REPOSITORY")
    print("=" * 80)

    repository_file = (
        PROCESSED_DIR
        / "protein_repository.pkl"
    )

    protein_repository.save(
        repository_file
    )

    metadata_file = (
        PROCESSED_DIR
        / "protein_repository_metadata.json"
    )

    save_repository_metadata(
        protein_repository,
        metadata_file
    )

    print(
        f"[SAVED] "
        f"{repository_file}"
    )

    print(
        f"[SAVED] "
        f"{metadata_file}"
    )

    print(
        f"Unique Proteins: "
        f"{len(protein_repository):,}"
    )


# ============================================================
# Repository Statistics
# ============================================================

def protein_repository_statistics():
    """
    Compute repository statistics.
    """

    lengths = []

    for sequence in (
        protein_repository
        .id_to_sequence
        .values()
    ):

        lengths.append(
            len(sequence)
        )

    if len(lengths) == 0:

        return {

            "num_proteins": 0,
            "min_length": 0,
            "max_length": 0,
            "mean_length": 0
        }

    return {

        "num_proteins":
            len(lengths),

        "min_length":
            int(
                np.min(lengths)
            ),

        "max_length":
            int(
                np.max(lengths)
            ),

        "mean_length":
            float(
                np.mean(lengths)
            )
    }


# ============================================================
# Dataset Integrity Check
# ============================================================

def validate_sample(
    sample
):
    """
    Validate individual sample.
    """

    required_attributes = [

        "x",
        "edge_index",
        "edge_attr",
        "protein_id",
        "y"
    ]

    for attribute in required_attributes:

        if not hasattr(
            sample,
            attribute
        ):

            raise RuntimeError(

                f"Missing attribute: "
                f"{attribute}"
            )

    if sample.x.shape[1] != NODE_FEATURE_DIM:

        raise RuntimeError(
            "Node feature mismatch."
        )

    if sample.edge_attr.shape[1] != EDGE_FEATURE_DIM:

        raise RuntimeError(
            "Edge feature mismatch."
        )

    return True


# ============================================================
# Dataset Validation
# ============================================================

def validate_dataset(
    dataset,
    dataset_name
):
    """
    Validate complete dataset.
    """

    print(
        f"\nValidating "
        f"{dataset_name}..."
    )

    valid = 0

    for sample in dataset:

        try:

            validate_sample(
                sample
            )

            valid += 1

        except Exception:

            pass

    print(
        f"Validated: "
        f"{valid:,}/"
        f"{len(dataset):,}"
    )

    if valid != len(dataset):

        raise RuntimeError(

            f"{dataset_name} "
            f"contains invalid samples."
        )

    print(
        f"[PASSED] "
        f"{dataset_name}"
    )


# ============================================================
# Cross Dataset Validation
# ============================================================

def validate_all_datasets(
    bindingdb=None,
    davis=None,
    kiba=None
):
    """
    Validate all datasets.
    """

    print("\n")
    print("=" * 80)
    print("GLOBAL DATASET VALIDATION")
    print("=" * 80)

    if bindingdb is not None:

        validate_dataset(
            bindingdb,
            "BindingDB"
        )

    if davis is not None:

        validate_dataset(
            davis,
            "DAVIS"
        )

    if kiba is not None:

        validate_dataset(
            kiba,
            "KIBA"
        )

    print(
        "\nAll datasets validated."
    )


# ============================================================
# Dataset Inventory
# ============================================================

def build_dataset_inventory():
    """
    Create inventory of processed files.
    """

    inventory = {}

    for file in sorted(
        PROCESSED_DIR.glob("*")
    ):

        if file.is_file():

            inventory[
                file.name
            ] = {

                "size_mb":
                    round(
                        file.stat().st_size
                        /
                        (1024 * 1024),
                        3
                    )
            }

    return inventory


# ============================================================
# Master Metadata
# ============================================================

def generate_master_metadata(
    bindingdb=None,
    davis=None,
    kiba=None
):
    """
    Generate master metadata file.
    """

    print("\n")
    print("=" * 80)
    print("GENERATING MASTER METADATA")
    print("=" * 80)

    metadata = {

        "repository":

            protein_repository_statistics(),

        "datasets": {}
    }

    if bindingdb is not None:

        metadata[
            "datasets"
        ][
            "bindingdb"
        ] = dataset_statistics(
            bindingdb
        )

    if davis is not None:

        metadata[
            "datasets"
        ][
            "davis"
        ] = dataset_statistics(
            davis
        )

    if kiba is not None:

        metadata[
            "datasets"
        ][
            "kiba"
        ] = dataset_statistics(
            kiba
        )

    metadata[
        "inventory"
    ] = build_dataset_inventory()

    save_metadata(
        metadata,
        "master_metadata.pkl"
    )

    print(
        "[SAVED] "
        "master_metadata.pkl"
    )

    return metadata


# ============================================================
# Processing Report
# ============================================================

def print_processing_report(
    bindingdb=None,
    davis=None,
    kiba=None
):
    """
    Final processing report.
    """

    print("\n")
    print("=" * 80)
    print("FINAL PROCESSING REPORT")
    print("=" * 80)

    repo_stats = (
        protein_repository_statistics()
    )

    print("\nProtein Repository")

    for key, value in repo_stats.items():

        print(
            f"{key}: {value}"
        )

    if bindingdb is not None:

        print(
            f"\nBindingDB Samples: "
            f"{len(bindingdb):,}"
        )

    if davis is not None:

        print(
            f"DAVIS Samples: "
            f"{len(davis):,}"
        )

    if kiba is not None:

        print(
            f"KIBA Samples: "
            f"{len(kiba):,}"
        )

    inventory = (
        build_dataset_inventory()
    )

    print(
        "\nGenerated Files:"
    )

    for filename in inventory:

        print(
            f" - {filename}"
        )

    print("=" * 80)


# ============================================================
# Processing Summary JSON
# ============================================================

def save_processing_summary():
    """
    Save lightweight JSON summary.
    """

    summary = {

        "repository":
            protein_repository_statistics(),

        "generated_at":
            str(
                pd.Timestamp.now()
            ),

        "processed_dir":
            str(
                PROCESSED_DIR
            )
    }

    output_file = (
        PROCESSED_DIR
        / "processing_summary.json"
    )

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print(
        f"[SAVED] "
        f"{output_file}"
    )

# ============================================================
# Dataset Selection
# ============================================================

AVAILABLE_DATASETS = {

    "bindingdb":
        process_bindingdb,

    "davis":
        process_davis,

    "kiba":
        process_kiba
}


# ============================================================
# Dataset Runner
# ============================================================

def run_dataset(
    dataset_name
):
    """
    Run single dataset processing.
    """

    dataset_name = (
        dataset_name
        .lower()
        .strip()
    )

    if dataset_name not in AVAILABLE_DATASETS:

        raise ValueError(

            f"Unknown dataset: "
            f"{dataset_name}"
        )

    print("\n")
    print("=" * 80)

    print(
        f"RUNNING {dataset_name.upper()}"
    )

    print("=" * 80)

    dataset = (
        AVAILABLE_DATASETS[
            dataset_name
        ]()
    )

    return dataset


# ============================================================
# Run Multiple Datasets
# ============================================================

def run_multiple_datasets(
    dataset_names
):
    """
    Process multiple datasets.
    """

    results = {}

    for dataset_name in dataset_names:

        dataset_name = (
            dataset_name
            .lower()
            .strip()
        )

        if dataset_name not in AVAILABLE_DATASETS:

            print(
                f"Skipping unknown dataset: "
                f"{dataset_name}"
            )

            continue

        dataset = (
            run_dataset(
                dataset_name
            )
        )

        results[
            dataset_name
        ] = dataset

    return results


# ============================================================
# Process All Datasets
# ============================================================

def process_all_datasets():
    """
    Process BindingDB, DAVIS and KIBA.
    """

    print("\n")
    print("=" * 80)

    print(
        "PROCESSING ALL DATASETS"
    )

    print("=" * 80)

    results = {}

    for dataset_name in [

        "bindingdb",

        "davis",

        "kiba"
    ]:

        try:

            results[
                dataset_name
            ] = run_dataset(
                dataset_name
            )

        except Exception as e:

            print(
                f"\nFailed: "
                f"{dataset_name}"
            )

            print(
                str(e)
            )

    return results


# ============================================================
# Validation Workflow
# ============================================================

def run_validation_workflow(
    results
):
    """
    Validate generated datasets.
    """

    bindingdb = results.get(
        "bindingdb"
    )

    davis = results.get(
        "davis"
    )

    kiba = results.get(
        "kiba"
    )

    validate_all_datasets(

        bindingdb,

        davis,

        kiba
    )


# ============================================================
# Repository Workflow
# ============================================================

def run_repository_workflow():
    """
    Save protein repository.
    """

    save_protein_repository()


# ============================================================
# Metadata Workflow
# ============================================================

def run_metadata_workflow(
    results
):
    """
    Generate metadata files.
    """

    bindingdb = results.get(
        "bindingdb"
    )

    davis = results.get(
        "davis"
    )

    kiba = results.get(
        "kiba"
    )

    generate_master_metadata(

        bindingdb,

        davis,

        kiba
    )

    save_processing_summary()


# ============================================================
# Reporting Workflow
# ============================================================

def run_reporting_workflow(
    results
):
    """
    Final reporting.
    """

    print_processing_report(

        results.get(
            "bindingdb"
        ),

        results.get(
            "davis"
        ),

        results.get(
            "kiba"
        )
    )


# ============================================================
# Full Workflow
# ============================================================

def run_full_workflow(
    selected_datasets
):
    """
    Complete preprocessing workflow.
    """

    print("\n")
    print("=" * 80)

    print(
        "STARTING PREPROCESSING"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Dataset Processing
    # --------------------------------------------------------

    if (
        len(selected_datasets) == 1
        and
        selected_datasets[0]
        == "all"
    ):

        results = (
            process_all_datasets()
        )

    else:

        results = (
            run_multiple_datasets(
                selected_datasets
            )
        )

    # --------------------------------------------------------
    # Repository
    # --------------------------------------------------------

    run_repository_workflow()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    run_validation_workflow(
        results
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    run_metadata_workflow(
        results
    )

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------

    run_reporting_workflow(
        results
    )

    print("\n")
    print("=" * 80)

    print(
        "PREPROCESSING COMPLETED"
    )

    print("=" * 80)

    return results


# ============================================================
# CLI Argument Parser
# ============================================================

def build_arg_parser():
    """
    Command line arguments.
    """

    import argparse

    parser = argparse.ArgumentParser(

        description=
        (
            "Causal Multi-Modal DTI "
            "Dataset Preprocessing"
        )
    )

    parser.add_argument(

        "--datasets",

        nargs="+",

        default=["all"],

        choices=[
            "all",
            "bindingdb",
            "davis",
            "kiba"
        ],

        help=
        (
            "Datasets to process"
        )
    )

    return parser


# ============================================================
# Configuration Display
# ============================================================

def print_configuration(
    args
):
    """
    Display configuration.
    """

    print("\n")
    print("=" * 80)

    print(
        "PREPROCESSING CONFIGURATION"
    )

    print("=" * 80)

    print(
        "Raw Directory:"
    )

    print(
        RAW_DIR
    )

    print(
        "\nProcessed Directory:"
    )

    print(
        PROCESSED_DIR
    )

    print(
        "\nSelected Datasets:"
    )

    for dataset in args.datasets:

        print(
            f" - {dataset}"
        )

    print("=" * 80)

# ============================================================
# Environment Validation
# ============================================================

def validate_environment():
    """
    Validate preprocessing environment.
    """

    print("\n")
    print("=" * 80)
    print("ENVIRONMENT VALIDATION")
    print("=" * 80)

    print(
        f"Python: {os.sys.version.split()[0]}"
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"NumPy: {np.__version__}"
    )

    print(
        f"Pandas: {pd.__version__}"
    )

    print(
        f"Raw Directory: {RAW_DIR}"
    )

    print(
        f"Processed Directory: {PROCESSED_DIR}"
    )

    print("=" * 80)


# ============================================================
# Dataset Availability Check
# ============================================================

def dataset_exists(
    dataset_name
):
    """
    Check if dataset files exist.
    """

    dataset_name = (
        dataset_name
        .lower()
        .strip()
    )

    try:

        if dataset_name == "bindingdb":

            locate_bindingdb_file()

            return True

        elif dataset_name == "davis":

            locate_davis_files()

            return True

        elif dataset_name == "kiba":

            locate_kiba_files()

            return True

        return False

    except Exception:

        return False


# ============================================================
# Validate Requested Datasets
# ============================================================

def validate_requested_datasets(
    datasets
):
    """
    Verify datasets exist before processing.
    """

    print("\n")
    print("=" * 80)

    print(
        "DATASET AVAILABILITY CHECK"
    )

    print("=" * 80)

    if (
        len(datasets) == 1
        and
        datasets[0] == "all"
    ):

        datasets = [

            "bindingdb",

            "davis",

            "kiba"
        ]

    available = []

    missing = []

    for dataset_name in datasets:

        if dataset_exists(
            dataset_name
        ):

            available.append(
                dataset_name
            )

            print(
                f"[FOUND] "
                f"{dataset_name}"
            )

        else:

            missing.append(
                dataset_name
            )

            print(
                f"[MISSING] "
                f"{dataset_name}"
            )

    if len(available) == 0:

        raise RuntimeError(
            "No valid datasets found."
        )

    print("=" * 80)

    return available


# ============================================================
# Runtime Summary
# ============================================================

def print_runtime_summary(
    elapsed_seconds
):
    """
    Display runtime summary.
    """

    hours = int(
        elapsed_seconds // 3600
    )

    minutes = int(
        (
            elapsed_seconds
            % 3600
        )
        // 60
    )

    seconds = int(
        elapsed_seconds % 60
    )

    print("\n")
    print("=" * 80)

    print(
        "RUNTIME SUMMARY"
    )

    print("=" * 80)

    print(
        f"Elapsed Time: "
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )

    print("=" * 80)


# ============================================================
# Completion Banner
# ============================================================

def print_completion_banner():
    """
    Final success banner.
    """

    print("\n")

    print("=" * 80)
    print("PREPROCESSING SUCCESSFUL")
    print("=" * 80)

    print(
        "\nGenerated Files:\n"
    )

    generated = sorted(
        PROCESSED_DIR.glob("*")
    )

    for file in generated:

        if file.is_file():

            print(
                f"✓ {file.name}"
            )

    print("\n")

    print(
        "Next Steps:"
    )

    print(
        "1. Generate ESM embeddings"
    )

    print(
        "   python scripts/precompute_esm_embeddings.py"
    )

    print()

    print(
        "2. Train BindingDB model"
    )

    print(
        "   python training/train_bindingdb.py"
    )

    print()

    print(
        "3. Train DAVIS model"
    )

    print(
        "   python training/train_davis.py"
    )

    print()

    print(
        "4. Train KIBA model"
    )

    print(
        "   python training/train_kiba.py"
    )

    print("\n")
    print("=" * 80)


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """
    Production preprocessing entry point.
    """

    import time

    start_time = time.time()

    try:

        # ----------------------------------------------------
        # Environment
        # ----------------------------------------------------

        validate_environment()

        # ----------------------------------------------------
        # Arguments
        # ----------------------------------------------------

        parser = (
            build_arg_parser()
        )

        args = (
            parser.parse_args()
        )

        print_configuration(
            args
        )

        # ----------------------------------------------------
        # Dataset Validation
        # ----------------------------------------------------

        selected_datasets = (
            validate_requested_datasets(
                args.datasets
            )
        )

        # ----------------------------------------------------
        # Processing
        # ----------------------------------------------------

        run_full_workflow(
            selected_datasets
        )

        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------

        elapsed = (
            time.time()
            - start_time
        )

        print_runtime_summary(
            elapsed
        )

        # ----------------------------------------------------
        # Success Banner
        # ----------------------------------------------------

        print_completion_banner()

    except KeyboardInterrupt:

        print("\n")
        print(
            "Processing interrupted "
            "by user."
        )

    except Exception as e:

        print("\n")
        print("=" * 80)

        print(
            "PREPROCESSING FAILED"
        )

        print("=" * 80)

        print(
            str(e)
        )

        raise


# ============================================================
# Script Execution
# ============================================================

if __name__ == "__main__":

    main()