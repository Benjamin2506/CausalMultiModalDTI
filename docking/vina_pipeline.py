#!/usr/bin/env python3

"""
=============================================================
AutoDock Vina Pipeline

Causal Multi-Modal DTI Framework

Requirements
------------
AutoDock Vina
OpenBabel
MGLTools

Outputs
-------
docking/results/

=============================================================
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# ============================================================
# Paths
# ============================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DOCKING_DIR = (
    ROOT_DIR
    / "docking"
)

RESULTS_DIR = (
    DOCKING_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# Configuration
# ============================================================

VINA_EXECUTABLE = "vina"

OBABEL_EXECUTABLE = "obabel"

PREPARE_LIGAND = "prepare_ligand4.py"

PREPARE_RECEPTOR = "prepare_receptor4.py"

# ============================================================
# Utility
# ============================================================

def run_command(cmd):

    result = subprocess.run(

        cmd,

        shell=True,

        capture_output=True,

        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(

            f"\nCommand Failed:\n{cmd}\n\n"
            f"{result.stderr}"
        )

    return result.stdout


# ============================================================
# Ligand Preparation
# ============================================================

def prepare_ligand(
    ligand_file,
    output_dir
):

    ligand_file = Path(
        ligand_file
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pdbqt_file = (
        output_dir
        /
        f"{ligand_file.stem}.pdbqt"
    )

    cmd = (
        f"{OBABEL_EXECUTABLE} "
        f"{ligand_file} "
        f"-O {pdbqt_file}"
    )

    run_command(cmd)

    return pdbqt_file


# ============================================================
# Protein Preparation
# ============================================================

def prepare_protein(
    protein_file,
    output_dir
):

    protein_file = Path(
        protein_file
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pdbqt_file = (
        output_dir
        /
        f"{protein_file.stem}.pdbqt"
    )

    cmd = (

        f"python {PREPARE_RECEPTOR} "

        f"-r {protein_file} "

        f"-o {pdbqt_file}"
    )

    run_command(cmd)

    return pdbqt_file


# ============================================================
# Docking Configuration
# ============================================================

def generate_vina_config(
    receptor_pdbqt,
    ligand_pdbqt,
    center_x,
    center_y,
    center_z,
    size_x,
    size_y,
    size_z,
    output_file
):

    config = f"""
receptor = {receptor_pdbqt}
ligand = {ligand_pdbqt}

center_x = {center_x}
center_y = {center_y}
center_z = {center_z}

size_x = {size_x}
size_y = {size_y}
size_z = {size_z}

num_modes = 10
exhaustiveness = 16
"""

    with open(
        output_file,
        "w"
    ) as f:

        f.write(config)


# ============================================================
# Run Docking
# ============================================================

def run_vina(
    receptor_pdbqt,
    ligand_pdbqt,
    config_file,
    output_pose
):

    log_file = (
        Path(output_pose)
        .with_suffix(".log")
    )

    cmd = (

        f"{VINA_EXECUTABLE} "

        f"--config {config_file} "

        f"--out {output_pose} "

        f"--log {log_file}"
    )

    run_command(cmd)

    return log_file


# ============================================================
# Parse Affinity
# ============================================================

def parse_affinity(
    vina_log
):

    vina_log = Path(
        vina_log
    )

    affinity = None

    with open(
        vina_log,
        "r"
    ) as f:

        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 4:
            continue

        if parts[0].isdigit():

            try:

                affinity = float(
                    parts[1]
                )

                break

            except:

                pass

    return affinity


# ============================================================
# Single Docking Job
# ============================================================

def dock_single_pair(
    ligand_file,
    protein_file,
    docking_box
):

    pair_name = (

        f"{Path(ligand_file).stem}"
        "_"
        f"{Path(protein_file).stem}"
    )

    pair_dir = (
        RESULTS_DIR
        / pair_name
    )

    pair_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    ligand_pdbqt = prepare_ligand(
        ligand_file,
        pair_dir
    )

    receptor_pdbqt = prepare_protein(
        protein_file,
        pair_dir
    )

    config_file = (
        pair_dir
        / "vina_config.txt"
    )

    generate_vina_config(

        receptor_pdbqt,

        ligand_pdbqt,

        docking_box["center_x"],
        docking_box["center_y"],
        docking_box["center_z"],

        docking_box["size_x"],
        docking_box["size_y"],
        docking_box["size_z"],

        config_file
    )

    output_pose = (
        pair_dir
        / "docked_pose.pdbqt"
    )

    log_file = run_vina(

        receptor_pdbqt,

        ligand_pdbqt,

        config_file,

        output_pose
    )

    affinity = parse_affinity(
        log_file
    )

    result = {

        "pair":
            pair_name,

        "ligand":
            str(ligand_file),

        "protein":
            str(protein_file),

        "affinity":
            affinity,

        "pose":
            str(output_pose)
    }

    with open(
        pair_dir / "result.json",
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )

    return result


# ============================================================
# Batch Docking
# ============================================================

def batch_docking(
    ligand_files,
    protein_files,
    docking_box,
    num_workers=4
):

    jobs = []

    for ligand in ligand_files:

        for protein in protein_files:

            jobs.append(
                (
                    ligand,
                    protein,
                    docking_box
                )
            )

    results = []

    with ProcessPoolExecutor(
        max_workers=num_workers
    ) as executor:

        futures = [

            executor.submit(
                dock_single_pair,
                *job
            )

            for job in jobs
        ]

        for future in futures:

            results.append(
                future.result()
            )

    return results


# ============================================================
# Export Summary
# ============================================================

def export_summary(
    results,
    output_file
):

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )


# ============================================================
# Top-K Selection
# ============================================================

def select_top_k(
    prediction_csv,
    k=20
):
    """
    Select top predicted DTI pairs
    for docking validation.
    """

    import pandas as pd

    df = pd.read_csv(
        prediction_csv
    )

    df = df.sort_values(

        "prediction",

        ascending=False
    )

    return df.head(k)


# ============================================================
# Main Example
# ============================================================

if __name__ == "__main__":

    docking_box = {

        "center_x": 10.0,
        "center_y": 10.0,
        "center_z": 10.0,

        "size_x": 20.0,
        "size_y": 20.0,
        "size_z": 20.0
    }

    print(
        "AutoDock Vina Pipeline Ready."
    )