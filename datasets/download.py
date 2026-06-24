#!/usr/bin/env python3

"""
=============================================================
Dataset Downloader

Causal Multi-Modal DTI Framework

Downloads
----------
1. BindingDB
2. DAVIS
3. KIBA

Output
-------
datasets/raw/

Author:
Roshan Kotkondawar
Kunal Nicose
=============================================================
"""

import os
import json
import hashlib
import zipfile
import tarfile
import shutil

from pathlib import Path

import requests
from tqdm import tqdm

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

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# Dataset URLs
# ============================================================

DATASETS = {

    "bindingdb": {

        "url":
        "https://bindingdb.org/rwd/bind/downloads/BindingDB_All.tsv.zip",

        "filename":
        "BindingDB_All.tsv.zip"
    },

    "davis": {

        "url":
        "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/folds/train_fold_setting1.txt",

        "note":
        "Manual DAVIS download required."
    },

    "kiba": {

        "url":
        "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/kiba/folds/train_fold_setting1.txt",

        "note":
        "Manual KIBA download required."
    }
}

# ============================================================
# Utility
# ============================================================

def md5sum(
    filepath
):
    """
    Compute MD5 hash.
    """

    md5 = hashlib.md5()

    with open(
        filepath,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                8192
            )

            if not chunk:
                break

            md5.update(
                chunk
            )

    return md5.hexdigest()


# ============================================================
# Download File
# ============================================================

def download_file(
    url,
    output_file
):
    """
    Robust downloader.
    """

    response = requests.get(
        url,
        stream=True
    )

    response.raise_for_status()

    total_size = int(
        response.headers.get(
            "content-length",
            0
        )
    )

    progress = tqdm(

        total=total_size,

        unit="B",

        unit_scale=True,

        desc=output_file.name
    )

    with open(
        output_file,
        "wb"
    ) as f:

        for chunk in response.iter_content(
            chunk_size=8192
        ):

            if chunk:

                f.write(
                    chunk
                )

                progress.update(
                    len(chunk)
                )

    progress.close()


# ============================================================
# Extraction
# ============================================================

def extract_archive(
    archive_path,
    output_dir
):
    """
    Extract zip/tar archives.
    """

    archive_path = Path(
        archive_path
    )

    output_dir = Path(
        output_dir
    )

    if archive_path.suffix == ".zip":

        with zipfile.ZipFile(
            archive_path,
            "r"
        ) as z:

            z.extractall(
                output_dir
            )

    elif archive_path.suffix in [

        ".gz",
        ".tgz"
    ]:

        with tarfile.open(
            archive_path,
            "r:*"
        ) as tar:

            tar.extractall(
                output_dir
            )


# ============================================================
# BindingDB
# ============================================================

def download_bindingdb():
    """
    Download BindingDB.
    """

    print("\n")
    print("=" * 80)
    print("DOWNLOADING BINDINGDB")
    print("=" * 80)

    info = DATASETS[
        "bindingdb"
    ]

    output_file = (
        RAW_DIR
        / info["filename"]
    )

    if output_file.exists():

        print(
            "BindingDB already exists."
        )

    else:

        download_file(

            info["url"],

            output_file
        )

    print(
        "\nExtracting..."
    )

    extract_archive(

        output_file,

        RAW_DIR
    )

    print(
        "BindingDB ready."
    )


# ============================================================
# DAVIS
# ============================================================

def setup_davis():
    """
    Create DAVIS directory.
    """

    print("\n")
    print("=" * 80)
    print("DAVIS SETUP")
    print("=" * 80)

    davis_dir = (
        RAW_DIR
        / "davis"
    )

    davis_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nManual download required."
    )

    print(
        "Place files:"
    )

    print(
        "  ligands_can.txt"
    )

    print(
        "  proteins.txt"
    )

    print(
        "  Y"
    )

    print(
        f"\nDirectory:\n{davis_dir}"
    )


# ============================================================
# KIBA
# ============================================================

def setup_kiba():
    """
    Create KIBA directory.
    """

    print("\n")
    print("=" * 80)
    print("KIBA SETUP")
    print("=" * 80)

    kiba_dir = (
        RAW_DIR
        / "kiba"
    )

    kiba_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nManual download required."
    )

    print(
        "Place files:"
    )

    print(
        "  ligands_can.txt"
    )

    print(
        "  proteins.txt"
    )

    print(
        "  Y"
    )

    print(
        f"\nDirectory:\n{kiba_dir}"
    )


# ============================================================
# Verification
# ============================================================

def verify_structure():
    """
    Verify raw dataset structure.
    """

    print("\n")
    print("=" * 80)
    print("VERIFYING DATASETS")
    print("=" * 80)

    checks = {

        "BindingDB":

        (
            RAW_DIR
            /
            "BindingDB_All.tsv"
        ),

        "DAVIS":

        (
            RAW_DIR
            /
            "davis"
        ),

        "KIBA":

        (
            RAW_DIR
            /
            "kiba"
        )
    }

    for name, path in checks.items():

        if path.exists():

            print(
                f"[OK] {name}"
            )

        else:

            print(
                f"[MISSING] {name}"
            )


# ============================================================
# Report
# ============================================================

def generate_report():
    """
    Save download report.
    """

    report = {}

    for file in RAW_DIR.rglob("*"):

        if file.is_file():

            report[
                str(
                    file.relative_to(
                        RAW_DIR
                    )
                )
            ] = {

                "size_mb":

                round(
                    file.stat().st_size
                    /
                    (1024 * 1024),
                    3
                )
            }

    report_file = (
        RAW_DIR
        / "download_report.json"
    )

    with open(
        report_file,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print(
        f"\nSaved:\n{report_file}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("DTI DATASET DOWNLOADER")
    print("=" * 80)

    download_bindingdb()

    setup_davis()

    setup_kiba()

    verify_structure()

    generate_report()

    print("\n")
    print("=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)

    print(
        "\nNext Step:\n"
        "python datasets/preprocess.py"
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()