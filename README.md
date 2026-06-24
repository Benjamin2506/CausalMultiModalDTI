# Causal Multi-Modal Transformer for Drug–Target Interaction Prediction System

## Overview

This repository implements a **Causal Multi-Modal Transformer Framework with Uncertainty-Aware Explainability for Mechanistic Drug–Target Interaction (DTI) Prediction**.

The framework combines:

* Graph Transformer-based molecular representation learning
* ESM-2 protein language embeddings
* Cross-modal transformer attention fusion
* Causal intervention-based explainability
* Counterfactual reasoning
* Monte Carlo Dropout uncertainty estimation
* Drug–target interaction classification
* Binding affinity prediction
* Molecular docking validation

The goal is to provide a reliable, interpretable, and biologically meaningful DTI prediction framework for computational drug discovery and precision medicine applications.

---

# Features

## Molecular Representation Learning

* SMILES to molecular graph conversion using RDKit
* Atom and bond feature extraction
* Graph Transformer encoder
* Edge-aware attention mechanisms

## Protein Representation Learning

* ESM-2 pretrained protein language model
* Contextual amino acid embeddings
* Protein sequence encoding

## Multi-Modal Fusion

* Cross-attention transformer fusion
* Drug-to-protein interaction modeling
* Residual fusion architecture

## Explainability

* Structural attribution maps
* Counterfactual perturbation analysis
* Feature intervention studies
* Attention visualization

## Uncertainty Quantification

* Monte Carlo Dropout
* Epistemic uncertainty estimation
* Predictive calibration
* Reliability analysis

## Validation

* Classification performance evaluation
* Affinity prediction evaluation
* External dataset validation
* Molecular docking validation using AutoDock Vina

---

# Datasets

The framework supports multiple benchmark DTI datasets.

## BindingDB

Large-scale experimentally validated drug–target interactions.

Download:

https://www.bindingdb.org

---

## DAVIS

Kinase binding affinity benchmark.

Download:

https://github.com/hkmztrk/DeepDTA

---

## KIBA

Kinase inhibitor bioactivity benchmark.

Download:

https://github.com/hkmztrk/DeepDTA

---

## DrugBank

Drug-target interaction validation dataset.

Download:

https://go.drugbank.com

License required.

---

## PDBBind

Protein–ligand structural complexes.

Download:

http://www.pdbbind.org.cn

Registration required.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/causal-multimodal-dti.git

cd causal-multimodal-dti
```

---

## Create Environment

### Using Conda

```bash
conda env create -f environment.yml

conda activate causal-dti
```

---

### Using Pip

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

Create dataset directories:

```bash
mkdir -p datasets/raw
mkdir -p datasets/processed
```

Download datasets:

```bash
python datasets/download.py
```

Preprocess datasets:

```bash
python datasets/preprocess.py
```

Generated outputs:

```text
datasets/
│
├── raw/
│
└── processed/
    ├── bindingdb.pt
    ├── davis.pt
    ├── kiba.pt
    └── metadata.pkl
```

---

# Training

## BindingDB Classification

```bash
python training/train_bindingdb.py \
--config configs/bindingdb.yaml
```

---

## DAVIS Affinity Prediction

```bash
python training/train_davis.py \
--config configs/davis.yaml
```

---

## KIBA Affinity Prediction

```bash
python training/train_kiba.py \
--config configs/kiba.yaml
```

---

# Model Architecture

```text
Drug SMILES
      │
      ▼
Graph Transformer
      │
      ▼
Drug Embedding
      │
      ├────────────┐
      │            │
      ▼            │
Cross-Modal Transformer
      ▲            │
      │            │
Protein Embedding  │
      ▲            │
      │            │
ESM-2 Encoder      │
      ▲            │
Protein Sequence   │
      │            │
      └────────────┘

             │
             ▼

      Causal Module

             │
             ▼

 Monte Carlo Dropout

             │
             ▼

      Prediction Head

      ├── Classification
      └── Affinity
```

---

# Evaluation

Run model evaluation:

```bash
python evaluation/test.py \
--checkpoint checkpoints/best_model.pt
```

---

# Metrics

## Classification

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC

## Affinity Prediction

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Concordance Index (CI)
* Pearson Correlation
* Spearman Correlation

## Calibration

* Expected Calibration Error (ECE)
* Maximum Calibration Error (MCE)
* Brier Score
* Negative Log Likelihood (NLL)

---

# Explainability Analysis

Generate attribution maps:

```bash
python evaluation/explainability.py
```

Outputs:

```text
outputs/explainability/
│
├── attention_maps.png
├── attribution_heatmaps.png
├── residue_importance.csv
└── counterfactual_results.csv
```

# Uncertainty Estimation

Run uncertainty analysis:

```bash
python evaluation/calibration.py
```

Outputs:

```text
outputs/calibration/
│
├── reliability_diagram.png
├── uncertainty_histogram.png
├── calibration_curve.png
└── calibration_metrics.csv
```

---

# Molecular Docking Validation

Install AutoDock Vina.

Run docking pipeline:

```bash
python docking/vina_pipeline.py
```

Validate predicted interactions:

```bash
python docking/docking_validation.py
```

Outputs:

```text
outputs/docking/
│
├── docking_scores.csv
├── docking_structures/
└── interaction_visualizations/
```

---

# Hardware Requirements

## Minimum

* GPU: NVIDIA RTX 3060 (12 GB)
* RAM: 32 GB
* Storage: 100 GB

## Recommended

* GPU: RTX 4090 / A100
* RAM: 64 GB+
* Storage: 200 GB+

---

# Reproducibility

Set seed:

```python
import torch
import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

---

# Citation

If you use this repository, please cite:

```bibtex
@article{kotkondawar2026causalDTI,
  title={A Causal Multi-Modal Transformer Framework with Uncertainty-Aware Explainability for Mechanistic DTI Prediction},
  author={Kotkondawar, Roshan and Nicose, Kunal},
  journal={--},
  year={2026}
}
```

---

# License

This project is released under the MIT License.

---

# Contact

Roshan Kotkondawar

St. Vincent Pallotti College of Engineering & Technology, India

Email: [lncs@springer.com](kotkondawarroshan@gmail.com)

---

Kunal Nicose

VNIT, India

Email: [kunal.nicose@outlook.com](mailto:kunal.nicose@outlook.com)

---

# Acknowledgements

The authors acknowledge:

* PyTorch
* PyTorch Geometric
* RDKit
* Meta AI ESM-2
* Scikit-learn
* AutoDock Vina
* BindingDB
* DAVIS
* KIBA
* DrugBank
* PDBBind

for providing open-source tools and datasets that made this research possible.
