"""
=============================================================
Prediction Heads

Causal Multi-Modal DTI Framework

Supported Tasks
---------------
1. Binary Classification
2. Affinity Regression
3. Multi-Task Learning

Input:
------
Fused Embedding [B, 512]

Outputs:
--------
Classification:
    Probability [B,1]

Regression:
    Affinity Score [B,1]

Author:
Roshan Kotkondawar
Kunal Nicose
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MLP Block
# ============================================================

class MLPBlock(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        dropout=0.2
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Linear(
                in_features,
                out_features
            ),

            nn.LayerNorm(
                out_features
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            )
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# Binary Classification Head
# ============================================================

class ClassificationHead(nn.Module):

    def __init__(
        self,
        input_dim=512,
        hidden_dims=(512, 256, 128),
        dropout=0.2
    ):
        super().__init__()

        layers = []

        prev_dim = input_dim

        for hidden_dim in hidden_dims:

            layers.append(
                MLPBlock(
                    prev_dim,
                    hidden_dim,
                    dropout
                )
            )

            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(
            *layers
        )

        self.classifier = nn.Linear(
            prev_dim,
            1
        )

    def forward(
        self,
        x,
        return_logits=False
    ):

        features = self.feature_extractor(
            x
        )

        logits = self.classifier(
            features
        )

        if return_logits:
            return logits

        probabilities = torch.sigmoid(
            logits
        )

        return probabilities


# ============================================================
# Affinity Regression Head
# ============================================================

class RegressionHead(nn.Module):

    def __init__(
        self,
        input_dim=512,
        hidden_dims=(512, 256, 128),
        dropout=0.2
    ):
        super().__init__()

        layers = []

        prev_dim = input_dim

        for hidden_dim in hidden_dims:

            layers.append(
                MLPBlock(
                    prev_dim,
                    hidden_dim,
                    dropout
                )
            )

            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(
            *layers
        )

        self.regressor = nn.Linear(
            prev_dim,
            1
        )

    def forward(self, x):

        features = self.feature_extractor(
            x
        )

        affinity = self.regressor(
            features
        )

        return affinity


# ============================================================
# Multi Task Predictor
# ============================================================

class MultiTaskPredictor(nn.Module):

    def __init__(
        self,
        input_dim=512,
        hidden_dims=(512, 256, 128),
        dropout=0.2
    ):
        super().__init__()

        self.shared = nn.Sequential(

            MLPBlock(
                input_dim,
                hidden_dims[0],
                dropout
            ),

            MLPBlock(
                hidden_dims[0],
                hidden_dims[1],
                dropout
            )
        )

        shared_dim = hidden_dims[1]

        # Classification Branch

        self.classification_branch = nn.Sequential(

            MLPBlock(
                shared_dim,
                hidden_dims[2],
                dropout
            ),

            nn.Linear(
                hidden_dims[2],
                1
            )
        )

        # Regression Branch

        self.regression_branch = nn.Sequential(

            MLPBlock(
                shared_dim,
                hidden_dims[2],
                dropout
            ),

            nn.Linear(
                hidden_dims[2],
                1
            )
        )

    def forward(self, x):

        shared_features = self.shared(
            x
        )

        logits = (
            self.classification_branch(
                shared_features
            )
        )

        probabilities = torch.sigmoid(
            logits
        )

        affinity = (
            self.regression_branch(
                shared_features
            )
        )

        return {

            "classification_logits":
                logits,

            "classification_probability":
                probabilities,

            "regression_output":
                affinity
        }


# ============================================================
# Unified Predictor
# ============================================================

class Predictor(nn.Module):

    """
    Wrapper used by full_model.py
    """

    def __init__(
        self,
        task="classification",
        input_dim=512,
        hidden_dims=(512, 256, 128),
        dropout=0.2
    ):
        super().__init__()

        self.task = task

        if task == "classification":

            self.head = ClassificationHead(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )

        elif task == "regression":

            self.head = RegressionHead(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )

        elif task == "multitask":

            self.head = MultiTaskPredictor(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )

        else:

            raise ValueError(
                f"Unsupported task: {task}"
            )

    def forward(self, x):

        return self.head(x)


# ============================================================
# Factory Function
# ============================================================

def build_predictor(config):

    predictor_cfg = config[
        "predictor"
    ]

    task = config[
        "dataset"
    ]["task"]

    model = Predictor(

        task=task,

        input_dim=512,

        hidden_dims=tuple(
            predictor_cfg[
                "hidden_dims"
            ]
        ),

        dropout=
        predictor_cfg[
            "dropout"
        ]
    )

    return model


# ============================================================
# Utility Functions
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":

    batch_size = 8

    x = torch.randn(
        batch_size,
        512
    )

    print(
        "\n=== Classification ==="
    )

    clf = Predictor(
        task="classification"
    )

    out = clf(x)

    print(
        out.shape
    )

    print(
        "\n=== Regression ==="
    )

    reg = Predictor(
        task="regression"
    )

    out = reg(x)

    print(
        out.shape
    )

    print(
        "\n=== MultiTask ==="
    )

    mt = Predictor(
        task="multitask"
    )

    out = mt(x)

    for k, v in out.items():

        print(
            k,
            v.shape
        )

    print(
        "\nTrainable Parameters:"
    )

    print(
        count_parameters(clf)
    )