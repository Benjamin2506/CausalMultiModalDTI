"""
=============================================================
Loss Functions

Causal Multi-Modal DTI Framework

Supported Tasks
---------------
1. Binary Classification
2. Affinity Regression
3. Multi-Task Learning

Additional Losses
-----------------
1. Causal Consistency Loss
2. Uncertainty Regularization
3. Total Combined Loss

Author:
Roshan Kotkondawar
Kunal Nicose
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Binary Classification Loss
# ============================================================

class BinaryClassificationLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits,
        targets
    ):

        targets = targets.float()

        return self.loss_fn(
            logits.view(-1),
            targets.view(-1)
        )


# ============================================================
# Regression Loss
# ============================================================

class RegressionLoss(nn.Module):

    def __init__(
        self,
        loss_type="mse"
    ):
        super().__init__()

        self.loss_type = loss_type

        if loss_type == "mse":

            self.loss_fn = nn.MSELoss()

        elif loss_type == "mae":

            self.loss_fn = nn.L1Loss()

        elif loss_type == "smooth_l1":

            self.loss_fn = nn.SmoothL1Loss()

        else:

            raise ValueError(
                f"Unsupported regression loss: "
                f"{loss_type}"
            )

    def forward(
        self,
        predictions,
        targets
    ):

        predictions = predictions.float()

        targets = targets.float()

        return self.loss_fn(
            predictions.view(-1),
            targets.view(-1)
        )


# ============================================================
# Causal Consistency Loss
# ============================================================

class CausalConsistencyLoss(nn.Module):

    """
    Encourage robustness under intervention.
    """

    def __init__(self):
        super().__init__()

        self.loss_fn = nn.MSELoss()

    def forward(
        self,
        original_embedding,
        counterfactual_embeddings
    ):

        losses = []

        num_cf = (
            counterfactual_embeddings.shape[0]
        )

        for i in range(num_cf):

            losses.append(

                self.loss_fn(
                    original_embedding,
                    counterfactual_embeddings[i]
                )
            )

        if len(losses) == 0:

            return torch.tensor(
                0.0,
                device=original_embedding.device
            )

        return torch.stack(
            losses
        ).mean()


# ============================================================
# Uncertainty Regularization
# ============================================================

class UncertaintyRegularization(nn.Module):

    """
    Penalize excessive predictive variance.
    """

    def __init__(
        self,
        reduction="mean"
    ):
        super().__init__()

        self.reduction = reduction

    def forward(
        self,
        predictive_variance
    ):

        if predictive_variance is None:

            return torch.tensor(
                0.0
            )

        if self.reduction == "mean":

            return predictive_variance.mean()

        elif self.reduction == "sum":

            return predictive_variance.sum()

        else:

            return predictive_variance.mean()


# ============================================================
# Multi Task Loss
# ============================================================

class MultiTaskLoss(nn.Module):

    """
    Classification + Regression
    """

    def __init__(
        self,
        classification_weight=1.0,
        regression_weight=1.0
    ):
        super().__init__()

        self.classification_weight = (
            classification_weight
        )

        self.regression_weight = (
            regression_weight
        )

        self.cls_loss = (
            BinaryClassificationLoss()
        )

        self.reg_loss = (
            RegressionLoss()
        )

    def forward(
        self,
        classification_logits,
        classification_targets,
        regression_predictions,
        regression_targets
    ):

        cls = self.cls_loss(
            classification_logits,
            classification_targets
        )

        reg = self.reg_loss(
            regression_predictions,
            regression_targets
        )

        total = (
            self.classification_weight * cls
            +
            self.regression_weight * reg
        )

        return {

            "total_loss": total,

            "classification_loss":
                cls,

            "regression_loss":
                reg
        }


# ============================================================
# Complete DTI Loss
# ============================================================

class DTICompositeLoss(nn.Module):

    """
    Main loss used by training scripts.
    """

    def __init__(
        self,
        task="classification",
        causal_weight=0.10,
        uncertainty_weight=0.05,
        regression_loss_type="mse"
    ):
        super().__init__()

        self.task = task

        self.causal_weight = (
            causal_weight
        )

        self.uncertainty_weight = (
            uncertainty_weight
        )

        self.causal_loss_fn = (
            CausalConsistencyLoss()
        )

        self.uncertainty_loss_fn = (
            UncertaintyRegularization()
        )

        if task == "classification":

            self.primary_loss = (
                BinaryClassificationLoss()
            )

        elif task == "regression":

            self.primary_loss = (
                RegressionLoss(
                    regression_loss_type
                )
            )

        else:

            raise ValueError(
                f"Unsupported task: {task}"
            )

    def forward(
        self,
        predictions,
        targets,
        refined_embedding=None,
        counterfactual_embeddings=None,
        predictive_variance=None
    ):

        primary_loss = (
            self.primary_loss(
                predictions,
                targets
            )
        )

        causal_loss = torch.tensor(
            0.0,
            device=predictions.device
        )

        uncertainty_loss = torch.tensor(
            0.0,
            device=predictions.device
        )

        # ----------------------------------------
        # Causal Regularization
        # ----------------------------------------

        if (
            refined_embedding is not None
            and
            counterfactual_embeddings
            is not None
        ):

            causal_loss = (
                self.causal_loss_fn(
                    refined_embedding,
                    counterfactual_embeddings
                )
            )

        # ----------------------------------------
        # Uncertainty Regularization
        # ----------------------------------------

        if predictive_variance is not None:

            uncertainty_loss = (
                self.uncertainty_loss_fn(
                    predictive_variance
                )
            )

        total_loss = (

            primary_loss

            +

            self.causal_weight
            *
            causal_loss

            +

            self.uncertainty_weight
            *
            uncertainty_loss
        )

        return {

            "total_loss":
                total_loss,

            "primary_loss":
                primary_loss,

            "causal_loss":
                causal_loss,

            "uncertainty_loss":
                uncertainty_loss
        }


# ============================================================
# Factory Function
# ============================================================

def build_loss_function(
    config
):

    task = config[
        "dataset"
    ]["task"]

    loss_cfg = config[
        "loss"
    ]

    regression_type = (
        loss_cfg.get(
            "regression_loss",
            "mse"
        )
    )

    criterion = DTICompositeLoss(

        task=task,

        causal_weight=
        loss_cfg[
            "causal_regularization_weight"
        ],

        uncertainty_weight=
        loss_cfg[
            "uncertainty_regularization_weight"
        ],

        regression_loss_type=
        regression_type
    )

    return criterion


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":

    B = 16

    predictions = torch.randn(
        B,
        1
    )

    targets = torch.randint(
        0,
        2,
        (
            B,
            1
        )
    ).float()

    refined_embedding = torch.randn(
        B,
        512
    )

    counterfactuals = torch.randn(
        10,
        B,
        512
    )

    variance = torch.rand(
        B,
        1
    )

    criterion = DTICompositeLoss(
        task="classification"
    )

    losses = criterion(

        predictions,

        targets,

        refined_embedding,

        counterfactuals,

        variance
    )

    print("\nLoss Summary")

    for k, v in losses.items():

        print(
            f"{k}: "
            f"{v.item():.6f}"
        )