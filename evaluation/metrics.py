"""
=============================================================
Evaluation Metrics

Causal Multi-Modal DTI Framework

Supports:
-----------
BindingDB  -> Classification
DAVIS      -> Regression
KIBA       -> Regression

=============================================================
"""

from __future__ import annotations

import numpy as np

from scipy.stats import (
    pearsonr,
    spearmanr
)

from sklearn.metrics import (

    roc_auc_score,

    average_precision_score,

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    matthews_corrcoef,

    mean_squared_error,

    mean_absolute_error,

    r2_score,

    brier_score_loss
)

# ============================================================
# Classification Metrics
# ============================================================

def classification_metrics(
    y_true,
    y_prob,
    threshold=0.5
):
    """
    Binary classification metrics.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (
        y_prob >= threshold
    ).astype(np.int32)

    metrics = {}

    try:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            y_prob
        )
    except Exception:
        metrics["roc_auc"] = 0.0

    try:
        metrics["pr_auc"] = average_precision_score(
            y_true,
            y_prob
        )
    except Exception:
        metrics["pr_auc"] = 0.0

    metrics["accuracy"] = accuracy_score(
        y_true,
        y_pred
    )

    metrics["precision"] = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    metrics["recall"] = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    metrics["f1"] = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    metrics["mcc"] = matthews_corrcoef(
        y_true,
        y_pred
    )

    return metrics


# ============================================================
# Concordance Index
# ============================================================

def concordance_index(
    y_true,
    y_pred
):
    """
    Concordance Index (CI)

    Used by DAVIS and KIBA.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = 0
    h_sum = 0.0

    for i in range(len(y_true)):

        for j in range(i + 1, len(y_true)):

            if y_true[i] == y_true[j]:
                continue

            n += 1

            if y_pred[i] == y_pred[j]:

                h_sum += 0.5

            elif (

                (y_true[i] > y_true[j]
                 and y_pred[i] > y_pred[j])

                or

                (y_true[i] < y_true[j]
                 and y_pred[i] < y_pred[j])

            ):

                h_sum += 1.0

    if n == 0:
        return 0.0

    return float(h_sum / n)


# ============================================================
# Regression Metrics
# ============================================================

def regression_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {}

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    metrics["mse"] = float(mse)

    metrics["rmse"] = float(
        np.sqrt(mse)
    )

    metrics["mae"] = float(
        mean_absolute_error(
            y_true,
            y_pred
        )
    )

    metrics["r2"] = float(
        r2_score(
            y_true,
            y_pred
        )
    )

    try:

        metrics["pearson"] = float(
            pearsonr(
                y_true,
                y_pred
            )[0]
        )

    except Exception:

        metrics["pearson"] = 0.0

    try:

        metrics["spearman"] = float(
            spearmanr(
                y_true,
                y_pred
            )[0]
        )

    except Exception:

        metrics["spearman"] = 0.0

    metrics["ci"] = concordance_index(
        y_true,
        y_pred
    )

    return metrics


# ============================================================
# Calibration Metrics
# ============================================================

def expected_calibration_error(
    y_true,
    y_prob,
    n_bins=10
):
    """
    ECE.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        mask = (

            (y_prob >= bins[i])

            &

            (y_prob < bins[i + 1])

        )

        if mask.sum() == 0:
            continue

        confidence = np.mean(
            y_prob[mask]
        )

        accuracy = np.mean(
            y_true[mask]
        )

        ece += (

            np.abs(
                confidence
                - accuracy
            )

            *

            (
                mask.sum()
                /
                len(y_true)
            )
        )

    return float(ece)


def calibration_metrics(
    y_true,
    y_prob
):
    """
    Calibration metrics.
    """

    return {

        "ece":
        expected_calibration_error(
            y_true,
            y_prob
        ),

        "brier":
        brier_score_loss(
            y_true,
            y_prob
        )
    }


# ============================================================
# Average Meter
# ============================================================

class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):

        self.sum = 0.0
        self.count = 0

    def update(
        self,
        value,
        n=1
    ):

        self.sum += value * n
        self.count += n

    @property
    def average(self):

        if self.count == 0:
            return 0.0

        return self.sum / self.count


# ============================================================
# Metric Tracker
# ============================================================

class MetricTracker:

    """
    Generic tracker.
    """

    def __init__(self):

        self.targets = []
        self.outputs = []

    def update(
        self,
        targets,
        outputs
    ):

        self.targets.extend(
            np.asarray(
                targets
            ).flatten().tolist()
        )

        self.outputs.extend(
            np.asarray(
                outputs
            ).flatten().tolist()
        )

    def classification(
        self,
        threshold=0.5
    ):

        return classification_metrics(

            self.targets,

            self.outputs,

            threshold
        )

    def regression(
        self
    ):

        return regression_metrics(

            self.targets,

            self.outputs
        )

    def reset(self):

        self.targets = []
        self.outputs = []


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":

    print(
        "Metrics module loaded successfully."
    )