"""
=============================================================
Uncertainty Estimation Module

Causal Multi-Modal DTI Framework

Methods:
--------
1. Monte Carlo Dropout
2. Predictive Mean
3. Predictive Variance
4. Epistemic Uncertainty
5. Confidence Score

Author:
Roshan Kotkondawar
Kunal Nicose
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Enable Dropout During Inference
# ============================================================

def enable_mc_dropout(model):
    """
    Activate dropout layers
    while keeping the rest
    of the model in evaluation mode.
    """

    for module in model.modules():

        if isinstance(
            module,
            nn.Dropout
        ):
            module.train()


# ============================================================
# Monte Carlo Dropout
# ============================================================

class MonteCarloDropout(nn.Module):
    """
    MC Dropout Wrapper
    """

    def __init__(
        self,
        mc_samples=30
    ):
        super().__init__()

        self.mc_samples = mc_samples

    @torch.no_grad()
    def forward(
        self,
        model,
        *inputs,
        **kwargs
    ):
        """
        Parameters
        ----------
        model : nn.Module

        Returns
        -------
        mean_prediction
        predictive_variance
        all_predictions
        """

        model.eval()

        enable_mc_dropout(
            model
        )

        predictions = []

        for _ in range(
            self.mc_samples
        ):

            output = model(
                *inputs,
                **kwargs
            )

            predictions.append(
                output.unsqueeze(0)
            )

        predictions = torch.cat(
            predictions,
            dim=0
        )

        mean_prediction = (
            predictions.mean(dim=0)
        )

        predictive_variance = (
            predictions.var(
                dim=0,
                unbiased=False
            )
        )

        return (
            mean_prediction,
            predictive_variance,
            predictions
        )


# ============================================================
# Uncertainty Metrics
# ============================================================

class UncertaintyEstimator:
    """
    Uncertainty Utility Class
    """

    @staticmethod
    def predictive_mean(
        predictions
    ):

        return predictions.mean(
            dim=0
        )

    @staticmethod
    def predictive_variance(
        predictions
    ):

        return predictions.var(
            dim=0,
            unbiased=False
        )

    @staticmethod
    def predictive_std(
        predictions
    ):

        return predictions.std(
            dim=0,
            unbiased=False
        )

    @staticmethod
    def confidence_score(
        variance,
        epsilon=1e-8
    ):
        """
        Higher variance
        → lower confidence
        """

        confidence = (
            1.0 /
            (
                variance +
                epsilon
            )
        )

        return confidence

    @staticmethod
    def normalized_confidence(
        variance,
        epsilon=1e-8
    ):

        confidence = (
            1.0 /
            (
                variance +
                epsilon
            )
        )

        confidence = (
            confidence
            /
            confidence.max()
        )

        return confidence


# ============================================================
# Classification Uncertainty
# ============================================================

class ClassificationUncertainty:
    """
    Binary Classification
    """

    @staticmethod
    def entropy(
        probabilities,
        epsilon=1e-8
    ):
        """
        Shannon entropy.
        """

        return -(
            probabilities *
            torch.log(
                probabilities + epsilon
            )
            +
            (
                1.0 -
                probabilities
            )
            *
            torch.log(
                1.0 -
                probabilities +
                epsilon
            )
        )

    @staticmethod
    def mutual_information(
        predictions,
        epsilon=1e-8
    ):
        """
        BALD score.

        Parameters
        ----------
        predictions:
            [T, B, 1]
        """

        mean_prob = (
            predictions.mean(
                dim=0
            )
        )

        predictive_entropy = -(
            mean_prob *
            torch.log(
                mean_prob + epsilon
            )
            +
            (
                1 -
                mean_prob
            )
            *
            torch.log(
                1 -
                mean_prob +
                epsilon
            )
        )

        entropy_per_sample = -(
            predictions *
            torch.log(
                predictions +
                epsilon
            )
            +
            (
                1 -
                predictions
            )
            *
            torch.log(
                1 -
                predictions +
                epsilon
            )
        )

        expected_entropy = (
            entropy_per_sample.mean(
                dim=0
            )
        )

        mutual_info = (
            predictive_entropy
            -
            expected_entropy
        )

        return mutual_info


# ============================================================
# Regression Uncertainty
# ============================================================

class RegressionUncertainty:
    """
    Regression Metrics
    """

    @staticmethod
    def prediction_interval(
        mean,
        variance,
        confidence=1.96
    ):
        """
        95% interval.
        """

        std = torch.sqrt(
            variance
        )

        lower = (
            mean -
            confidence * std
        )

        upper = (
            mean +
            confidence * std
        )

        return (
            lower,
            upper
        )

    @staticmethod
    def coefficient_of_variation(
        mean,
        variance,
        epsilon=1e-8
    ):
        """
        CV = std / mean
        """

        std = torch.sqrt(
            variance
        )

        return (
            std /
            (
                mean.abs()
                + epsilon
            )
        )


# ============================================================
# Calibration Metrics
# ============================================================

class CalibrationMetrics:
    """
    Calibration utilities.
    """

    @staticmethod
    def expected_calibration_error(
        confidences,
        accuracies,
        n_bins=15
    ):
        """
        ECE computation.
        """

        bin_boundaries = torch.linspace(
            0,
            1,
            n_bins + 1
        )

        ece = torch.zeros(
            1,
            device=confidences.device
        )

        for i in range(n_bins):

            lower = (
                bin_boundaries[i]
            )

            upper = (
                bin_boundaries[i + 1]
            )

            mask = (
                confidences > lower
            ) & (
                confidences <= upper
            )

            if mask.sum() > 0:

                acc = (
                    accuracies[
                        mask
                    ].float().mean()
                )

                conf = (
                    confidences[
                        mask
                    ].mean()
                )

                weight = (
                    mask.float().mean()
                )

                ece += (
                    torch.abs(
                        conf - acc
                    )
                    * weight
                )

        return ece.item()


# ============================================================
# Factory
# ============================================================

def build_uncertainty_module(
    config
):
    """
    Build MC Dropout module
    from YAML config.
    """

    uncertainty_cfg = config[
        "uncertainty"
    ]

    module = MonteCarloDropout(
        mc_samples=
        uncertainty_cfg[
            "mc_samples"
        ]
    )

    return module


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":

    class DummyModel(nn.Module):

        def __init__(self):

            super().__init__()

            self.fc = nn.Sequential(
                nn.Linear(
                    512,
                    128
                ),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(
                    128,
                    1
                )
            )

        def forward(
            self,
            x
        ):
            return self.fc(x)

    model = DummyModel()

    x = torch.randn(
        8,
        512
    )

    mc = MonteCarloDropout(
        mc_samples=20
    )

    mean_pred, variance, preds = mc(
        model,
        x
    )

    print(
        "Mean Shape:",
        mean_pred.shape
    )

    print(
        "Variance Shape:",
        variance.shape
    )

    print(
        "Predictions Shape:",
        preds.shape
    )