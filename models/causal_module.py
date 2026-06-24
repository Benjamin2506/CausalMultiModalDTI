"""
=============================================================
Causal Explainability Module

Causal Multi-Modal DTI Framework

Components
----------
1. Feature Intervention
2. Counterfactual Generation
3. Feature Masking
4. Causal Importance Scoring
5. Attribution Refinement
6. Causal Consistency Loss

Input
-----
Fused Representation [B, 512]

Output
------
Refined Representation [B, 512]

Author:
Roshan Kotkondawar
Kunal Nicose
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Feature Masking
# ============================================================

class FeatureMasking(nn.Module):
    """
    Random intervention masking.
    """

    def __init__(
        self,
        intervention_probability=0.10
    ):
        super().__init__()

        self.intervention_probability = (
            intervention_probability
        )

    def forward(self, x):

        if not self.training:
            return x

        mask = (
            torch.rand_like(x)
            > self.intervention_probability
        ).float()

        return x * mask


# ============================================================
# Counterfactual Generator
# ============================================================

class CounterfactualGenerator(nn.Module):
    """
    Generate perturbed samples.
    """

    def __init__(
        self,
        noise_std=0.05
    ):
        super().__init__()

        self.noise_std = noise_std

    def forward(self, x):

        noise = torch.randn_like(x)

        noise = noise * self.noise_std

        return x + noise


# ============================================================
# Causal Importance Network
# ============================================================

class CausalImportanceNetwork(nn.Module):
    """
    Learns feature-level causal importance.
    """

    def __init__(
        self,
        hidden_dim=512
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim // 2,
                hidden_dim
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        importance = self.network(x)

        return importance


# ============================================================
# Attribution Refiner
# ============================================================

class AttributionRefiner(nn.Module):
    """
    Refine representation using
    learned causal importance.
    """

    def __init__(
        self,
        hidden_dim=512
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

    def forward(
        self,
        x,
        importance
    ):

        refined = (
            x * importance
        )

        return refined


# ============================================================
# Main Causal Module
# ============================================================

class CausalModule(nn.Module):
    """
    Main causal reasoning module.
    """

    def __init__(
        self,
        hidden_dim=512,
        intervention_probability=0.10,
        counterfactual_samples=10,
        noise_std=0.05
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.counterfactual_samples = (
            counterfactual_samples
        )

        self.masking = FeatureMasking(
            intervention_probability
        )

        self.counterfactual_generator = (
            CounterfactualGenerator(
                noise_std
            )
        )

        self.importance_network = (
            CausalImportanceNetwork(
                hidden_dim
            )
        )

        self.refiner = (
            AttributionRefiner(
                hidden_dim
            )
        )

        self.output_norm = nn.LayerNorm(
            hidden_dim
        )

    # =========================================================
    # Intervention
    # =========================================================

    def intervene(self, x):

        return self.masking(x)

    # =========================================================
    # Counterfactual Samples
    # =========================================================

    def generate_counterfactuals(
        self,
        x
    ):

        samples = []

        for _ in range(
            self.counterfactual_samples
        ):

            cf = (
                self.counterfactual_generator(
                    x
                )
            )

            samples.append(cf)

        samples = torch.stack(
            samples,
            dim=0
        )

        return samples

    # =========================================================
    # Importance Scores
    # =========================================================

    def compute_importance(
        self,
        x
    ):

        importance = (
            self.importance_network(
                x
            )
        )

        return importance

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        fused_embedding
    ):
        """
        Parameters
        ----------
        fused_embedding : [B,512]

        Returns
        -------
        refined_embedding
        causal_info
        """

        # ----------------------------------------
        # Intervention
        # ----------------------------------------

        intervened = self.intervene(
            fused_embedding
        )

        # ----------------------------------------
        # Importance Estimation
        # ----------------------------------------

        importance = (
            self.compute_importance(
                intervened
            )
        )

        # ----------------------------------------
        # Attribution Refinement
        # ----------------------------------------

        refined = self.refiner(
            intervened,
            importance
        )

        refined = (
            self.output_norm(
                refined
            )
        )

        # ----------------------------------------
        # Counterfactual Analysis
        # ----------------------------------------

        counterfactuals = (
            self.generate_counterfactuals(
                refined
            )
        )

        causal_info = {

            "importance_scores":
                importance,

            "counterfactuals":
                counterfactuals,

            "intervened_embedding":
                intervened
        }

        return (
            refined,
            causal_info
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

        self.mse = nn.MSELoss()

    def forward(
        self,
        original_embedding,
        counterfactual_embeddings
    ):

        losses = []

        for i in range(
            counterfactual_embeddings.size(0)
        ):

            loss = self.mse(
                original_embedding,
                counterfactual_embeddings[i]
            )

            losses.append(loss)

        return torch.stack(
            losses
        ).mean()


# ============================================================
# Factory Function
# ============================================================

def build_causal_module(
    config
):

    causal_cfg = config[
        "causal_module"
    ]

    model = CausalModule(

        hidden_dim=512,

        intervention_probability=
        causal_cfg[
            "intervention_probability"
        ],

        counterfactual_samples=
        causal_cfg[
            "counterfactual_samples"
        ]
    )

    return model


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":

    x = torch.randn(
        16,
        512
    )

    module = CausalModule()

    refined, info = module(x)

    print(
        "Input Shape:",
        x.shape
    )

    print(
        "Refined Shape:",
        refined.shape
    )

    print(
        "Importance Shape:",
        info[
            "importance_scores"
        ].shape
    )

    print(
        "Counterfactual Shape:",
        info[
            "counterfactuals"
        ].shape
    )