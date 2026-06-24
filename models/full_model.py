"""
=============================================================
Full Causal Multi-Modal DTI Model

Supports:
----------
✓ BindingDB
✓ DAVIS
✓ KIBA

✓ ProteinRepository
✓ Protein IDs
✓ ESM Embedding Cache
✓ Causal Explainability
✓ Uncertainty Estimation

Author:
Roshan Kotkondawar
Kunal Nicose
=============================================================
"""

import torch
import torch.nn as nn

from models.graph_transformer import (
    build_graph_transformer
)

from models.esm_encoder import (
    build_esm_encoder
)

from models.cross_modal_transformer import (
    build_cross_modal_transformer
)

from models.causal_module import (
    build_causal_module
)

from models.predictor import (
    build_predictor
)

from datasets.protein_dataset import (
    get_sequences_from_ids,
    get_cached_embeddings
)


# ============================================================
# Full Model
# ============================================================

class CausalDTIModel(nn.Module):

    def __init__(
        self,
        config,
        protein_repository=None,
        embedding_cache=None
    ):
        super().__init__()

        self.config = config

        self.task = (
            config["dataset"]["task"]
        )

        self.protein_repository = (
            protein_repository
        )

        self.embedding_cache = (
            embedding_cache
        )

        # ----------------------------------------------------
        # Drug Encoder
        # ----------------------------------------------------

        self.graph_encoder = (
            build_graph_transformer(
                config
            )
        )

        # ----------------------------------------------------
        # Protein Encoder
        # ----------------------------------------------------

        self.protein_encoder = (
            build_esm_encoder(
                config
            )
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        self.cross_modal_transformer = (
            build_cross_modal_transformer(
                config
            )
        )

        # ----------------------------------------------------
        # Causal Module
        # ----------------------------------------------------

        self.causal_module = (
            build_causal_module(
                config
            )
        )

        # ----------------------------------------------------
        # Predictor
        # ----------------------------------------------------

        self.predictor = (
            build_predictor(
                config
            )
        )

    # =========================================================
    # Drug Encoding
    # =========================================================

    def encode_drug(
        self,
        graph_batch
    ):

        return self.graph_encoder(
            graph_batch
        )

    # =========================================================
    # Protein Retrieval
    # =========================================================

    def get_sequences(
        self,
        protein_ids
    ):

        if self.protein_repository is None:

            raise RuntimeError(
                "ProteinRepository is required."
            )

        return get_sequences_from_ids(
            protein_ids,
            self.protein_repository
        )

    # =========================================================
    # Protein Encoding
    # =========================================================

    def encode_protein(
        self,
        protein_ids
    ):

        device = next(
            self.parameters()
        ).device

        # --------------------------------------------
        # Cached ESM Embeddings
        # --------------------------------------------

        if self.embedding_cache is not None:

            try:

                embeddings = (
                    get_cached_embeddings(
                        protein_ids,
                        self.embedding_cache,
                        device=device
                    )
                )

                return embeddings

            except Exception:

                pass

        # --------------------------------------------
        # Sequence Retrieval
        # --------------------------------------------

        sequences = (
            self.get_sequences(
                protein_ids
            )
        )

        embeddings = (
            self.protein_encoder(
                sequences
            )
        )

        return embeddings

    # =========================================================
    # Fusion
    # =========================================================

    def fuse(
        self,
        drug_embedding,
        protein_embedding
    ):

        return (
            self.cross_modal_transformer(
                drug_embedding,
                protein_embedding
            )
        )

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        graph_batch,
        protein_ids
    ):
        """
        Parameters
        ----------
        graph_batch:
            PyG Batch

        protein_ids:
            Tensor [B]

        Returns
        -------
        Dictionary
        """

        # --------------------------------------------
        # Drug Encoder
        # --------------------------------------------

        drug_embedding = (
            self.encode_drug(
                graph_batch
            )
        )

        # --------------------------------------------
        # Protein Encoder
        # --------------------------------------------

        protein_embedding = (
            self.encode_protein(
                protein_ids
            )
        )

        # --------------------------------------------
        # Fusion
        # --------------------------------------------

        fused_embedding = (
            self.fuse(
                drug_embedding,
                protein_embedding
            )
        )

        # --------------------------------------------
        # Causal Module
        # --------------------------------------------

        (
            refined_embedding,
            causal_info
        ) = self.causal_module(
            fused_embedding
        )

        # --------------------------------------------
        # Prediction
        # --------------------------------------------

        prediction = (
            self.predictor(
                refined_embedding
            )
        )

        outputs = {

            "prediction":
                prediction,

            "drug_embedding":
                drug_embedding,

            "protein_embedding":
                protein_embedding,

            "fused_embedding":
                fused_embedding,

            "refined_embedding":
                refined_embedding,

            "causal_info":
                causal_info
        }

        return outputs

    # =========================================================
    # Predict
    # =========================================================

    @torch.no_grad()
    def predict(
        self,
        graph_batch,
        protein_ids
    ):

        self.eval()

        outputs = self.forward(
            graph_batch,
            protein_ids
        )

        return outputs[
            "prediction"
        ]

    # =========================================================
    # Embedding Extraction
    # =========================================================

    @torch.no_grad()
    def extract_embeddings(
        self,
        graph_batch,
        protein_ids
    ):

        self.eval()

        outputs = self.forward(
            graph_batch,
            protein_ids
        )

        return {

            "drug_embedding":
                outputs[
                    "drug_embedding"
                ],

            "protein_embedding":
                outputs[
                    "protein_embedding"
                ],

            "fused_embedding":
                outputs[
                    "fused_embedding"
                ],

            "refined_embedding":
                outputs[
                    "refined_embedding"
                ]
        }

    # =========================================================
    # Parameter Count
    # =========================================================

    def count_parameters(
        self
    ):

        return sum(

            p.numel()

            for p in self.parameters()

            if p.requires_grad
        )


# ============================================================
# Factory
# ============================================================

def build_model(
    config,
    protein_repository=None,
    embedding_cache=None
):

    return CausalDTIModel(
        config,
        protein_repository,
        embedding_cache
    )


# ============================================================
# Utilities
# ============================================================

def count_trainable_parameters(
    model
):

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad
    )


def count_total_parameters(
    model
):

    return sum(

        p.numel()

        for p in model.parameters()
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print(
        "\nRepository-Compatible Full Model Loaded"
    )