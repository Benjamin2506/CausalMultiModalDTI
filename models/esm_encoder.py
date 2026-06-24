"""
=============================================================
ESM-2 Protein Encoder

Causal Multi-Modal DTI Framework

Architecture

Protein Sequence
        ↓
ESM-2 Tokenizer
        ↓
ESM-2 Transformer
        ↓
Mean Pooling
        ↓
Projection Head
        ↓
Protein Embedding (1280)

Supported Models
----------------
esm2_t6_8M_UR50D
esm2_t12_35M_UR50D
esm2_t30_150M_UR50D
esm2_t33_650M_UR50D

Default:
esm2_t33_650M_UR50D

Author:
Roshan Kotkondawar
Kunal Nicose
=============================================================
"""

import torch
import torch.nn as nn

import esm


class ESMProteinEncoder(nn.Module):
    """
    ESM-2 Protein Encoder
    """

    def __init__(
        self,
        model_name="esm2_t33_650M_UR50D",
        embedding_dim=1280,
        freeze_encoder=True,
        pooling="mean",
        max_length=1024,
        dropout=0.2
    ):
        super().__init__()

        self.model_name = model_name

        self.embedding_dim = embedding_dim

        self.pooling = pooling

        self.max_length = max_length

        self.freeze_encoder = freeze_encoder

        # =====================================================
        # Load ESM Model
        # =====================================================

        self.esm_model, self.alphabet = (
            self._load_model(
                model_name
            )
        )

        self.batch_converter = (
            self.alphabet.get_batch_converter()
        )

        # =====================================================
        # Freeze Parameters
        # =====================================================

        if freeze_encoder:

            for param in (
                self.esm_model.parameters()
            ):
                param.requires_grad = False

        # =====================================================
        # Projection Head
        # =====================================================

        self.projection_head = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                embedding_dim,
                embedding_dim
            )
        )

        self.output_norm = nn.LayerNorm(
            embedding_dim
        )

    # =========================================================
    # Model Loader
    # =========================================================

    def _load_model(
        self,
        model_name
    ):

        if model_name == "esm2_t6_8M_UR50D":

            return esm.pretrained.esm2_t6_8M_UR50D()

        elif model_name == "esm2_t12_35M_UR50D":

            return esm.pretrained.esm2_t12_35M_UR50D()

        elif model_name == "esm2_t30_150M_UR50D":

            return esm.pretrained.esm2_t30_150M_UR50D()

        elif model_name == "esm2_t33_650M_UR50D":

            return esm.pretrained.esm2_t33_650M_UR50D()

        else:

            raise ValueError(
                f"Unsupported ESM model: {model_name}"
            )

    # =========================================================
    # Tokenization
    # =========================================================

    def tokenize_sequences(
        self,
        sequences
    ):

        processed = []

        for idx, seq in enumerate(
            sequences
        ):

            seq = str(seq)

            seq = seq[: self.max_length]

            processed.append(
                (
                    f"protein_{idx}",
                    seq
                )
            )

        _, _, tokens = (
            self.batch_converter(
                processed
            )
        )

        return tokens

    # =========================================================
    # Mean Pooling
    # =========================================================

    def mean_pooling(
        self,
        representations,
        tokens
    ):

        embeddings = []

        for i in range(
            representations.size(0)
        ):

            valid_tokens = (
                tokens[i]
                != self.alphabet.padding_idx
            )

            valid_repr = (
                representations[i][
                    valid_tokens
                ]
            )

            pooled = valid_repr.mean(
                dim=0
            )

            embeddings.append(
                pooled
            )

        embeddings = torch.stack(
            embeddings,
            dim=0
        )

        return embeddings

    # =========================================================
    # CLS Pooling
    # =========================================================

    def cls_pooling(
        self,
        representations
    ):

        return representations[:, 0, :]

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        protein_sequences
    ):
        """
        Parameters
        ----------
        protein_sequences : List[str]

        Returns
        -------
        embeddings : [B, embedding_dim]
        """

        device = next(
            self.parameters()
        ).device

        tokens = self.tokenize_sequences(
            protein_sequences
        )

        tokens = tokens.to(device)

        with torch.set_grad_enabled(
            not self.freeze_encoder
        ):

            results = self.esm_model(
                tokens,
                repr_layers=[
                    self.esm_model.num_layers
                ],
                return_contacts=False
            )

        representations = (
            results["representations"][
                self.esm_model.num_layers
            ]
        )

        if self.pooling == "mean":

            embeddings = (
                self.mean_pooling(
                    representations,
                    tokens
                )
            )

        elif self.pooling == "cls":

            embeddings = (
                self.cls_pooling(
                    representations
                )
            )

        else:

            raise ValueError(
                f"Unknown pooling: {self.pooling}"
            )

        embeddings = (
            self.projection_head(
                embeddings
            )
        )

        embeddings = (
            self.output_norm(
                embeddings
            )
        )

        return embeddings

    # =========================================================
    # Embedding Extraction
    # =========================================================

    @torch.no_grad()
    def extract_embeddings(
        self,
        protein_sequences
    ):

        self.eval()

        embeddings = self.forward(
            protein_sequences
        )

        return embeddings


# =============================================================
# Factory Function
# =============================================================

def build_esm_encoder(
    config
):

    protein_cfg = config[
        "protein_encoder"
    ]

    model = ESMProteinEncoder(
        model_name=protein_cfg[
            "model_name"
        ],

        embedding_dim=protein_cfg[
            "embedding_dim"
        ],

        freeze_encoder=protein_cfg[
            "freeze_encoder"
        ],

        pooling=protein_cfg[
            "pooling"
        ],

        max_length=protein_cfg[
            "max_length"
        ]
    )

    return model


# =============================================================
# Testing
# =============================================================

if __name__ == "__main__":

    sequences = [

        "MKWVTFISLLLLFSSAYSRGVFRR",

        "MKAILVVLLYTFATANAD"
    ]

    model = ESMProteinEncoder(
        model_name="esm2_t6_8M_UR50D"
    )

    embeddings = model(
        sequences
    )

    print(
        "Protein Embeddings Shape:",
        embeddings.shape
    )