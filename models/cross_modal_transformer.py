"""
=============================================================
Cross-Modal Transformer Fusion Module

Causal Multi-Modal DTI Framework

Purpose:
--------
Fuse drug and protein embeddings using
cross-attention transformer architecture.

Inputs:
-------
Drug Embedding    : [B, 256]
Protein Embedding : [B, 1280]

Output:
-------
Fused Embedding   : [B, 512]

Architecture:
-------------
1. Linear Projection (Drug → 512)
2. Linear Projection (Protein → 512)
3. Cross-Attention (Drug ↔ Protein)
4. Self-Attention Refinement
5. Residual + LayerNorm
6. Final Fusion Representation

Author:
Roshan Kotkondawar
Kunal Nicose
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Cross Attention Block
# ============================================================

class CrossAttentionBlock(nn.Module):
    """
    Multi-head Cross Attention
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.2
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        query: [B, 1, D]
        key_value: [B, 1, D]
        """

        attn_output, _ = self.attn(
            query,
            key_value,
            key_value
        )

        x = self.norm1(query + self.dropout(attn_output))

        ff = self.ffn(x)

        x = self.norm2(x + self.dropout(ff))

        return x


# ============================================================
# Cross Modal Transformer
# ============================================================

class CrossModalTransformer(nn.Module):
    """
    Drug-Protein Fusion Network
    """

    def __init__(
        self,
        drug_dim=256,
        protein_dim=1280,
        hidden_dim=512,
        num_heads=8,
        dropout=0.2
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # =====================================================
        # Projection Layers
        # =====================================================

        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # =====================================================
        # Cross Attention Blocks
        # =====================================================

        self.drug_to_protein = CrossAttentionBlock(
            hidden_dim,
            num_heads,
            dropout
        )

        self.protein_to_drug = CrossAttentionBlock(
            hidden_dim,
            num_heads,
            dropout
        )

        # =====================================================
        # Fusion Layer
        # =====================================================

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    # =========================================================
    # Forward
    # =========================================================

    def forward(self, drug_embedding, protein_embedding):
        """
        Parameters
        ----------
        drug_embedding : [B, 256]
        protein_embedding : [B, 1280]

        Returns
        -------
        fused_embedding : [B, 512]
        """

        # -----------------------------------------------------
        # Add sequence dimension for attention
        # -----------------------------------------------------

        d = self.drug_proj(drug_embedding).unsqueeze(1)
        p = self.protein_proj(protein_embedding).unsqueeze(1)

        # -----------------------------------------------------
        # Cross Attention
        # -----------------------------------------------------

        d_attn = self.drug_to_protein(d, p)
        p_attn = self.protein_to_drug(p, d)

        # -----------------------------------------------------
        # Pool back to vector
        # -----------------------------------------------------

        d_attn = d_attn.squeeze(1)
        p_attn = p_attn.squeeze(1)

        # -----------------------------------------------------
        # Fusion
        # -----------------------------------------------------

        fused = torch.cat([d_attn, p_attn], dim=-1)

        fused = self.fusion_mlp(fused)

        fused = self.norm(fused)

        fused = self.dropout(fused)

        return fused


# ============================================================
# Factory
# ============================================================

def build_cross_modal_transformer(config):

    fusion_cfg = config["cross_modal_transformer"]

    model = CrossModalTransformer(
        drug_dim=256,
        protein_dim=1280,
        hidden_dim=fusion_cfg["hidden_dim"],
        num_heads=fusion_cfg["num_heads"],
        dropout=fusion_cfg["dropout"]
    )

    return model


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    B = 4

    drug = torch.randn(B, 256)

    protein = torch.randn(B, 1280)

    model = CrossModalTransformer()

    out = model(drug, protein)

    print("Fused shape:", out.shape)