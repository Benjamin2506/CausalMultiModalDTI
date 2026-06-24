"""
=============================================================
Graph Transformer Encoder

Causal Multi-Modal DTI Framework

Architecture:
    Node Features (9)
            ↓
    Linear Projection
            ↓
    TransformerConv Layers
            ↓
    Residual Connections
            ↓
    LayerNorm
            ↓
    Global Mean Pool
    Global Max Pool
            ↓
    Projection Head
            ↓
    Drug Embedding (256)

Author:
Roshan Kotkondawar
Kunal Nicose
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    TransformerConv,
    global_mean_pool,
    global_max_pool
)


class GraphTransformerBlock(nn.Module):
    """
    Single Graph Transformer Block
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        edge_dim,
        dropout=0.2
    ):
        super().__init__()

        self.conv = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            beta=True
        )

        self.norm1 = nn.LayerNorm(
            hidden_dim
        )

        self.norm2 = nn.LayerNorm(
            hidden_dim
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim * 4
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim * 4,
                hidden_dim
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr
    ):

        residual = x

        x = self.conv(
            x,
            edge_index,
            edge_attr
        )

        x = self.dropout(x)

        x = self.norm1(
            x + residual
        )

        residual = x

        x = self.ffn(x)

        x = self.dropout(x)

        x = self.norm2(
            x + residual
        )

        return x


class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer Drug Encoder
    """

    def __init__(
        self,
        input_dim=9,
        hidden_dim=256,
        output_dim=256,
        num_layers=6,
        num_heads=8,
        edge_dim=7,
        dropout=0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # ==========================================
        # Input Projection
        # ==========================================

        self.node_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.GELU()
        )

        # ==========================================
        # Edge Projection
        # ==========================================

        self.edge_projection = nn.Sequential(
            nn.Linear(
                edge_dim,
                32
            ),
            nn.GELU(),
            nn.Linear(
                32,
                edge_dim
            )
        )

        # ==========================================
        # Graph Transformer Layers
        # ==========================================

        self.layers = nn.ModuleList(
            [
                GraphTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    edge_dim=edge_dim,
                    dropout=dropout
                )
                for _ in range(
                    num_layers
                )
            ]
        )

        # ==========================================
        # Pooling Projection
        # ==========================================

        pooled_dim = hidden_dim * 2

        self.projection_head = nn.Sequential(
            nn.Linear(
                pooled_dim,
                hidden_dim
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                output_dim
            )
        )

        self.output_norm = nn.LayerNorm(
            output_dim
        )

    def forward(
        self,
        data
    ):
        """
        Parameters
        ----------
        data : PyG Batch

        Returns
        -------
        drug_embedding : [B, output_dim]
        """

        x = data.x.float()

        edge_index = data.edge_index

        edge_attr = data.edge_attr.float()

        batch = data.batch

        # ==========================================
        # Input Projection
        # ==========================================

        x = self.node_projection(
            x
        )

        edge_attr = (
            self.edge_projection(
                edge_attr
            )
        )

        # ==========================================
        # Transformer Layers
        # ==========================================

        for layer in self.layers:

            x = layer(
                x,
                edge_index,
                edge_attr
            )

        # ==========================================
        # Global Pooling
        # ==========================================

        mean_pool = global_mean_pool(
            x,
            batch
        )

        max_pool = global_max_pool(
            x,
            batch
        )

        pooled = torch.cat(
            [
                mean_pool,
                max_pool
            ],
            dim=1
        )

        # ==========================================
        # Projection Head
        # ==========================================

        drug_embedding = (
            self.projection_head(
                pooled
            )
        )

        drug_embedding = (
            self.output_norm(
                drug_embedding
            )
        )

        return drug_embedding

    @torch.no_grad()
    def extract_embeddings(
        self,
        data
    ):
        """
        Inference utility
        """

        self.eval()

        embedding = self.forward(
            data
        )

        return embedding


class GraphTransformerClassifier(nn.Module):
    """
    Debug/Standalone Classifier

    Useful for validating
    graph encoder independently.
    """

    def __init__(
        self,
        encoder,
        num_classes=1
    ):
        super().__init__()

        self.encoder = encoder

        self.classifier = nn.Sequential(
            nn.Linear(
                encoder.output_dim,
                128
            ),
            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(
        self,
        data
    ):

        embedding = self.encoder(
            data
        )

        logits = self.classifier(
            embedding
        )

        return logits


def build_graph_transformer(
    config
):
    """
    Build Graph Transformer
    from YAML config
    """

    graph_cfg = config[
        "graph_transformer"
    ]

    model = GraphTransformerEncoder(
        input_dim=graph_cfg[
            "input_dim"
        ],

        hidden_dim=graph_cfg[
            "hidden_dim"
        ],

        output_dim=graph_cfg[
            "output_dim"
        ],

        num_layers=graph_cfg[
            "num_layers"
        ],

        num_heads=graph_cfg[
            "num_heads"
        ],

        edge_dim=7,

        dropout=graph_cfg[
            "dropout"
        ]
    )

    return model


if __name__ == "__main__":

    print(
        "\nGraph Transformer Module Loaded Successfully"
    )

    model = GraphTransformerEncoder()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {total_params:,}"
    )