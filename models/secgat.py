import torch.nn as nn
from .layers import CustomStaticEdgeConv, CustomGATLayer


class Net(nn.Module):
    """
    Hybrid graph network interleaving StaticEdgeConv and GAT layers:
        StaticEdge -> GAT -> StaticEdge -> GAT -> StaticEdge

    Combines the local neighbourhood aggregation of StaticEdgeConv with the
    attention-weighted aggregation of GAT. All layers use residual connections.

    Input:  (N, 16) trackster features
    Output: (N, contrastive_dim) embeddings
    """

    def __init__(self, hidden_dim=128, dropout=0.3, contrastive_dim=128, heads=4, **kwargs):
        super().__init__()

        self.lc_encode = nn.Sequential(
            nn.Linear(16, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )

        def static_layer():
            return CustomStaticEdgeConv(nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(p=dropout),
            ))

        def gat_layer():
            return CustomGATLayer(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)

        self.convs = nn.ModuleList([
            static_layer(), gat_layer(),
            static_layer(), gat_layer(),
            static_layer(),
        ])

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ELU(), nn.Dropout(p=dropout),
            nn.Linear(64, 32),        nn.ELU(), nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim),
        )

    def forward(self, x, edge_index, batch):
        feats = self.lc_encode(x)
        for conv in self.convs:
            feats = conv(feats, edge_index) + feats
        return self.output(feats), batch
