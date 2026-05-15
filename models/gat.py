import torch.nn as nn
from .layers import CustomGATLayer


class Net(nn.Module):
    """
    Graph network using stacked multi-head GAT layers with residual connections.

    hidden_dim must be divisible by heads (per-head dim = hidden_dim // heads).

    Input:  (N, 16) trackster features
    Output: (N, contrastive_dim) embeddings
    """

    def __init__(self, hidden_dim=128, num_layers=4, dropout=0.3, contrastive_dim=128, heads=4, **kwargs):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        per_head_dim = hidden_dim // heads

        self.lc_encode = nn.Sequential(
            nn.Linear(16, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )

        self.convs = nn.ModuleList([
            CustomGATLayer(hidden_dim, per_head_dim, heads=heads, concat=True, dropout=dropout)
            for _ in range(num_layers)
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
