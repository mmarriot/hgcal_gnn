import torch.nn as nn
from .layers import CustomStaticEdgeConv


class Net(nn.Module):
    """
    Graph network using stacked StaticEdgeConv layers with residual connections.

    Input:  (N, 16) trackster features
    Output: (N, contrastive_dim) embeddings
    """

    def __init__(self, hidden_dim=128, num_layers=3, dropout=0.3, contrastive_dim=128, **kwargs):
        super().__init__()

        self.lc_encode = nn.Sequential(
            nn.Linear(16, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )

        self.convs = nn.ModuleList([
            CustomStaticEdgeConv(nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(p=dropout),
            ))
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
