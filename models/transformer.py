import math
import torch
import torch.nn as nn


class GraphTransformerLayer(nn.Module):
    """
    Graph transformer layer: multi-head attention restricted to edges in edge_index.

    Uses scaled dot-product attention with per-target-node softmax normalization,
    followed by a feed-forward sublayer. Both sublayers use residual + LayerNorm.
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.3):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.d_k        = hidden_dim // num_heads

        self.q_linear   = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear   = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear   = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.dropout     = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x, edge_index):
        N = x.size(0)
        src, tgt = edge_index

        Q = self.q_linear(x).view(N, self.num_heads, self.d_k)
        K = self.k_linear(x).view(N, self.num_heads, self.d_k)
        V = self.v_linear(x).view(N, self.num_heads, self.d_k)

        scores = (Q[tgt] * K[src]).sum(dim=-1) / math.sqrt(self.d_k)

        scores_max = torch.full((N, self.num_heads), -float('inf'), device=x.device)
        scores_max = scores_max.scatter_reduce(
            0, tgt.unsqueeze(-1).expand(-1, self.num_heads),
            scores, reduce="amax", include_self=True,
        )
        scores_exp = torch.exp(scores - scores_max[tgt])

        scores_sum = torch.zeros(N, self.num_heads, device=x.device).scatter_add(
            0, tgt.unsqueeze(-1).expand(-1, self.num_heads), scores_exp,
        )
        alpha    = self.att_dropout(scores_exp / (scores_sum[tgt] + 1e-16))
        messages = alpha.unsqueeze(-1) * V[src]

        out = torch.zeros(N, self.num_heads, self.d_k, device=x.device).scatter_add(
            0,
            tgt.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.d_k),
            messages,
        )
        out = self.out_linear(out.view(N, self.hidden_dim))
        x   = self.layer_norm1(x + self.dropout(out))
        x   = self.layer_norm2(x + self.dropout(self.ff(x)))
        return x


class Net(nn.Module):
    """
    Graph transformer network using edge-restricted multi-head attention.

    Input:  (N, 16) trackster features
    Output: (N, contrastive_dim) embeddings
    """

    def __init__(self, hidden_dim=128, num_layers=4, dropout=0.3, contrastive_dim=128, num_heads=4, **kwargs):
        super().__init__()

        self.lc_encode = nn.Sequential(
            nn.Linear(16, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )

        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ELU(), nn.Dropout(p=dropout),
            nn.Linear(64, 32),        nn.ELU(), nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim),
        )

    def forward(self, x, edge_index, batch=None):
        x_enc = self.lc_encode(x)
        for layer in self.transformer_layers:
            x_enc = layer(x_enc, edge_index)
        return self.output(x_enc), batch
