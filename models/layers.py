import torch
import torch.nn as nn


class CustomStaticEdgeConv(nn.Module):
    """
    Edge convolution on a fixed (static) graph.

    For each edge (i -> j): computes f([x_i | x_j - x_i]) and aggregates
    back to nodes by mean-pooling over incoming messages.
    """

    def __init__(self, nn_module):
        super().__init__()
        self.nn_module = nn_module

    def forward(self, x, edge_index):
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col] - x[row]], dim=-1)
        edge_features = self.nn_module(edge_features)

        num_nodes = x.size(0)
        node_features = torch.zeros(num_nodes, edge_features.size(-1), device=x.device)
        node_features.index_add_(0, row, edge_features)

        counts = torch.bincount(row, minlength=num_nodes).clamp(min=1).view(-1, 1)
        return node_features / counts


class CustomGATLayer(nn.Module):
    """
    Multi-head graph attention layer with separate source/target attention vectors.

    Supports both concatenation (concat=True) and averaging (concat=False) of heads.
    """

    def __init__(self, in_dim, out_dim, heads=1, concat=True, dropout=0.3, alpha=0.4):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.heads   = heads
        self.concat  = concat

        self.W     = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(heads, out_dim))
        self.a_tgt = nn.Parameter(torch.zeros(heads, out_dim))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_tgt.data, gain=1.414)

        self.leakyrelu  = nn.LeakyReLU(alpha)
        self.dropout    = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(heads * out_dim if concat else out_dim)

    def forward(self, x, edge_index):
        src, tgt = edge_index
        N = x.size(0)

        h = self.W(x).view(N, self.heads, self.out_dim)
        h_src = h[src]
        h_tgt = h[tgt]

        e = self.leakyrelu((h_src * self.a_src).sum(-1) + (h_tgt * self.a_tgt).sum(-1))
        e = e - e.max(dim=0, keepdim=True)[0]
        alpha = torch.exp(e)

        alpha_sum = torch.zeros(N, self.heads, device=x.device).scatter_add_(
            0, tgt.unsqueeze(-1).expand(-1, self.heads), alpha
        ) + 1e-16
        alpha = self.dropout(alpha / alpha_sum[tgt])

        h_prime = h_src * alpha.unsqueeze(-1)
        out = torch.zeros(N, self.heads, self.out_dim, device=x.device).scatter_add_(
            0,
            tgt.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, self.out_dim),
            h_prime,
        )

        out = out.view(N, self.heads * self.out_dim) if self.concat else out.mean(dim=1)
        return self.batch_norm(out)
