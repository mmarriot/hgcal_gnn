"""
Circle Loss (Sun et al., CVPR 2020).

Re-parameterises similarity optimisation as minimising a softplus of the product
of positive and negative exponential sums. Weights each positive/negative pair
by how far its similarity is from the target margin, giving larger gradients to
pairs that are furthest from convergence.

Key hyperparameters:
    margin (m): target similarity gap between positives and negatives (default 0.40).
    gamma:      scaling factor controlling the sharpness of the loss (default 32.0).

Reference: https://arxiv.org/abs/2002.10857
"""

import torch
import torch.nn.functional as F


def circle_loss_vectorized(embeddings, group_ids, temperature=None, alpha=None,
                           margin=0.40, gamma=32.0):
    """
    Args:
        embeddings: (N, D) float tensor.
        group_ids:  (N,)   int tensor.
        temperature, alpha: unused — present for API compatibility with other losses.
        margin: similarity margin m.
        gamma:  scale factor.

    Returns:
        Scalar mean loss over valid anchors.
    """
    norm_emb   = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(norm_emb, norm_emb.T)

    N      = sim_matrix.size(0)
    device = sim_matrix.device
    idx    = torch.arange(N, device=device)

    pos_mask = (group_ids.unsqueeze(0) == group_ids.unsqueeze(1))
    pos_mask[idx, idx] = False
    neg_mask = ~pos_mask.clone()
    neg_mask[idx, idx] = False

    alpha_p = 1.0 - margin   # 0.60 at default margin=0.40
    alpha_n = margin          # 0.40

    # pos_mask / neg_mask multiplied at the end ensures non-relevant pairs
    # contribute exactly 0 to the sums.
    w_p = (alpha_p - sim_matrix).clamp_min(0.0) * pos_mask.float()
    w_n = (sim_matrix - alpha_n).clamp_min(0.0) * neg_mask.float()

    exp_pos = w_p * torch.exp(-gamma * (sim_matrix - alpha_p))
    exp_neg = w_n * torch.exp( gamma * (sim_matrix - alpha_n))

    sum_pos = exp_pos.sum(dim=1)
    sum_neg = exp_neg.sum(dim=1)

    valid_mask = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)
    losses = torch.zeros(N, device=device)
    losses[valid_mask] = torch.log1p(sum_pos[valid_mask] * sum_neg[valid_mask])
    return losses.mean()
