"""
NT-Xent loss summing over all negatives (InfoNCE-style).

Unlike the negative_mining loss which compares the positive against a single
(blended) negative, this sums the exponentiated similarities of *all* negatives
in the denominator — the standard InfoNCE formulation.

A random positive is sampled from the same group for each anchor (same strategy
as the negative_mining loss) so no explicit positive index is required.
"""

import torch
import torch.nn.functional as F


def nt_xent_loss(embeddings, group_ids, temperature=0.1, alpha=None):
    """
    Args:
        embeddings: (N, D) float tensor.
        group_ids:  (N,)   int tensor.
        temperature: temperature scaling.
        alpha: unused — present for API compatibility.

    Returns:
        Scalar mean loss.
    """
    norm_emb   = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = norm_emb @ norm_emb.t()
    N   = embeddings.size(0)
    idx = torch.arange(N, device=embeddings.device)

    # Random positive from same group (excluding self).
    pos_mask = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0))
    pos_mask.fill_diagonal_(False)
    no_valid_pos = (pos_mask.sum(dim=1) == 0)

    rand_vals = torch.rand_like(sim_matrix) * pos_mask.float() - (1 - pos_mask.float())
    pos_indices = torch.argmax(rand_vals, dim=1)
    pos_indices[no_valid_pos] = idx[no_valid_pos]  # fallback to self

    pos_sim = sim_matrix[idx, pos_indices]

    # All negatives.
    neg_mask = (group_ids.unsqueeze(1) != group_ids.unsqueeze(0))
    neg_mask.fill_diagonal_(False)

    exp_sim     = torch.exp(sim_matrix / temperature)
    pos_exp     = torch.exp(pos_sim / temperature)
    neg_exp_sum = (exp_sim * neg_mask.float()).sum(dim=1)

    loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-16))
    loss = loss.masked_fill(no_valid_pos, 0.0)
    return loss.mean()
