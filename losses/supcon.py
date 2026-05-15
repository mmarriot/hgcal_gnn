"""
Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

All tracksters that share the same CaloParticle group ID are treated as positives
for each anchor. The loss averages over all positive pairs, making it more stable
for groups with many members than a single-positive formulation.

Reference: https://arxiv.org/abs/2004.11362
"""

import torch
import torch.nn.functional as F


def supcon_loss(embeddings, group_ids, temperature=0.1, alpha=None):
    """
    Args:
        embeddings: (N, D) float tensor.
        group_ids:  (N,)   int tensor.
        temperature: temperature scaling.
        alpha: unused — present for API compatibility.

    Returns:
        Scalar mean loss over anchors that have at least one positive.
    """
    norm_emb   = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.div(torch.matmul(norm_emb, norm_emb.t()), temperature)

    N         = embeddings.size(0)
    self_mask = torch.eye(N, dtype=torch.bool, device=embeddings.device)

    # Mask out self-similarity from the denominator.
    sim_masked    = sim_matrix.masked_fill(self_mask, -float('inf'))
    positive_mask = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0)) & ~self_mask

    exp_sim      = torch.exp(sim_masked)
    denominator  = exp_sim.sum(dim=1)
    pos_exp_sum  = (exp_sim * positive_mask.float()).sum(dim=1)
    pos_counts   = positive_mask.sum(dim=1).float()

    loss  = torch.zeros(N, device=embeddings.device)
    valid = pos_counts > 0
    loss[valid] = -(1.0 / pos_counts[valid]) * torch.log(pos_exp_sum[valid] / denominator[valid])
    return loss.mean()
