"""
Fractional (shared-energy) contrastive loss.

Unlike the other losses which assign each trackster to a single CaloParticle,
this loss handles the case where a trackster may be partially matched to several
CaloParticles. Pair similarity is weighted by the amount of energy they share
through common CaloParticle associations.

For a pair (i, j):
    shared_energy = sum over common CaloParticles of min(energy_i, energy_j)

    If shared_energy >= 0.5:  treat as positive, weight = 2*(shared_energy - 0.5)
    If shared_energy <  0.5:  treat as negative, weight = 2*(0.5 - shared_energy)

Anchors with no positive neighbours (all shared_energy < 0.5) are skipped.

NOTE: this loss requires data.scores and data.links tensors (shape (N, 4)) and
uses a different training loop — see train.py.
"""

import torch
import torch.nn.functional as F


def contrastive_loss_fractional(embeddings, groups, scores, temperature=0.1):
    """
    Args:
        embeddings: (N, D) float tensor.
        groups:     (N, 4) int tensor — CaloParticle indices per trackster slot.
        scores:     (N, 4) float tensor — reco-to-sim scores (lower = better match).
        temperature: temperature scaling.

    Returns:
        Scalar mean loss over anchors that have at least one positive.
    """
    device = embeddings.device
    N      = embeddings.size(0)

    norm_emb   = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = norm_emb @ norm_emb.t()

    # Energy proxy: 1 - score (higher = more energy in that CaloParticle slot).
    energy_i = (1.0 - scores).unsqueeze(1)   # (N, 1, 4)
    energy_j = (1.0 - scores).unsqueeze(0)   # (1, N, 4)

    groups_i = groups.unsqueeze(1)            # (N, 1, 4)
    groups_j = groups.unsqueeze(0)            # (1, N, 4)

    # match[i, j, s, t] = 1 if groups_i[i, 0, s] == groups_j[0, j, t]
    match = (groups_i.unsqueeze(-1) == groups_j.unsqueeze(-2)).float()  # (N, N, 4, 4)

    min_energy   = torch.min(energy_i.unsqueeze(-1), energy_j.unsqueeze(-2))  # (N, N, 4, 4)
    shared_energy = (match * min_energy).sum(dim=(-1, -2))  # (N, N)

    pos_mask = (shared_energy >= 0.5)
    neg_mask = ~pos_mask

    pos_weight = torch.zeros_like(shared_energy)
    neg_weight = torch.zeros_like(shared_energy)
    pos_weight[pos_mask] = 2.0 * (shared_energy[pos_mask] - 0.5)
    neg_weight[neg_mask] = 2.0 * (0.5 - shared_energy[neg_mask])
    pos_weight.fill_diagonal_(0)
    neg_weight.fill_diagonal_(0)

    exp_sim    = torch.exp(sim_matrix / temperature)
    numerator  = (pos_weight * exp_sim).sum(dim=1)
    denominator = ((pos_weight + neg_weight) * exp_sim).sum(dim=1)

    anchor_has_pos = (pos_weight.sum(dim=1) > 0)
    if anchor_has_pos.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss_per_anchor = -torch.log(numerator[anchor_has_pos] / denominator[anchor_has_pos])
    return loss_per_anchor.mean()
