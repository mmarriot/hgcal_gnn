"""
Curriculum hard negative mining loss (NT-Xent style).

This was the best-performing loss in experiments. For each anchor:
  - Positive: a randomly sampled trackster from the same CaloParticle group.
  - Negative: a blend of (1-alpha)*random_negative + alpha*hardest_negative,
    where the hardest negative is the most similar trackster from a different group.

alpha is ramped from 0 to 1 during training (curriculum schedule), starting with
purely random negatives and gradually shifting to hard negatives.
"""

import torch
import torch.nn.functional as F


def contrastive_loss_curriculum_both(embeddings, group_ids, temperature=0.1, alpha=1.0):
    norm_emb   = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = norm_emb @ norm_emb.t()
    N   = embeddings.size(0)
    idx = torch.arange(N, device=embeddings.device)

    # --- Positives: random sample from same group ---
    pos_mask = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0))
    pos_mask.fill_diagonal_(False)
    no_valid_pos = (pos_mask.sum(dim=1) == 0)

    rand_vals_pos = torch.rand_like(sim_matrix) * pos_mask.float() - (1 - pos_mask.float())
    rand_pos_idx  = torch.argmax(rand_vals_pos, dim=1)
    rand_pos_sim  = sim_matrix[idx, rand_pos_idx]
    # Fall back to self-similarity for anchors with no positive partner.
    blended_pos   = torch.where(no_valid_pos, sim_matrix[idx, idx], rand_pos_sim)

    # --- Negatives: blend of random and hardest ---
    neg_mask      = (group_ids.unsqueeze(1) != group_ids.unsqueeze(0))
    no_valid_neg  = (neg_mask.sum(dim=1) == 0)

    rand_vals_neg = torch.rand_like(sim_matrix) * neg_mask.float() - (1 - neg_mask.float())
    rand_neg_idx  = torch.argmax(rand_vals_neg, dim=1)
    rand_neg_sim  = sim_matrix[idx, rand_neg_idx]

    hard_neg_sim, _ = sim_matrix.masked_fill(~neg_mask, -float('inf')).max(dim=1)
    hard_neg_sim    = torch.where(no_valid_neg, torch.tensor(-1.0, device=embeddings.device), hard_neg_sim)

    blended_neg = (1 - alpha) * rand_neg_sim + alpha * hard_neg_sim

    # --- Loss ---
    numerator   = torch.exp(blended_pos / temperature)
    denominator = numerator + torch.exp(blended_neg / temperature)
    loss = -torch.log(numerator / denominator)
    loss = loss.masked_fill(no_valid_neg, 0.0)
    return loss.mean()


def contrastive_loss_curriculum(embeddings, group_ids, temperature=0.1, alpha=1.0):
    return contrastive_loss_curriculum_both(embeddings, group_ids, temperature, alpha)
