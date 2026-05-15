from .negative_mining import contrastive_loss_curriculum as negative_mining_loss
from .circle          import circle_loss_vectorized      as circle_loss
from .nt_xent         import nt_xent_loss
from .supcon          import supcon_loss
from .fractional      import contrastive_loss_fractional as fractional_loss

# All losses except 'fractional' share the signature:
#     loss_fn(embeddings, group_ids, temperature=0.1, alpha=1.0)
# 'fractional' uses a different training loop (needs data.scores and data.links).
LOSSES = {
    'negative_mining': negative_mining_loss,
    'circle':          circle_loss,
    'nt_xent':         nt_xent_loss,
    'supcon':          supcon_loss,
    'fractional':      fractional_loss,   # different loop — see train.py
}
