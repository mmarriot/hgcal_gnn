# TracksterCL — Contrastive Learning for HGCAL Trackster Linking

Graph neural network training code for linking HGCAL tracksters to their
originating CaloParticles using contrastive representation learning.

---

## Overview

Each event contains a set of *tracksters* — energy clusters reconstructed by
CLUE3D in the CMS High-Granularity Calorimeter. The goal is to learn embeddings
where tracksters that originated from the same CaloParticle (particle shower) are
close together, and tracksters from different particles are far apart.

The model takes a k-NN graph over trackster barycentre coordinates and learns
embeddings via a contrastive loss. At inference time, embeddings can be clustered
to merge tracksters belonging to the same shower.

---

## Directory Structure

```
TracksterCL/
├── data.py          — Dataset class (reads ROOT files via uproot)
├── train.py         — Unified training entry point
├── run.sh           — Shell launcher
│
├── models/          — GNN backbone architectures
│   ├── static_edge.py   — StaticEdgeConv stack (default, fastest)
│   ├── gat.py           — Graph Attention Network
│   ├── transformer.py   — Graph Transformer
│   └── secgat.py        — Interleaved StaticEdge + GAT hybrid
│
└── losses/          — Contrastive loss functions
    ├── negative_mining.py  — Curriculum hard negative mining (BEST)
    ├── circle.py           — Circle Loss
    ├── nt_xent.py          — NT-Xent (InfoNCE, all negatives)
    ├── supcon.py           — Supervised Contrastive Loss
    └── fractional.py       — Fractional shared-energy loss
```

---

## Input Features (16 per trackster)

| Index | Feature | Description |
|-------|---------|-------------|
| 0–2   | barycenter_x/y/z | 3D barycentre position (used for kNN graph) |
| 3     | raw_energy       | Total raw energy |
| 4–5   | barycenter_eta/phi | Pseudorapidity and azimuthal angle |
| 6–8   | EV1/EV2/EV3      | PCA eigenvalues (shower shape) |
| 9–11  | eVector0_x/y/z   | First PCA eigenvector |
| 12–14 | sigmaPCA1/2/3    | PCA spread along each axis |
| 15    | raw_pt           | Transverse momentum |

---

## Neural Network Models

### StaticEdge (`--model static_edge`) — default

Stacked **StaticEdgeConv** layers. For each edge (i→j) on a fixed k-NN graph,
computes `f([x_i | x_j − x_i])` and mean-pools messages back to nodes. Simple,
fast, and works well in practice.

```
lc_encode → [StaticEdgeConv + residual] × num_layers → output MLP → embedding
```

Key parameters: `hidden_dim`, `num_layers`.

---

### GAT (`--model gat`)

Stacked **multi-head Graph Attention** layers. Each layer computes per-edge
attention weights using separate learnable vectors for source and target nodes,
then aggregates weighted source features. Concatenates heads by default.

```
lc_encode → [GATLayer + residual] × num_layers → output MLP → embedding
```

Key parameters: `hidden_dim`, `num_layers`, `heads` (must divide `hidden_dim`).

---

### Graph Transformer (`--model transformer`)

Scaled dot-product multi-head attention restricted to edges in the k-NN graph
(not full self-attention). Each layer has: attention sublayer + FFN sublayer,
both with residual + LayerNorm.

```
lc_encode → [GraphTransformerLayer] × num_layers → output MLP → embedding
```

Key parameters: `hidden_dim`, `num_layers`, `num_heads`.

---

### SECGAT (`--model secgat`)

Fixed 5-layer hybrid: **StaticEdge → GAT → StaticEdge → GAT → StaticEdge**.
Combines local neighbourhood aggregation (StaticEdge) with attention-weighted
aggregation (GAT).

Key parameters: `hidden_dim`, `heads`.

---

## Loss Functions

All losses operate on L2-normalised embeddings (cosine similarity). They share
the interface `loss_fn(embeddings, group_ids, temperature, alpha)` except
`fractional` which needs a different training loop.

---

### Negative Mining (`--loss negative_mining`) — BEST

**NT-Xent with curriculum hard negative mining.** For each anchor:

- **Positive**: randomly sampled trackster from the same CaloParticle group.
- **Negative**: blend of a random negative and the *hardest* negative (most
  similar embedding from a different group).

The blend is controlled by `alpha`:

```
blended_neg = (1 − alpha) × random_neg + alpha × hardest_neg
```

`alpha` is scheduled over training (curriculum):

| Epochs | alpha | Effect |
|--------|-------|--------|
| 0–74   | 0.0   | Random negatives — easy, stable early training |
| 75–149 | 0→1   | Linear ramp — gradually harder |
| 150+   | 1.0   | Hard negatives only |

Loss per anchor:
```
L_i = −log [ exp(pos / τ) / (exp(pos / τ) + exp(blended_neg / τ)) ]
```

---

### Circle Loss (`--loss circle`)

**Re-weighted pair similarity loss** (Sun et al., CVPR 2020). Assigns larger
gradients to positive pairs that are too far apart and to negative pairs that
are too close, based on how far each similarity is from a margin target.

```
L = log(1 + Σ_p w_p·exp(−γ(s_p − α_p)) × Σ_n w_n·exp(γ(s_n − α_n)))
```

Default: `margin=0.40`, `gamma=32.0`. Does not use `alpha` scheduling.

Reference: https://arxiv.org/abs/2002.10857

---

### NT-Xent All Negatives (`--loss nt_xent`)

Standard **InfoNCE** loss: sums exponentiated similarities over *all* negatives
in the denominator rather than a single sampled one.

```
L_i = −log [ exp(pos / τ) / (exp(pos / τ) + Σ_{j: group≠i} exp(sim(i,j) / τ)) ]
```

Positive is sampled randomly from the same group. More stable than single-negative
but higher memory cost (O(N²) similarity matrix must be fully realised).

---

### Supervised Contrastive (`--loss supcon`)

**SupCon** (Khosla et al., NeurIPS 2020). All tracksters in the same group are
positives for each anchor — not just one. Loss averages over all positive pairs:

```
L_i = −(1/|P_i|) Σ_{p∈P_i} log [ exp(s_{ip} / τ) / Σ_{j≠i} exp(s_{ij} / τ) ]
```

Better than single-positive when groups are large (many tracksters per shower).

Reference: https://arxiv.org/abs/2004.11362

---

### Fractional (`--loss fractional`)

Handles **partial shower overlap**: a trackster may be matched to several
CaloParticles with different scores. Instead of a hard group assignment, pair
affinity is computed as:

```
shared_energy(i, j) = Σ_{common CaloParticles k} min(energy_i_k, energy_j_k)
```

where `energy_x_k = 1 − score_x_k`.

- `shared_energy ≥ 0.5` → positive pair, weight `= 2 × (shared_energy − 0.5)`
- `shared_energy < 0.5` → negative pair, weight `= 2 × (0.5 − shared_energy)`

Requires `data.scores` and `data.links` tensors (both shape `(N, 4)`).
Uses a separate training loop in `train.py`.

---

## Training

### Quick start (default: StaticEdge + NegativeMining)

```bash
bash run.sh
```

### Custom run

```bash
python train.py \
    --model           static_edge \       # static_edge | gat | transformer | secgat
    --loss            negative_mining \   # negative_mining | circle | nt_xent | supcon | fractional
    --hidden_dim      128 \
    --num_layers      3 \
    --contrastive_dim 128 \
    --lr              5e-4 \
    --epochs          220 \
    --batch_size      64 \
    --k_value         24 \
    --output_dir      runs/my_experiment/
```

### Outputs

Each run saves to `--output_dir`:

| File | Content |
|------|---------|
| `best_model.pt`   | Model weights at best validation loss |
| `final_model.pt`  | Model weights after last epoch |
| `epoch-N.pt`      | Checkpoint (model + optimiser + scheduler) |
| `loss_curves.csv` | Train and validation loss per epoch |

---

## Data Format

The `CCV1` dataset reads `.root` files from a directory. Pass the directory path
as the `root` argument — it will look for all `*.root` files inside `raw/`.

Events are filtered to require:
- ≥ 2 CaloParticles per event
- ≥ 2 trackster associations per event

Group IDs (`data.assoc`) are assigned by taking the CaloParticle with the lowest
reco-to-sim score as the "best match" for each trackster.
