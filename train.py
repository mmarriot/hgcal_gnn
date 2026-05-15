#!/usr/bin/env python
"""
Unified training entry point for trackster-linking contrastive learning.

Select a model architecture and loss function via --model and --loss.
All non-fractional losses share the same training loop; the fractional loss
uses a separate loop that reads data.scores / data.links instead of data.assoc.

Alpha curriculum (negative_mining only):
    epochs 0–25:    alpha = 0.0   random negatives only
    epochs 25–50:  alpha 0 → 1   linear ramp
    epochs 50+:    alpha = 1.0   hardest negatives only

Example:
    python train.py --model static_edge --loss negative_mining
    python train.py --model gat         --loss circle --hidden_dim 64 --heads 4
"""

import argparse
import os

import numpy as np
import torch
import pandas as pd
from torch_geometric.data import DataLoader
from torch_geometric.nn import knn_graph
from tqdm import tqdm

from data   import CCV1
from models import MODELS
from losses import LOSSES
from losses.fractional import contrastive_loss_fractional


# Helpers
def _assoc_tensor(data):
    """Coerce data.assoc (list or tensor) to a flat int64 tensor on the right device."""
    if isinstance(data.assoc, list):
        if isinstance(data.assoc[0], list):
            return torch.cat([
                torch.tensor(a, dtype=torch.int64, device=data.x.device)
                for a in data.assoc
            ])
        return torch.tensor(data.assoc, device=data.x.device)
    return data.assoc


def _alpha_schedule(epoch):
    """Curriculum ramp for negative_mining. Returns 0 outside that loss."""
    if epoch < 25:
        return 0.0
    if epoch < 50:
        return (epoch - 50) / 50.0
    return 1.0


def _event_slices(batch_tensor):
    """Return (start, count) pairs for each event in a batched loader output."""
    _, counts = np.unique(batch_tensor.cpu().numpy(), return_counts=True)
    starts = np.concatenate([[0], counts.cumsum()[:-1]])
    return list(zip(starts, counts))


# Standard training loop  (negative_mining / circle / nt_xent / supcon)
def train_epoch(loader, model, loss_fn, optimizer, device, k, alpha):
    """
    Args:
        alpha: blending parameter passed to loss_fn.
               Only negative_mining uses it; all other losses accept and ignore it.
    """
    model.train()
    total_loss = torch.zeros(1, device=device)

    for data in tqdm(loader, desc="Train"):
        data = data.to(device)
        optimizer.zero_grad()

        assoc      = _assoc_tensor(data)
        edge_index = knn_graph(data.x[:, :3], k=k, batch=data.x_batch)
        embeddings, _ = model(data.x, edge_index, data.x_batch)

        slices     = _event_slices(data.x_batch)
        event_loss = torch.zeros(1, device=device)
        for start, c in slices:
            event_loss += loss_fn(
                embeddings[start:start+c], assoc[start:start+c],
                temperature=0.1, alpha=alpha,
            )

        loss = event_loss / len(slices)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()

    return (total_loss / len(loader.dataset)).item()


@torch.no_grad()
def eval_epoch(loader, model, loss_fn, device, k, alpha):
    model.eval()
    total_loss = torch.zeros(1, device=device)

    for data in tqdm(loader, desc="Val  "):
        data = data.to(device)
        assoc      = _assoc_tensor(data)
        edge_index = knn_graph(data.x[:, :3], k=k, batch=data.x_batch)
        embeddings, _ = model(data.x, edge_index, data.x_batch)

        slices     = _event_slices(data.x_batch)
        event_loss = torch.zeros(1, device=device)
        for start, c in slices:
            event_loss += loss_fn(
                embeddings[start:start+c], assoc[start:start+c],
                temperature=0.1, alpha=alpha,
            )
        total_loss += event_loss / len(slices)

    return (total_loss / len(loader.dataset)).item()


# Fractional training loop  (needs data.scores and data.links)
def train_epoch_fractional(loader, model, optimizer, device, k):
    model.train()
    total_loss, n = 0.0, 0

    for data in tqdm(loader, desc="Train"):
        data = data.to(device)
        optimizer.zero_grad()

        edge_index = knn_graph(data.x[:, :3], k=k, batch=data.x_batch)
        embeddings, _ = model(data.x, edge_index, data.x_batch)

        slices     = _event_slices(data.x_batch)
        event_loss = torch.zeros(1, device=device)
        for start, c in slices:
            event_loss += contrastive_loss_fractional(
                embeddings[start:start+c],
                data.links[start:start+c],
                data.scores[start:start+c],
                temperature=0.1,
            )

        loss = event_loss / len(slices)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * embeddings.size(0)
        n += embeddings.size(0)

    return total_loss / n


@torch.no_grad()
def eval_epoch_fractional(loader, model, device, k):
    model.eval()
    total_loss, n = 0.0, 0

    for data in tqdm(loader, desc="Val  "):
        data = data.to(device)
        edge_index = knn_graph(data.x[:, :3], k=k, batch=data.x_batch)
        embeddings, _ = model(data.x, edge_index, data.x_batch)

        slices     = _event_slices(data.x_batch)
        event_loss = torch.zeros(1, device=device)
        for start, c in slices:
            event_loss += contrastive_loss_fractional(
                embeddings[start:start+c],
                data.links[start:start+c],
                data.scores[start:start+c],
                temperature=0.1,
            )
        total_loss += (event_loss / len(slices)).item()
        n += embeddings.size(0)

    return total_loss / n


# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Architecture
    parser.add_argument('--model',           type=str,   default='static_edge',
                        choices=list(MODELS.keys()))
    parser.add_argument('--loss',            type=str,   default='negative_mining',
                        choices=list(LOSSES.keys()))
    parser.add_argument('--hidden_dim',      type=int,   default=128)
    parser.add_argument('--num_layers',      type=int,   default=3)
    parser.add_argument('--contrastive_dim', type=int,   default=128)
    parser.add_argument('--heads',           type=int,   default=4,
                        help="Attention heads (gat / secgat / transformer only)")
    parser.add_argument('--dropout',         type=float, default=0.3)
    # Optimiser
    parser.add_argument('--lr',         type=float, default=5e-4)
    parser.add_argument('--epochs',     type=int,   default=220)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--k_value',    type=int,   default=24)
    parser.add_argument('--patience',   type=int,   default=300)
    # Data
    parser.add_argument('--train_path',       type=str, default='/vols/cms/mm1221/Data/mix/train/')
    parser.add_argument('--val_path',         type=str, default='/vols/cms/mm1221/Data/mix/val/')
    parser.add_argument('--max_train_events', type=int, default=40000)
    parser.add_argument('--max_val_events',   type=int, default=15000)
    # Output
    parser.add_argument('--output_dir', type=str,
                        default='/vols/cms/mm1221/hgcal/Mixed/Track/TracksterCL/runs/')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Model: {args.model}  |  Loss: {args.loss}")

    print("Loading data...")
    data_train = CCV1(args.train_path, max_events=args.max_train_events, inp='train')
    data_val   = CCV1(args.val_path,   max_events=args.max_val_events,   inp='val')

    train_loader = DataLoader(data_train, batch_size=args.batch_size,
                              shuffle=True,  follow_batch=['x'])
    val_loader   = DataLoader(data_val,   batch_size=args.batch_size,
                              shuffle=False, follow_batch=['x'])

    print("Building model...")
    model = MODELS[args.model](
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        contrastive_dim=args.contrastive_dim,
        heads=args.heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    loss_fn        = LOSSES[args.loss]
    use_fractional = (args.loss == 'fractional')
    use_curriculum = (args.loss == 'negative_mining')

    best_val_loss            = float('inf')
    no_improve               = 0
    train_losses, val_losses = [], []

    print("Training...")
    for epoch in range(args.epochs):
        # Alpha is only meaningful for negative_mining; all other losses receive 0.
        alpha = _alpha_schedule(epoch) if use_curriculum else 0.0
        print(f"Epoch {epoch+1}/{args.epochs}"
              + (f"  alpha={alpha:.2f}" if use_curriculum else ""))

        if use_fractional:
            tr_loss = train_epoch_fractional(train_loader, model, optimizer, device, args.k_value)
            va_loss = eval_epoch_fractional(val_loader,   model,           device, args.k_value)
        else:
            tr_loss = train_epoch(train_loader, model, loss_fn, optimizer, device, args.k_value, alpha)
            va_loss = eval_epoch( val_loader,   model, loss_fn,            device, args.k_value, alpha)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        scheduler.step()

        print(f"  train={tr_loss:.6f}  val={va_loss:.6f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            no_improve    = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  -> new best model saved")
        else:
            no_improve += 1

        torch.save(
            {'model': model.state_dict(), 'opt': optimizer.state_dict(),
             'lr': scheduler.state_dict()},
            os.path.join(args.output_dir, f'epoch-{epoch+1}.pt'),
        )

        if no_improve >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    pd.DataFrame({
        'epoch':      list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_loss':   val_losses,
    }).to_csv(os.path.join(args.output_dir, 'loss_curves.csv'), index=False)

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))
    print(f"Done. Best val loss: {best_val_loss:.6f}")
