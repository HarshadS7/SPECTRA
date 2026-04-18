"""
Improved Temporal GNN Trainer
============================
- Trains on 5 years of data (more samples, more crises)
- Monitors train/val loss to detect overfitting
- Saves best model based on validation loss
- Uses early stopping with patience
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from data.loader import TimeSeriesLoader
from models.temporal_gnn import TemporalGNN


def build_multi_horizon_targets(y_all: torch.Tensor, num_horizons: int) -> torch.Tensor:
    W, N = y_all.shape
    usable = W - num_horizons + 1
    targets = torch.zeros(usable, num_horizons, N)
    for i in range(usable):
        for k in range(num_horizons):
            targets[i, k, :] = y_all[i + k]
    return targets


def horizon_weighted_loss(preds, targets, criterion, num_horizons):
    # Bell-curve weights: T+2/T+3 are primary signal;
    # T+4/T+5 lifted to equal T+1 so longer horizons aren't under-trained.
    weights = torch.tensor(
        [0.8, 1.0, 1.0, 0.9, 0.9][:num_horizons], device=preds.device
    )
    loss = 0.0
    for k in range(num_horizons):
        loss += weights[k] * criterion(preds[:, k], targets[:, k])
    return loss / weights.sum()


def train(
    period: str = "5y",
    epochs: int = 300,
    lr: float = 0.001,
    hidden_dim: int = 64,
    num_horizons: int = 5,
    batch_size: int = 64,
    val_split: float = 0.2,
    patience: int = 40,
    save_path: str | None = None,
):
    if save_path is None:
        save_path = os.path.join(BACKEND_DIR, "temporal_gnn_v2.pth")

    print(f"\n{'=' * 70}")
    print(f"IMPROVED TRAINING - More Data, Better Regularization")
    print(f"{'=' * 70}")
    print(f"Started: {datetime.now().isoformat()}")
    print(
        f"Period: {period}, Epochs: {epochs}, Batch: {batch_size}, Hidden: {hidden_dim}"
    )
    print(f"{'=' * 70}\n")

    # 1. Load Data
    loader = TimeSeriesLoader(period=period).load()
    x_all, y_all, edges_all = loader.get_windows()

    # 2. Build multi-horizon targets
    y_multi = build_multi_horizon_targets(y_all, num_horizons)
    usable = y_multi.shape[0]
    x_all = x_all[:usable]
    edges_all = edges_all[:usable]

    # 3. Time-based split (with gap to prevent evaluation leakage)
    split = int(usable * (1 - val_split))
    train_end = max(1, split - num_horizons)
    train_x, train_y = x_all[:train_end], y_multi[:train_end]
    val_x, val_y = x_all[split:], y_multi[split:]
    train_edges, val_edges = edges_all[:train_end], edges_all[split:]

    print(f"[DATA] Total windows: {usable}")
    print(f"[DATA] Train: {len(train_x)}, Val: {len(val_x)}")
    print(f"[DATA] Features: {x_all.shape}")

    # 4. Normalize targets (per-bank for stability)
    train_target_mean = train_y.mean(dim=(0, 1), keepdim=True)
    train_target_std = train_y.std(dim=(0, 1), keepdim=True) + 1e-6
    train_y_norm = (train_y - train_target_mean) / train_target_std
    val_y_norm = (val_y - train_target_mean) / train_target_std

    # 5. Model, optimizer, scheduler
    model = TemporalGNN(
        node_features=4,
        hidden_dim=hidden_dim,
        num_horizons=num_horizons,
        dropout=0.3,  # increased from 0.2 for better generalization at longer horizons
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    # 6. DataLoader with shuffling
    train_dataset = TensorDataset(train_x, train_y_norm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 7. Training loop
    best_val_loss = float("inf")
    best_val_mae = float("inf")
    patience_counter = 0
    best_epoch = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "lr": [],
    }

    print(f"[TRAIN] Starting {epochs} epochs with patience={patience}...")
    print("-" * 70)
    print(
        f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val MAE':>10} | {'LR':>10} | Status"
    )
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # Use same edge index for all samples in batch (simplified)
            batch_edges = [train_edges[0]] * batch_x.size(0)
            preds = model(batch_x, batch_edges)

            loss = horizon_weighted_loss(preds, batch_y, criterion, num_horizons)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds_list = []
            for i in range(0, len(val_x), batch_size):
                batch_x = val_x[i : i + batch_size]
                batch_edges = [val_edges[0]] * batch_x.size(0)
                val_preds_list.append(model(batch_x, batch_edges))
            val_preds = torch.cat(val_preds_list, dim=0)

            val_loss = horizon_weighted_loss(
                val_preds, val_y_norm, criterion, num_horizons
            ).item()

            # Denormalize for MAE 
            val_preds_denorm = val_preds * train_target_std + train_target_mean
            val_mae = torch.mean(torch.abs(val_preds_denorm - val_y)).item()

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["lr"].append(current_lr)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_mean": train_target_mean,
                    "target_std": train_target_std,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "history": history,
                },
                save_path,
            )
            status = "★ SAVED"
        else:
            patience_counter += 1
            status = f"patience={patience_counter}/{patience}"

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            overfit_ratio = avg_train_loss / max(val_loss, 1e-8)
            overfit_warn = " ⚠️ OVERFITTING" if overfit_ratio < 0.7 else ""
            print(
                f"  {epoch:>6} | {avg_train_loss:>12.6f} | {val_loss:>12.6f} | {val_mae:>10.6f} | {current_lr:>10.2e} | {status}{overfit_warn}"
            )

        if patience_counter >= patience:
            print(f"\n[EVAL] Early stopping at epoch {epoch}")
            break

    print("-" * 70)

    # 8. Final report
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best val MAE: {best_val_mae:.6f}")
    print(f"Saved to: {save_path}")

    # 9. Overfitting analysis
    final_train = history["train_loss"][-1]
    final_val = history["val_loss"][-1]
    overfit_ratio = final_train / max(final_val, 1e-8)

    print(f"\n[OVERFIT CHECK]")
    print(f"  Final train loss: {final_train:.6f}")
    print(f"  Final val loss: {final_val:.6f}")
    print(f"  Ratio (train/val): {overfit_ratio:.3f}")
    if overfit_ratio < 0.7:
        print(f"  ⚠️  WARNING: Model may be overfitting!")
        print(f"  ⚠️  Consider: lower hidden_dim, more dropout, or more data")
    else:
        print(f"  ✅ Good fit: train loss ~ val loss")

    # 10. Sample predictions
    print(f"\n[SAMPLE PREDICTIONS]")
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        test_pred = model.forecast_single(x_all[-1], edges_all[-1])
        test_pred_denorm = (
            test_pred * train_target_std.squeeze() + train_target_mean.squeeze()
        )

    tickers = loader.bank_tickers
    print(f"  Predicted returns (latest window):")
    for k in range(num_horizons):
        vals = ", ".join(
            f"{t}={test_pred_denorm[k, n]:.5f}" for n, t in enumerate(tickers)
        )
        print(f"    T+{k + 1}: {vals}")

    print(f"\nCompleted: {datetime.now().isoformat()}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved Temporal GNN Training")
    parser.add_argument(
        "--period", default="5y", help="Training data period (default: 5y)"
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--horizons", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--patience", type=int, default=40, help="Early stopping patience"
    )
    parser.add_argument("--save", default=None, help="Save path")
    args = parser.parse_args()

    train(
        period=args.period,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden,
        num_horizons=args.horizons,
        batch_size=args.batch_size,
        patience=args.patience,
        save_path=args.save,
    )
