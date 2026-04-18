"""
Backtest Runner
===============
Easy-to-use script to run model backtests with trained TemporalGNN.

Usage:
    cd backend
    python -m scripts.run_backtest
    python -m scripts.run_backtest --checkpoints temporal_gnn_v1.pth
"""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from data.loader import TimeSeriesLoader
from models.temporal_gnn import TemporalGNN
from core.backtest import BacktestEngine
import torch


def main(checkpoint_path: str, period: str = "2y", lookback: int = 60, stride: int = 5):
    print(f"\n{'=' * 60}")
    print(f"[Backtest] Running model evaluation")
    print(f"{'=' * 60}\n")

    print(f"[Data] Loading {period} of market data...")
    loader = TimeSeriesLoader(period=period).load()

    print(f"[Model] Loading checkpoint: {checkpoint_path}")
    model = TemporalGNN(node_features=4, hidden_dim=64, num_horizons=5)

    backtest = BacktestEngine(loader, model)
    backtest.set_model(model, checkpoint_path)

    print(f"[Backtest] Testing {lookback} days with stride {stride}...")
    results = backtest.run_model_backtest(
        lookback_days=lookback,
        stride=stride,
        return_limit=None,
    )

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}\n")

    print(f"Total predictions: {results['aggregate']['overall']['total_predictions']}")
    print(f"Overall MAE: {results['aggregate']['overall']['avg_mae']:.6f}\n")

    print("Per-Horizon Performance:")
    print("-" * 60)
    print(f"{'Horizon':<10} {'MAE':<12} {'Correlation':<12}")
    print("-" * 60)

    for h, metrics in results["aggregate"]["per_horizon"].items():
        mae = metrics["avg_mae"] if metrics["avg_mae"] is not None else 0
        corr = (
            metrics["avg_correlation"] if metrics["avg_correlation"] is not None else 0
        )
        print(f"T+{h:<9} {mae:.6f}     {corr:.4f}")

    print("-" * 60)
    print(f"\n[Backtest] Complete @ {results['timestamp']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model backtest")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(BACKEND_DIR, "temporal_gnn_v2.pth"),
        help="Path to model checkpoint",
    )
    parser.add_argument("--period", default="2y", help="Data period (default: 2y)")
    parser.add_argument(
        "--lookback", type=int, default=60, help="Days to backtest (default: 60)"
    )
    parser.add_argument(
        "--stride", type=int, default=5, help="Days between tests (default: 5)"
    )
    args = parser.parse_args()

    main(args.checkpoint, args.period, args.lookback, args.stride)
