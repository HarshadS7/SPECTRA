"""
Backtesting Engine
==================
Compares historical forecasts with actual outcomes to measure model accuracy.
"""

import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import os

BACKTEST_STORAGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "backtest_history.json"
)


class BacktestEngine:
    """Engine for backtesting model predictions against actual outcomes."""

    def __init__(self, loader, model=None, device: str | None = None):
        self.loader = loader
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.history: List[Dict] = self._load_history()
        self.normalization_stats = None

    def _load_history(self) -> List[Dict]:
        if os.path.exists(BACKTEST_STORAGE_PATH):
            try:
                with open(BACKTEST_STORAGE_PATH, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self):
        try:
            with open(BACKTEST_STORAGE_PATH, "w") as f:
                json.dump(self.history[-100:], f, indent=2)
        except Exception as e:
            print(f"[Backtest] Failed to save history: {e}")

    def set_model(self, model, checkpoint_path: str | None = None):
        self.model = model.to(self.device)
        if checkpoint_path:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.normalization_stats = {
                    "mean": checkpoint.get("target_mean"),
                    "std": checkpoint.get("target_std"),
                }
            else:
                self.model.load_state_dict(checkpoint)
        self.model.eval()

    def store_prediction(
        self,
        horizon: int,
        predictions: Dict[str, float],
        timestamp: Optional[str] = None,
    ):
        entry = {
            "timestamp": timestamp or datetime.now().isoformat(),
            "horizon": horizon,
            "predictions": predictions,
            "actual": None,
            "metrics": None,
        }
        self.history.append(entry)
        self._save_history()

    def update_actuals(self, target_date: str, actuals: Dict[str, float]):
        for entry in self.history:
            if entry["actual"] is None:
                entry["actual"] = actuals
                entry["metrics"] = self._compute_metrics(entry["predictions"], actuals)
                self._save_history()
                break

    def _compute_metrics(
        self, predictions: Dict[str, float], actuals: Dict[str, float]
    ) -> Dict[str, float]:
        common_tickers = set(predictions.keys()) & set(actuals.keys())
        if not common_tickers:
            return {"mae": None, "directional_accuracy": None, "correlation": None}

        pred_vals = np.array([predictions[t] for t in common_tickers])
        actual_vals = np.array([actuals[t] for t in common_tickers])

        mae = float(np.mean(np.abs(pred_vals - actual_vals)))

        # Fraction of nodes where predicted sign matches actual sign
        directional_accuracy = float(
            np.mean(np.sign(pred_vals) == np.sign(actual_vals))
        )

        if np.std(pred_vals) > 1e-8 and np.std(actual_vals) > 1e-8:
            correlation = float(np.corrcoef(pred_vals, actual_vals)[0, 1])
        else:
            correlation = 0.0

        return {
            "mae": round(mae, 6),
            "directional_accuracy": round(directional_accuracy, 4),
            "correlation": round(correlation, 4),
        }

    def run_model_backtest(
        self,
        lookback_days: int = 60,
        stride: int = 5,
        retrain_every: int | None = None,
        return_limit: int | None = 20,
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest using actual model predictions.

        For each date in the lookback period:
        1. Train on all data prior to that date
        2. Predict next K horizons
        3. Compare predictions to actual outcomes

        Args:
            lookback_days: Number of days to backtest
            stride: Days between backtest points (reduce computation)
            retrain_every: Retrain model every N days (None = no retraining, use current model)
            return_limit: Max results to return (None for all)

        Returns:
            Dict with per-horizon metrics and aggregate stats
        """
        if self.loader.bank_returns is None:
            return {"error": "No data loaded", "results": []}

        if self.model is None:
            return {"error": "No model set. Use set_model() first.", "results": []}

        returns = self.loader.bank_returns
        tickers = self.loader.bank_tickers
        n_days = len(returns)

        num_horizons = getattr(self.model, "num_horizons", 5)

        results = []
        all_metrics = {
            h: {"mae": [], "dir_acc": [], "corr": []}
            for h in range(1, num_horizons + 1)
        }

        test_indices = list(range(n_days - lookback_days, n_days - 1, stride))

        for idx in test_indices:
            if idx + num_horizons >= n_days:
                continue

            x_all, y_all, edges_all = self._get_windows_up_to(idx)
            if x_all is None:
                continue

            with torch.no_grad():
                x_input = x_all[-1].to(self.device)
                edge_input = edges_all[-1].to(self.device)
                pred_normalized = self.model.forecast_single(x_input, edge_input)

            if (
                self.normalization_stats
                and self.normalization_stats["mean"] is not None
            ):
                mean = self.normalization_stats["mean"].squeeze().to(self.device)
                std = self.normalization_stats["std"].squeeze().to(self.device)
                pred = pred_normalized * std + mean
            else:
                pred = pred_normalized

            pred = pred.cpu().numpy()
            if pred.ndim == 3:
                pred = pred.squeeze(0)

            for h in range(1, num_horizons + 1):
                if idx + h >= n_days:
                    continue

                h_idx = h - 1
                if h_idx >= len(pred):
                    continue

                pred_h = pred[h_idx]
                if pred_h.ndim > 1:
                    pred_h = pred_h.flatten()
                pred_vals = {t: float(pred_h[i]) for i, t in enumerate(tickers)}
                actual_vals = {
                    t: float(returns.iloc[idx + h][t]) for i, t in enumerate(tickers)
                }

                metrics = self._compute_metrics(pred_vals, actual_vals)

                result_entry = {
                    "date": str(returns.index[idx].date())
                    if hasattr(returns.index[idx], "date")
                    else str(returns.index[idx]),
                    "horizon": h,
                    "predictions": pred_vals,
                    "actuals": actual_vals,
                    "metrics": metrics,
                }
                results.append(result_entry)

                if metrics["mae"] is not None:
                    all_metrics[h]["mae"].append(metrics["mae"])
                if metrics["directional_accuracy"] is not None:
                    all_metrics[h]["dir_acc"].append(metrics["directional_accuracy"])
                if metrics["correlation"] is not None:
                    all_metrics[h]["corr"].append(metrics["correlation"])

        aggregate = {"per_horizon": {}, "overall": {}}
        for h in range(1, num_horizons + 1):
            h_metrics = all_metrics[h]
            aggregate["per_horizon"][h] = {
                "avg_mae": round(float(np.mean(h_metrics["mae"])), 6)
                if h_metrics["mae"]
                else None,
                "avg_directional_accuracy": round(float(np.mean(h_metrics["dir_acc"])), 4)
                if h_metrics["dir_acc"]
                else None,
                "avg_correlation": round(float(np.mean(h_metrics["corr"])), 4)
                if h_metrics["corr"]
                else None,
                "num_samples": len(h_metrics["mae"]),
            }

        all_mae = [m for h in range(1, num_horizons + 1) for m in all_metrics[h]["mae"]]
        all_dir_acc = [
            d for h in range(1, num_horizons + 1) for d in all_metrics[h]["dir_acc"]
        ]

        aggregate["overall"] = {
            "avg_mae": round(float(np.mean(all_mae)), 6) if all_mae else None,
            "avg_directional_accuracy": round(float(np.mean(all_dir_acc)), 4)
            if all_dir_acc
            else None,
            "total_predictions": len(all_mae),
        }

        output_results = results if return_limit is None else results[:return_limit]

        return {
            "aggregate": aggregate,
            "results": output_results,
            "timestamp": datetime.now().isoformat(),
            "model_type": type(self.model).__name__,
            "num_horizons": num_horizons,
        }

    def _get_windows_up_to(self, end_idx: int) -> tuple:
        """Get windows up to a specific index."""
        try:
            x_all, y_all, edges_all = self.loader.get_windows()
            if end_idx > len(x_all):
                return None, None, None
            return x_all[:end_idx], y_all[:end_idx], edges_all[:end_idx]
        except Exception as e:
            print(f"[Backtest] Error getting windows: {e}")
            return None, None, None

    def run_backtest(
        self, lookback_days: int = 30, return_limit: Optional[int] = 10
    ) -> Dict[str, Any]:
        """
        Legacy backtest using lag correlation (not model predictions).

        Note: For actual model evaluation, use run_model_backtest() instead.
        """
        if self.loader.bank_returns is None:
            return {"error": "No data loaded", "results": []}

        returns = self.loader.bank_returns
        tickers = self.loader.bank_tickers
        n_days = min(lookback_days, len(returns) - 1)

        results = []
        all_mae = []

        for i in range(n_days, 0, -1):
            if i + 1 >= len(returns):
                continue

            pred_day = returns.iloc[-(i + 1)]
            actual_day = returns.iloc[-i]

            predictions = {t: float(pred_day[t]) for t in tickers}
            actuals = {t: float(actual_day[t]) for t in tickers}

            metrics = self._compute_metrics(predictions, actuals)

            results.append(
                {
                    "date": str(returns.index[-i].date())
                    if hasattr(returns.index[-i], "date")
                    else str(returns.index[-i]),
                    "predictions": predictions,
                    "actuals": actuals,
                    "metrics": metrics,
                }
            )

            if metrics["mae"] is not None:
                all_mae.append(metrics["mae"])
            if metrics["directional_accuracy"] is not None:
                all_dir_acc.append(metrics["directional_accuracy"])

        aggregate = {
            "total_days": len(results),
            "avg_mae": round(float(np.mean(all_mae)), 6) if all_mae else None,
            "best_day": min(results, key=lambda x: x["metrics"]["mae"] or float("inf"))["date"]
            if results
            else None,
            "worst_day": max(results, key=lambda x: x["metrics"]["mae"] or 0)["date"]
            if results
            else None,
        }

        output_results = results if return_limit is None else results[:return_limit]

        return {
            "aggregate": aggregate,
            "results": output_results,
            "timestamp": datetime.now().isoformat(),
            "note": "Legacy backtest using lag correlation. Use run_model_backtest() for model evaluation.",
        }

    def get_history(self, limit: int = 10) -> List[Dict]:
        return self.history[-limit:]
