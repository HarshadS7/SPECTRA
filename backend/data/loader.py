"""
Time-Series Data Loader
========================
Fetches live market data via yfinance and produces rolling-window tensors
that the Temporal GNN consumes for multi-horizon forecasting.

Outputs:
  x_windows : [num_windows, num_nodes, seq_len, num_features]
  y_targets  : [num_windows, num_nodes]
  edge_index : [2, num_edges]  (correlation-based graph)
  metadata   : dict with timestamps, tickers, correlation matrix
"""

import torch
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from data.constants import (
    BANK_TICKERS,
    MACRO_TICKERS,
    CORRELATION_THRESHOLD,
    RISK_FACTOR_WINDOW_DAYS,
    RISK_EWMA_LAMBDA,
    RISK_DOWNSIDE_WEIGHT,
    ANOMALY_Z_THRESHOLD,
    ANOMALY_LOOKBACK_DAYS,
    ANOMALY_RECENT_DAYS,
)


class TimeSeriesLoader:
    """Loads real market data and yields rolling-window graph snapshots."""

    def __init__(
        self,
        bank_tickers: list[str] = BANK_TICKERS,
        macro_tickers: list[str] = MACRO_TICKERS,
        period: str = "2y",
        window_size: int = 10,
        correlation_threshold: float = CORRELATION_THRESHOLD,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        self.bank_tickers = bank_tickers
        self.macro_tickers = macro_tickers
        self.period = period
        self.window_size = window_size
        self.corr_threshold = correlation_threshold
        self.start_date = start_date
        self.end_date = end_date

        # Populated by .load()
        self.bank_returns: pd.DataFrame | None = None
        self.macro_returns: pd.DataFrame | None = None
        self.bank_macd: pd.DataFrame | None = None
        self.bank_vol: pd.DataFrame | None = None
        self.timestamps: list[str] = []
        self.corr_matrix: np.ndarray | None = None
        self.edge_index: torch.Tensor | None = None
        self.edge_indices: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self) -> "TimeSeriesLoader":
        """Download data and build the correlation graph.  Returns self."""
        all_tickers = self.bank_tickers + self.macro_tickers

        if self.start_date and self.end_date:
            print(
                f"[DataLoader] Downloading market data from {self.start_date} to {self.end_date} …"
            )
            df = yf.download(all_tickers, start=self.start_date, end=self.end_date)
        else:
            print(f"[DataLoader] Downloading {self.period} of market data …")
            df = yf.download(all_tickers, period=self.period)

        raw = df["Close"].dropna()
        volume = df["Volume"].fillna(0)

        # Log returns
        log_ret = np.log(raw / raw.shift(1)).dropna()

        # MACD (12-day - 26-day EMA)
        ema12 = raw.ewm(span=12, adjust=False).mean()
        ema26 = raw.ewm(span=26, adjust=False).mean()
        macd = ((ema12 - ema26) / (raw + 1e-8)).loc[log_ret.index]

        # Volume log change
        vol_change = np.log((volume + 1) / (volume.shift(1) + 1)).dropna()
        # Align indexes
        vol_change = vol_change.loc[log_ret.index].fillna(0)

        self.bank_returns = log_ret[self.bank_tickers]
        self.macro_returns = log_ret[self.macro_tickers]
        self.bank_macd = macd[self.bank_tickers]
        self.bank_vol = vol_change[self.bank_tickers]
        self.timestamps = [str(d.date()) for d in self.bank_returns.index]

        # Global Graph construction (for backward compatibility/legacy)
        corr_global = self.bank_returns.corr()
        self.corr_matrix = corr_global.values

        # Build edges from multiple factors, not just correlation threshold:
        # 1. Correlation (business relationship proxy)
        # 2. Rolling co-movement events (stress contagion channels)
        # 3. Market cap similarity (peer relationships)

        from data.constants import MARKET_CAPS

        caps = np.array([MARKET_CAPS.get(t, 50) for t in self.bank_tickers])

        # Co-movement: count days where both banks move > 1 std in same direction
        ret_std = self.bank_returns.std().values
        ret_normalized = self.bank_returns.values / (ret_std + 1e-8)
        comove = np.zeros((len(self.bank_tickers), len(self.bank_tickers)))

        for i in range(len(self.bank_tickers)):
            for j in range(i + 1, len(self.bank_tickers)):
                same_extreme = (
                    (np.abs(ret_normalized[:, i]) > 1.0)
                    & (np.abs(ret_normalized[:, j]) > 1.0)
                    & (np.sign(ret_normalized[:, i]) == np.sign(ret_normalized[:, j]))
                ).sum()
                comove[i, j] = comove[j, i] = same_extreme / len(ret_normalized)

        # Market cap similarity: similar-sized banks more likely to transact
        cap_sim = np.zeros((len(self.bank_tickers), len(self.bank_tickers)))
        for i in range(len(self.bank_tickers)):
            for j in range(len(self.bank_tickers)):
                cap_sim[i, j] = 1 - np.abs(caps[i] - caps[j]) / (
                    caps[i] + caps[j] + 1e-8
                )

        # Combine into connectivity score (Purely temporal behavior)
        connectivity = (
            0.6 * np.abs(self.corr_matrix)  # Correlation weight
            + 0.4 * comove  # Co-movement weight
        )

        # Threshold for edges
        edge_threshold = np.percentile(
            connectivity[~np.eye(len(self.bank_tickers), dtype=bool)], 70
        )
        mask = (connectivity > edge_threshold) & (
            ~np.eye(len(self.bank_tickers), dtype=bool)
        )
        self.edge_index = torch.tensor(np.array(np.nonzero(mask)), dtype=torch.long)

        print(
            f"[DataLoader] {len(self.bank_returns)} trading days, "
            f"{self.edge_index.shape[1]} edges (multi-factor connectivity)"
        )
        return self

    def _build_edge_index(
        self, ret_window: pd.DataFrame, caps: np.ndarray
    ) -> torch.Tensor:
        """Dynamically build an edge mask based on rolling 30-day correlations."""
        corr = ret_window.corr().fillna(0).values
        n = len(self.bank_tickers)

        # Cap similarity
        cap_sim = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cap_sim[i, j] = 1 - np.abs(caps[i] - caps[j]) / (
                    caps[i] + caps[j] + 1e-8
                )

        # Combine (Purely correlation based for rolling window)
        connectivity = np.abs(corr)
        edge_threshold = np.percentile(connectivity[~np.eye(n, dtype=bool)], 70)
        mask = (connectivity > edge_threshold) & (~np.eye(n, dtype=bool))
        return torch.tensor(np.array(np.nonzero(mask)), dtype=torch.long)

    def get_windows(self) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Build rolling windows with Z-Score normalization and dynamic edges.

        Returns
        -------
        x_windows : Tensor [num_windows, N, W, 4]
            Feature 0 = Return, 1 = Macro, 2 = MACD, 3 = Volume
        y_targets : Tensor [num_windows, N]
        edge_indices : List[Tensor [2, E]]
        """
        assert self.bank_returns is not None, "Call .load() first"

        bank_vals = self.bank_returns.values
        macro_vals = self.macro_returns.values
        macd_vals = self.bank_macd.values
        vol_vals = self.bank_vol.values

        # Precompute 30-day rolling moments for Z-scoring
        # IMPORTANT: Must use only PAST data to avoid leakage
        # For each point t, use [t-30:t) for mean/std - strictly past
        def rolling_z(arr):
            df = pd.DataFrame(arr)
            # Shift by 1 to ensure we use only PAST data (not including current)
            mean = df.shift(1).rolling(30, min_periods=1).mean()
            std = df.shift(1).rolling(30, min_periods=1).std().fillna(1e-4)
            # Add small epsilon to prevent div by zero
            std[std < 1e-4] = 1e-4
            return ((df - mean) / std).fillna(0).values

        z_bank = rolling_z(bank_vals)
        z_macro = rolling_z(macro_vals)
        z_macd = rolling_z(macd_vals)
        z_vol = rolling_z(vol_vals)

        from data.constants import MARKET_CAPS

        caps = np.array([MARKET_CAPS.get(t, 50) for t in self.bank_tickers])

        xs, ys, edges = [], [], []
        for i in range(len(bank_vals) - self.window_size):
            # Features over window W
            b_win = z_bank[i : i + self.window_size].T  # [N, W]
            m_win = z_macro[i : i + self.window_size].T  # [1, W]
            macd_win = z_macd[i : i + self.window_size].T  # [N, W]
            vol_win = z_vol[i : i + self.window_size].T  # [N, W]

            m_broad = np.tile(m_win, (b_win.shape[0], 1))  # [N, W]

            # Stack all 4 features: [N, W, 4]
            x_step = np.stack([b_win, m_broad, macd_win, vol_win], axis=-1)
            xs.append(x_step)

            # Target is the absolute raw return (as a proxy for volatility)
            ys.append(np.abs(bank_vals[i + self.window_size]))

            # Dynamic Edge Index - Use ONLY past data (strictly before target day)
            # Target is at day (i + window_size)
            # Edge window: [i - 30, i) - strictly before feature window
            target_day = i + self.window_size
            edge_start = max(0, i - 30)  # 30 days strictly before feature window
            edge_end = i  # End at start of feature window
            ret_window = self.bank_returns.iloc[edge_start:edge_end]
            edges.append(self._build_edge_index(ret_window, caps))

        x_windows = torch.tensor(np.array(xs), dtype=torch.float32)
        y_targets = torch.tensor(np.array(ys), dtype=torch.float32)
        self.edge_indices = edges
        return x_windows, y_targets, edges

    def get_latest_window(self) -> torch.Tensor:
        """Return the most-recent window for live inference.  Shape [N, W, 4]."""
        x_all, _, _ = self.get_windows()
        return x_all[-1]

    def get_latest_edge_index(self) -> torch.Tensor:
        if not self.edge_indices:
            _, _, _ = self.get_windows()
        return self.edge_indices[-1]

    def get_recent_windows(self, n: int = 5) -> torch.Tensor:
        x_all, _, _ = self.get_windows()
        return x_all[-n:]

    def get_metadata(self) -> dict:
        return {
            "tickers": self.bank_tickers,
            "num_banks": len(self.bank_tickers),
            "window_size": self.window_size,
            "correlation_matrix": self.corr_matrix.tolist()
            if self.corr_matrix is not None
            else [],
            "total_days": len(self.timestamps),
            "date_range": (self.timestamps[0], self.timestamps[-1])
            if self.timestamps
            else ("", ""),
            "last_updated": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Convenience: build base obligations from financial network structure
    # ------------------------------------------------------------------
    def build_base_obligations(self, scale: float = 10.0) -> torch.Tensor:
        """Derive a realistic asymmetric obligations matrix.

        Based on empirical properties of interbank networks:
        1. Larger banks (by market cap) have more total obligations
        2. Obligations follow a power-law-like distribution
        3. Asymmetry: smaller banks often owe larger ones for funding,
           larger banks owe smaller ones for settlement

        The matrix is constructed from:
        - Bank size (market cap) as primary driver
        - Correlation as proxy for business relationship strength
        - Volatility and returns as directional indicators
        """
        from data.constants import MARKET_CAPS

        n = len(self.bank_tickers)

        # Fallback: random if no data loaded
        if self.corr_matrix is None or self.bank_returns is None:
            obl = torch.abs(torch.randn(n, n)) * scale
            obl.fill_diagonal_(0)
            return obl

        # Get market caps (in billions USD) - these drive interbank volume
        caps = np.array([MARKET_CAPS.get(t, 50) for t in self.bank_tickers])

        # Total interbank obligations roughly proportional to bank size
        # Large banks have O(large bank) = k * market_cap * interbank_ratio
        # Typical interbank ratio: 1-5% of total assets
        total_ib_obligations = caps * scale * 0.05  # Scale to ~$5-30B range

        # Directional tilt based on recent returns and volatility
        # Banks under stress (negative returns, high vol) tend to be net payers
        recent = self.bank_returns.iloc[-20:]
        vol = recent.std().values
        mu = recent.mean().values

        # Net position indicator: negative returns + high vol = net payer
        net_payer_score = -mu + 0.5 * vol  # Higher = more likely to be net payer
        net_payer_score = (net_payer_score - net_payer_score.min()) / (
            net_payer_score.max() - net_payer_score.min() + 1e-8
        )

        # Correlation as connectivity proxy
        corr_abs = np.abs(self.corr_matrix)

        # Build obligations matrix
        obl = np.zeros((n, n))

        for i in range(n):
            # Total obligations for bank i (row sum)
            total_Oi = total_ib_obligations[i]

            # Distribute obligations across other banks based on:
            # 1. Counterparty size (larger = more likely to lend)
            # 2. Correlation strength (higher = stronger business ties)
            # 3. Network effects

            for j in range(n):
                if i == j:
                    continue

                # Counterparty weight: larger banks receive more obligations
                cap_weight = caps[j] / caps.sum()

                # Connectivity weight: stronger correlation = more likely to transact
                conn_weight = corr_abs[i, j]

                # Combine weights
                weight = cap_weight * (
                    0.5 + 0.5 * conn_weight
                )  # Cap gets 50% base weight

                # Preliminary obligation
                obl[i, j] = weight

            # Normalize row to sum to total obligations
            row_sum = obl[i, :].sum()
            if row_sum > 1e-8:
                obl[i, :] *= total_Oi / row_sum

        # Add asymmetry based on net payer scores
        # Net payers send more than they receive
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # If i is net payer relative to j, increase i->j
                payer_diff = net_payer_score[i] - net_payer_score[j]
                asymmetry_factor = 1.0 + 0.2 * payer_diff
                obl[i, j] *= max(0.5, asymmetry_factor)

        # Apply correlation-based adjustments for realistic pattern
        # Banks with high correlation have more bilateral activity
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Boost obligations between highly correlated banks
                obl[i, j] *= 1.0 + corr_abs[i, j]

        # Small deterministic noise for numerical stability (not random guess)
        rng = np.random.RandomState(42)
        noise = rng.uniform(0.98, 1.02, size=(n, n))
        obl *= noise

        # Ensure non-negative and zero diagonal
        obl = np.maximum(obl, 0)
        np.fill_diagonal(obl, 0)

        return torch.tensor(obl, dtype=torch.float32)

    def build_liquidity(self) -> torch.Tensor:
        """Estimate liquidity from multiple financial indicators.

        In reality, liquidity depends on:
        1. Bank size (larger = more deposits = more liquidity)
        2. Recent volatility (high vol = market stress = less liquidity)
        3. Return stability (consistent returns = stable funding)
        4. Market confidence (proxy via relative performance)
        """
        from data.constants import MARKET_CAPS

        n = len(self.bank_tickers)

        if self.bank_returns is None:
            return torch.abs(torch.randn(n)) * 100 + 50

        # Base liquidity from market cap (larger banks have more liquidity buffers)
        caps = np.array([MARKET_CAPS.get(t, 50) for t in self.bank_tickers])
        liq_base = caps / caps.max() * 100  # Normalize to 0-100

        # Volatility penalty: high vol = less stable funding
        recent = self.bank_returns.iloc[-20:]
        vol = recent.std().values
        vol_normalized = vol / (vol.max() + 1e-8)
        vol_penalty = (1 - vol_normalized) * 50  # 0-50 bonus for low vol

        # Consistency bonus: steady returns indicate reliable funding
        mu = recent.mean().values
        skew_indicator = np.abs(mu) / (np.abs(mu).max() + 1e-8)
        consistency_bonus = (1 - skew_indicator) * 20  # 0-20 bonus

        # Relative performance (outperforming banks have better market access)
        cumulative_ret = (1 + recent.values).prod(axis=0) - 1
        perf_normalized = (cumulative_ret - cumulative_ret.min()) / (
            cumulative_ret.max() - cumulative_ret.min() + 1e-8
        )
        perf_bonus = perf_normalized * 30  # 0-30 bonus

        # Combine factors
        liq = liq_base + vol_penalty + consistency_bonus + perf_bonus
        liq = np.clip(liq, 20, 200)  # Realistic liquidity range in billions

        return torch.tensor(liq, dtype=torch.float32)

    def build_time_series_risk_factor(
        self,
        window_days: int = RISK_FACTOR_WINDOW_DAYS,
        ewma_lambda: float = RISK_EWMA_LAMBDA,
        downside_weight: float = RISK_DOWNSIDE_WEIGHT,
    ) -> torch.Tensor:
        """Comprehensive per-bank risk factor in [0,1].

        Risk factors are based on multiple dimensions:
        1. Volatility (EWMA) - recent price instability
        2. Downside risk - magnitude of negative returns
        3. Tail risk - probability of extreme losses
        4. Drawdown - peak-to-trough decline
        5. Relative stress - performance vs peer group

        Returns
        -------
        Tensor [N] with values in [0, 1]. Higher = riskier.
        """
        n = len(self.bank_tickers)
        if self.bank_returns is None or len(self.bank_returns) == 0:
            rf = torch.rand(n)
            return torch.clamp(rf, 0.0, 1.0)

        r = self.bank_returns.iloc[-window_days:].values  # [T, N]
        if r.shape[0] < 2:
            return torch.zeros(n, dtype=torch.float32)

        # 1. EWMA volatility (RiskMetrics-style)
        lam = float(np.clip(ewma_lambda, 0.0, 0.9999))
        w = (1.0 - lam) * (lam ** np.arange(r.shape[0] - 1, -1, -1))
        w = w / (w.sum() + 1e-12)
        ewma_var = (w[:, None] * (r**2)).sum(axis=0)
        ewma_vol = np.sqrt(np.maximum(ewma_var, 0.0))

        # 2. Downside risk (semi-deviation)
        negative_returns = r * (r < 0)
        downside = np.sqrt((negative_returns**2).mean(axis=0))

        # 3. Tail risk: Value at Risk approximation (95% VaR)
        var_95 = np.percentile(np.abs(r), 95, axis=0)

        # 4. Maximum drawdown in the window
        cumulative = np.cumprod(1 + r, axis=0)
        running_max = np.maximum.accumulate(cumulative, axis=0)
        drawdowns = (cumulative - running_max) / (running_max + 1e-8)
        max_drawdown = np.abs(drawdowns.min(axis=0))

        # 5. Relative stress: deviation from sector average
        sector_avg = r.mean(axis=1, keepdims=True)
        rel_deviation = np.abs(r - sector_avg).mean(axis=0)

        def _norm01(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            x_min, x_max = x.min(), x.max()
            if not np.isfinite(x_max) or (x_max - x_min) < 1e-12:
                return np.zeros_like(x)
            return np.clip((x - x_min) / (x_max - x_min), 0.0, 1.0)

        vol_n = _norm01(ewma_vol)
        down_n = _norm01(downside)
        var_n = _norm01(var_95)
        dd_n = _norm01(max_drawdown)
        stress_n = _norm01(rel_deviation)

        risk = (
            0.20 * vol_n  # Volatility component
            + 0.25 * down_n  # Downside risk
            + 0.20 * var_n  # Tail risk
            + 0.20 * dd_n  # Drawdown
            + 0.15 * stress_n  # Relative stress
        )

        return torch.tensor(np.clip(risk, 0.0, 1.0), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Anomaly detection on time-series returns
    # ------------------------------------------------------------------
    def detect_anomalies(
        self,
        lookback_days: int = ANOMALY_LOOKBACK_DAYS,
        recent_days: int = ANOMALY_RECENT_DAYS,
        z_threshold: float = ANOMALY_Z_THRESHOLD,
    ) -> list[dict]:
        """Flag banks whose recent returns spike beyond z_threshold σ.

        Returns a list of dicts, one per anomaly detected:
          { "bank", "date", "return", "z_score", "direction" }
        Empty list if no anomalies.
        """
        if self.bank_returns is None or len(self.bank_returns) < lookback_days:
            return []

        window = self.bank_returns.iloc[-lookback_days:]
        mu = window.mean()  # [N]
        sigma = window.std()  # [N]

        recent = self.bank_returns.iloc[-recent_days:]
        anomalies = []
        for ticker in self.bank_tickers:
            s = sigma[ticker]
            m = mu[ticker]
            if s < 1e-12:
                continue
            for date_idx, ret_val in recent[ticker].items():
                z = (ret_val - m) / s
                if abs(z) >= z_threshold:
                    anomalies.append(
                        {
                            "bank": ticker,
                            "date": str(date_idx.date())
                            if hasattr(date_idx, "date")
                            else str(date_idx),
                            "return": float(ret_val),
                            "z_score": round(float(z), 3),
                            "direction": "SPIKE UP" if z > 0 else "SPIKE DOWN",
                        }
                    )
        return anomalies


# --------------- quick test ---------------
if __name__ == "__main__":
    loader = TimeSeriesLoader(period="1y").load()
    x, y = loader.get_windows()
    print(f"x_windows : {x.shape}")
    print(f"y_targets : {y.shape}")
    print(f"edge_index: {loader.edge_index.shape}")
    print(f"metadata  : {loader.get_metadata()}")
