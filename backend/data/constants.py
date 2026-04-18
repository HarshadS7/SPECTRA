"""
Shared constants for the ReLuLu / Spectra financial engine.
"""

# --- Bank universe ---
BANK_TICKERS = ["JPM", "BAC", "WFC", "C", "USB", "GS", "MS"]
MACRO_TICKERS = ["^TNX"]  # 10-Year Treasury Yield

# --- Graph construction ---
CORRELATION_THRESHOLD = 0.7

# --- Real-time tick intervals (seconds) ---
DATA_REFRESH_INTERVAL = 60        # re-fetch market data
FORECAST_RECOMPUTE_INTERVAL = 60  # re-run the GNN + netting pipeline
MAX_TICK_ERRORS = 5               # pause ticking after N consecutive failures

# --- Risk-adjusted settlement sizing ---
# Used to compute a risk-adjusted "required money" after netting:
#   required = net_load + multiplier * sum_i(risk_factor_i * outflow_i)
RISK_BUFFER_MULTIPLIER = 1.0

# Worst-case scenario: assume the top fraction of highest-risk banks
# require buffering their full outflow (in addition to netted load).
WORST_CASE_TOP_FRACTION = 0.30

# --- Time-series risk factor (per bank) ---
# Computed from recent log-returns. Default: EWMA volatility mixed with downside.
RISK_FACTOR_WINDOW_DAYS = 20
RISK_EWMA_LAMBDA = 0.94
RISK_DOWNSIDE_WEIGHT = 0.50  # 0=vol only, 1=downside only

# --- Anomaly detection ---
# A bank is flagged if its recent return exceeds this many σ from its own mean.
ANOMALY_Z_THRESHOLD = 2.5
ANOMALY_LOOKBACK_DAYS = 60    # window for computing μ and σ
ANOMALY_RECENT_DAYS  = 5      # how many recent days to scan for spikes

# --- Model paths (relative to backend/) ---
TEMPORAL_MODEL_PATH = "temporal_gnn_v2.pth"
LEGACY_MODEL_PATH = "super_node_v1.pth"

# --- Market caps for netting reports (billions USD) ---
MARKET_CAPS = {
    "JPM": 580, "BAC": 300, "WFC": 210,
    "C": 110, "USB": 65, "GS": 140, "MS": 150,
}
