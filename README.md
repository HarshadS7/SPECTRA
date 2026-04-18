# 🌌 SPECTRA

Systemic risk engine for interbank payment networks. Temporal GNN forecasts how liquidity shocks cascade across banks, identifies systemically critical institutions, and quantifies settlement savings through multilateral obligation netting.

Built with PyTorch Geometric, FastAPI, and Next.js 16. Ingests live Yahoo Finance data every 60 seconds.

## ML Pipeline (`models/`, `data/`, `training/`)

- **TemporalGNN** — GCNConv×2 → LSTM×2 → 5 independent horizon heads. Processes 10-day rolling windows of 4 features (log-return, MACD, volume Δ, Treasury yield) across 7 bank nodes (JPM, BAC, WFC, C, USB, GS, MS). Predicts absolute log-returns at T+1 through T+5.
- **Dynamic graph** — Edges rebuilt per window from rolling 30-day correlation (60%) + co-movement events (40%), thresholded at 70th percentile. ~12 edges for 7 banks. Strictly past data only.
- **Training** — 5y data, horizon-weighted MSE loss `[0.8, 1.0, 1.0, 0.9, 0.9]`, dropout=0.3, early stopping (patience=40), time-based 80/20 split with horizon gap to prevent leakage.

## Risk Pipeline (`core/`)

- **Obligation matrix** — Simulated from market caps, return correlations, and GNN-predicted stress scores. Banks predicted to move more get obligations scaled up to 3×.
- **Risk Jacobian** — `∂stress/∂obligations` via `torch.autograd`. Stress function: `σ(−(liquidity + inflow − outflow) / outflow)`. Row-sum of the Jacobian identifies systemic hubs.
- **Hub detection** — Eigen-decomposition of the risk-adjusted matrix. Principal eigenvector = hub centrality scores. Spectral radius < 1.0 = stable; ≥ 1.0 = cascading failure risk.
- **Circular netting** — Finds obligation cycles (A→B→C→A) via NetworkX, cancels bottleneck weights, priority-sorted by hub risk. Typical payload reduction: 30–60%.
- **Three scenarios** per horizon: base (post-netting), risk-adjusted (+ per-bank risk buffer), worst-case (+ full outflow of top 30% riskiest banks).

## Dashboard (`frontend/`)

- **Main** (`/`) — Live forecast with horizon selector, obligation matrices (before/after netting), hub rankings, stability index, risk-adjusted settlement cards, auto-refresh every 60s
- **Backtest** (`/backtest`) — Walk-forward eval with per-horizon MAE + directional accuracy, dual-axis chart, daily results table
- **AI Analyst** (`/analyst`) — LLM risk narratives via Meta-Llama 3.1 70B (Featherless API)
- **Alerts** (`/alerts`) — Threshold-based alerts on stability index, payload reduction, hub shifts

## Performance

Walk-forward backtest, stride=5, 7 banks, 60-day window:

| Horizon | MAE | Dir. Accuracy |
|---------|-----|---------------|
| T+1 | 0.0175 | 57% |
| T+2 | 0.0119 | **81%** |
| T+3 | 0.0102 | **71%** |
| T+4 | 0.0223 | 29% |
| T+5 | 0.0249 | 44% |

T+1–T+3 are reliable. T+4/T+5 degrade toward random — expected with 7 nodes at longer horizons. Magnitude estimates remain useful for risk sizing.

## Getting Started

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m training.train_improved --period 5y --epochs 300  # optional retrain
python app.py                                                # :8000

cd frontend
npm install && npm run dev                                   # :3000
```

Optional: `backend/.env` with `FEATHERLESS_API_KEY=...` for the AI analyst.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/forecast` | Cached 5-horizon forecast |
| `GET` | `/api/backtest?days=N` | Walk-forward eval (MAE + dir. accuracy) |
| `GET` | `/api/analyst/risk?horizon=N` | LLM risk narrative |
| `GET/POST/DELETE` | `/api/alerts` | Alert CRUD + `/triggered` + `/check` |
| `GET` | `/api/tick` | Background ticker status |
| `GET` | `/api/health` | Model/data readiness |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 64 | GNN + LSTM hidden size |
| `window_size` | 10 | Rolling window (trading days) |
| `num_horizons` | 5 | Forecast steps ahead |
| `dropout` | 0.3 | LSTM + encoder dropout |
| `horizon_weights` | `[0.8,1.0,1.0,0.9,0.9]` | Loss weighting per horizon |
| `DATA_REFRESH_INTERVAL` | 60s | Market data refresh cycle |
| `FORECAST_RECOMPUTE_INTERVAL` | 60s | Forecast recompute cycle |
| `CORRELATION_THRESHOLD` | 70th pct | Edge connectivity threshold |
| `RISK_BUFFER_MULTIPLIER` | configurable | Scales risk-adjusted settlement buffer |

## License

MIT
