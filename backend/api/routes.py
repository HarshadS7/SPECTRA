"""
FastAPI route handlers.
"""

from fastapi import APIRouter

from api.schemas import (
    BankResult,
    EdgeResult,
    HorizonSnapshot,
    DataMeta,
    ForecastResponse,
    PipelineResponse,
    AnalystResponse,
    BacktestResponse,
    AlertConfig,
    AlertTriggered,
    AlertCreateRequest,
)
import os
from core.analyst import ReLuLuAnalyst
from core.backtest import BacktestEngine
from core.alerts import get_alert_manager
from api import ticker
from data.constants import DATA_REFRESH_INTERVAL, FORECAST_RECOMPUTE_INTERVAL

router = APIRouter(prefix="/api")

# Reference to the ForecastEngine — set by app.py
_engine = None


def set_engine(engine):
    global _engine
    _engine = engine


# =====================================================================
# Helpers
# =====================================================================
def _format_snapshot(snap: dict, tickers: list[str]) -> HorizonSnapshot:
    """Convert a raw engine snapshot dict into the API schema."""
    n = len(tickers)
    banks = []
    for i, name in enumerate(tickers):
        banks.append(
            BankResult(
                name=name,
                predicted_score=round(float(snap["node_scores"][i]), 6),
                hub_score=round(float(snap["systemic_hubs"][i]), 4),
                risk_factor=round(float(snap["risk_factor"][i]), 6),
            )
        )

    ob_before = snap["obligations_before"]
    ob_after = snap["obligations_after"]

    edges_before, edges_after = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            wb = float(ob_before[i][j])
            wa = float(ob_after[i][j])
            if wb > 0.01:
                edges_before.append(
                    EdgeResult(
                        source=tickers[i],
                        target=tickers[j],
                        weight_before=round(wb, 4),
                        weight_after=round(wa, 4),
                    )
                )
            if wa > 0.01:
                edges_after.append(
                    EdgeResult(
                        source=tickers[i],
                        target=tickers[j],
                        weight_before=round(wb, 4),
                        weight_after=round(wa, 4),
                    )
                )

    return HorizonSnapshot(
        horizon=snap["horizon"],
        banks=banks,
        edges_before=edges_before,
        edges_after=edges_after,
        stability=round(snap["stability"], 6),
        is_stable=snap["stability"] < 1.0,
        payload_reduction=round(snap["payload_reduction"], 2),
        raw_load=round(snap["raw_load"], 2),
        net_load=round(snap["net_load"], 2),
        risk_buffer=round(float(snap.get("risk_buffer", 0.0)), 2),
        risk_adjusted_net_load=round(
            float(snap.get("risk_adjusted_net_load", snap["net_load"])), 2
        ),
        risk_adjusted_payload_reduction=round(
            float(
                snap.get("risk_adjusted_payload_reduction", snap["payload_reduction"])
            ),
            2,
        ),
        worst_case_buffer=round(float(snap.get("worst_case_buffer", 0.0)), 2),
        worst_case_net_load=round(
            float(snap.get("worst_case_net_load", snap["net_load"])), 2
        ),
        worst_case_payload_reduction=round(
            float(snap.get("worst_case_payload_reduction", snap["payload_reduction"])),
            2,
        ),
        obligations_before=[[round(float(v), 4) for v in row] for row in ob_before],
        obligations_after=[[round(float(v), 4) for v in row] for row in ob_after],
    )


# =====================================================================
# Routes
# =====================================================================
@router.get("/forecast", response_model=ForecastResponse)
def run_forecast():
    """Multi-horizon temporal forecast — returns the latest cached result."""
    result = ticker.cached_forecast

    if result is None:
        result = _engine.run_forecast()

    meta = result["metadata"]
    tickers = meta["tickers"]
    snapshots = [_format_snapshot(s, tickers) for s in result["horizons"]]
    model_type = "TemporalGNN" if _engine.is_temporal else "SuperNodeGNN (legacy)"

    return ForecastResponse(
        horizons=snapshots,
        metadata=DataMeta(
            tickers=tickers,
            num_banks=meta["num_banks"],
            total_days=meta["total_days"],
            date_range=list(meta["date_range"]),
            last_updated=meta["last_updated"],
            model_type=model_type,
        ),
    )


@router.get("/tick")
def tick_status():
    """Live tick status — used by the frontend to show freshness."""
    return {
        "tick_count": ticker.tick_count,
        "last_data_refresh": ticker.last_data_refresh,
        "last_forecast_time": ticker.last_forecast_time,
        "ticker_active": ticker.tick_running,
        "consecutive_errors": ticker.tick_errors,
        "data_refresh_interval_s": DATA_REFRESH_INTERVAL,
        "forecast_recompute_interval_s": FORECAST_RECOMPUTE_INTERVAL,
    }


@router.get("/config")
def config():
    """Returns tick intervals so the frontend can synchronize its polling."""
    return {
        "data_refresh_interval_s": DATA_REFRESH_INTERVAL,
        "forecast_recompute_interval_s": FORECAST_RECOMPUTE_INTERVAL,
        "frontend_poll_interval_s": FORECAST_RECOMPUTE_INTERVAL,
    }


@router.get("/run", response_model=PipelineResponse)
def run_pipeline():
    """Backward-compatible single-snapshot endpoint (horizon 1 only)."""
    result = _engine.run_forecast()
    tickers = result["metadata"]["tickers"]
    snap = _format_snapshot(result["horizons"][0], tickers)
    return PipelineResponse(
        banks=snap.banks,
        edges_before=snap.edges_before,
        edges_after=snap.edges_after,
        stability=snap.stability,
        is_stable=snap.is_stable,
        payload_reduction=snap.payload_reduction,
        raw_load=snap.raw_load,
        net_load=snap.net_load,
        risk_buffer=snap.risk_buffer,
        risk_adjusted_net_load=snap.risk_adjusted_net_load,
        risk_adjusted_payload_reduction=snap.risk_adjusted_payload_reduction,
        worst_case_buffer=snap.worst_case_buffer,
        worst_case_net_load=snap.worst_case_net_load,
        worst_case_payload_reduction=snap.worst_case_payload_reduction,
        obligations_before=snap.obligations_before,
        obligations_after=snap.obligations_after,
    )


@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _engine is not None,
        "model_type": "TemporalGNN" if (_engine and _engine.is_temporal) else "legacy",
        "data_loaded": ticker.cached_forecast is not None,
    }


@router.get("/analyst/risk", response_model=AnalystResponse)
def get_risk_assessment(horizon: int = 1):
    """
    Get AI-generated risk assessment for a specific horizon (default T+1).
    """
    api_key = os.getenv("FEATHERLESS_API_KEY")
    if not api_key:
        return AnalystResponse(
            risk_assessment="⚠️ Featherless API key not found. Please set FEATHERLESS_API_KEY in .env.",
            status="error",
        )

    analyst = ReLuLuAnalyst(api_key)

    # Get cached forecast
    result = ticker.cached_forecast
    if result is None:
        return AnalystResponse(
            risk_assessment="⚠️ System initializing - no forecast data available yet.",
            status="error",
        )

    # Validate horizon
    if horizon < 1 or horizon > len(result["horizons"]):
        return AnalystResponse(
            risk_assessment=f"⚠️ Invalid horizon T+{horizon}. Max horizon is T+{len(result['horizons'])}.",
            status="error",
        )

    # Extract horizon data
    snap = result["horizons"][horizon - 1]

    # Prepare data for analyst
    # adapt snapshot data to flat dictionary expected by analyst.py
    forecast_data = {
        "hubs": snap[
            "systemic_hubs"
        ],  # This assumes systemic_hubs is a list of names/scores?
        # Wait, in forecast_engine.py:
        # hubs, stability = self.optimizer.get_systemic_hubs(risk_adj)
        # get_systemic_hubs returns (centrality_scores, stability_limit)
        # It doesn't seem to return names directly in 'systemic_hubs'.
        # Let's check _format_snapshot.
        # snap["systemic_hubs"] is used as hub_score in BankResult.
        # so snap["systemic_hubs"] is a list of floats (centrality scores).
        # analyst.py expects: "hubs: List of primary systemic hub bank names"
        # So I need to find the hubs from the scores.
        "reduction_pct": snap["payload_reduction"],
        "stability_index": snap["stability"],
        "horizon": horizon,
        "is_stable": snap["stability"] < 1.0,
        "raw_load": snap["raw_load"],
        "net_load": snap["net_load"],
        "all_horizons": [],  # populate if we want temporal analysis
    }

    # Find hub names (banks with high centrality)
    # We need the list of tickers.
    tickers = result["metadata"]["tickers"]
    hub_scores = snap["systemic_hubs"]  # list of floats

    # Identify top hub(s)
    # Simple heuristic: max score
    if hub_scores is not None and len(hub_scores) == len(tickers):
        max_score = max(hub_scores)
        hub_indices = [i for i, s in enumerate(hub_scores) if s == max_score]
        forecast_data["hubs"] = [tickers[i] for i in hub_indices]
    else:
        forecast_data["hubs"] = ["Unknown"]

    # Add temporal data if requested or available
    # analyst.py logic: if all_horizons and len > 1 -> temporal prompt
    # Let's populate all_horizons with simplified data
    all_horizons_data = []
    for h_snap in result["horizons"]:
        all_horizons_data.append(
            {
                "stability_index": h_snap["stability"],
                "reduction_pct": h_snap["payload_reduction"],
                "net_load": h_snap["net_load"],
                "horizon": h_snap.get("horizon", 0),
            }
        )
    forecast_data["all_horizons"] = all_horizons_data

    assessment = analyst.summarize_risk(forecast_data)

    return AnalystResponse(risk_assessment=assessment, status="ok")


# =====================================================================
# Backtesting Routes
# =====================================================================
@router.get("/backtest")
def run_backtest(days: int = 30):
    """
    Run a model-based backtest comparing predictions to actual outcomes.
    Uses the trained TemporalGNN model for forecasting.

    Args:
        days: Number of days to backtest (default 30)
    """
    if _engine is None or _engine.loader is None:
        return {
            "aggregate": {
                "total_days": 0,
                "avg_mae": None,
                "avg_directional_accuracy": None,
                "best_day": None,
                "worst_day": None,
                "per_horizon": {},
                "overall": {
                    "avg_mae": None,
                    "avg_directional_accuracy": None,
                    "total_predictions": 0,
                },
            },
            "results": [],
            "timestamp": "",
        }

    backtest = BacktestEngine(_engine.loader, _engine.model, device=_engine.device)
    backtest.set_model(_engine.model)
    backtest.normalization_stats = _engine.norm_stats
    result = backtest.run_model_backtest(lookback_days=days, stride=1)

    # Format for frontend compatibility
    agg = result.get("aggregate", {})
    overall = agg.get("overall", {})

    # Find best/worst days
    best_day = None
    worst_day = None
    best_mae = float('inf')
    worst_mae = -float('inf')

    for r in result.get("results", []):
        mae = r.get("metrics", {}).get("mae")
        if mae is not None:
            if mae < best_mae:
                best_mae = mae
                best_day = r.get("date")
            if mae > worst_mae:
                worst_mae = mae
                worst_day = r.get("date")

    return {
        "aggregate": {
            "total_days": overall.get("total_predictions", 0),
            "avg_mae": overall.get("avg_mae"),
            "avg_directional_accuracy": overall.get("avg_directional_accuracy"),
            "best_day": best_day,
            "worst_day": worst_day,
            "per_horizon": agg.get("per_horizon", {}),
            "overall": overall,
        },
        "results": result.get("results", []),
        "timestamp": result.get("timestamp", ""),
    }


@router.get("/backtest/history")
def get_backtest_history(limit: int = 10):
    """Get recent backtest history entries."""
    if _engine is None or _engine.loader is None:
        return {"history": []}

    backtest = BacktestEngine(_engine.loader)
    return {"history": backtest.get_history(limit=limit)}


# =====================================================================
# Alert Routes
# =====================================================================
@router.get("/alerts", response_model=list[AlertConfig])
def get_alerts():
    """Get all configured alerts."""
    manager = get_alert_manager()
    return manager.get_alerts()


@router.post("/alerts", response_model=AlertConfig)
def create_alert(request: AlertCreateRequest):
    """Create a new alert configuration."""
    manager = get_alert_manager()
    return manager.create_alert(
        alert_type=request.type,
        name=request.name,
        threshold=request.threshold,
        description=request.description,
        enabled=request.enabled,
    )


@router.delete("/alerts/{alert_id}")
def delete_alert(alert_id: str):
    """Delete an alert by ID."""
    manager = get_alert_manager()
    success = manager.delete_alert(alert_id)
    return {"success": success, "id": alert_id}


@router.get("/alerts/triggered", response_model=list[AlertTriggered])
def get_triggered_alerts(since: str | None = None):
    """Get recently triggered alerts."""
    manager = get_alert_manager()
    return manager.get_triggered_alerts(since=since)


@router.post("/alerts/check")
def check_alerts():
    """
    Manually check all alerts against current forecast data.
    Returns list of newly triggered alerts.
    """
    result = ticker.cached_forecast
    if result is None:
        return {"triggered": [], "message": "No forecast data available"}

    # Get first horizon data for checking
    snap = result["horizons"][0]
    forecast_data = {
        "stability": snap["stability"],
        "payload_reduction": snap["payload_reduction"],
        "net_load": snap["net_load"],
    }

    manager = get_alert_manager()
    triggered = manager.check_alerts(forecast_data)

    return {
        "triggered": triggered,
        "checked_at": forecast_data,
    }
