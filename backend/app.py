"""
ReLuLu / Spectra — Application entry point.

Start the server:
    cd backend
    python app.py
"""

import os
import sys
import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Make 'backend' a package-level import root ──────────────────────
# This lets `from data.loader import …` etc. work when running
# `python app.py` directly (i.e. not via `python -m`).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.forecast_engine import ForecastEngine
from data.loader import TimeSeriesLoader
from api.routes import router, set_engine
from api import ticker


# ── Blocking helpers injected into the ticker ───────────────────────
_engine: ForecastEngine | None = None
_loader: TimeSeriesLoader | None = None


def _refresh_data():
    """Download market data and rebuild the forecast engine (blocking)."""
    global _engine, _loader

    _loader = TimeSeriesLoader(period="2y").load()

    temporal_path = os.path.join(SCRIPT_DIR, "temporal_gnn_v2.pth")
    legacy_path = os.path.join(SCRIPT_DIR, "super_node_v1.pth")

    if os.path.exists(temporal_path):
        model, norm_stats = ForecastEngine.load_temporal(temporal_path)
        _engine = ForecastEngine(model, _loader, norm_stats=norm_stats)
    else:
        model = ForecastEngine.load_legacy(legacy_path)
        _engine = ForecastEngine(model, _loader)
    set_engine(_engine)


def _recompute_forecast():
    """Run the forecast pipeline and return the result (blocking)."""
    if _engine is None:
        return None
    return _engine.run_forecast()


# ── Lifespan (startup / shutdown) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data → cache first forecast → start background ticker → yield."""
    ticker.configure(_refresh_data, _recompute_forecast)

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, ticker.refresh_data_sync)
        await loop.run_in_executor(None, ticker.recompute_forecast_sync)
        print("[APP] Engine ready — starting live ticker")
    except Exception as exc:
        import traceback

        print(f"[APP] STARTUP ERROR: {exc}")
        traceback.print_exc()

    task = asyncio.create_task(ticker.ticker_task())
    print("[APP] Ticker task created, yielding control to uvicorn")

    try:
        yield
    finally:
        ticker.tick_running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        print("[APP] Ticker stopped")


# ── FastAPI app ─────────────────────────────────────────────────────
app = FastAPI(title="ReLuLu / Spectra Financial Engine API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ── Run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
