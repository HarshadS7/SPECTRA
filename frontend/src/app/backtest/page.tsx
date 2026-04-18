"use client";

import React, { useState, useEffect } from "react";
import axios from "axios";
import Link from "next/link";
import {
    ArrowLeft,
    RefreshCw,
    BarChart3,
    Calendar,
    TrendingUp,
    Target,
} from "lucide-react";
import {
    ResponsiveContainer,
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
} from "recharts";

interface BacktestResult {
    date: string;
    horizon: number;
    predictions: Record<string, number>;
    actuals: Record<string, number>;
    metrics: {
        mae: number | null;
        directional_accuracy: number | null;
        correlation: number | null;
    };
}

interface BacktestResponse {
    aggregate: {
        total_days: number;
        avg_mae: number | null;
        avg_directional_accuracy: number | null;
        best_day: string | null;
        worst_day: string | null;
        per_horizon: {
            [key: string]: {
                avg_mae: number | null;
                avg_directional_accuracy: number | null;
                avg_correlation: number | null;
                num_samples: number;
            };
        };
        overall: {
            avg_mae: number | null;
            avg_directional_accuracy: number | null;
            total_predictions: number;
        };
    };
    results: BacktestResult[];
    timestamp: string;
}

// Colour helper: green when DA is high, red when low
function daColor(da: number | null | undefined): string {
    if (da == null) return "text-zinc-400";
    if (da >= 0.65) return "text-emerald-400";
    if (da >= 0.50) return "text-yellow-400";
    return "text-red-400";
}

export default function BacktestPage() {
    const [data, setData] = useState<BacktestResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [days, setDays] = useState(30);

    const fetchBacktest = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await axios.get<BacktestResponse>(
                `http://localhost:8000/api/backtest?days=${days}`
            );
            setData(res.data);
        } catch (err: any) {
            setError(err.response?.data?.detail || "Failed to run backtest");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchBacktest();
    }, [days]);

    // Build chart data: one point per T+1 result date (aggregate across horizons)
    const chartData = (() => {
        if (!data?.results?.length) return [];
        const byDate: Record<string, { mae: number[]; da: number[] }> = {};
        for (const r of data.results) {
            if (!byDate[r.date]) byDate[r.date] = { mae: [], da: [] };
            if (r.metrics?.mae != null) byDate[r.date].mae.push(r.metrics.mae);
            if (r.metrics?.directional_accuracy != null)
                byDate[r.date].da.push(r.metrics.directional_accuracy);
        }
        return Object.entries(byDate)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([date, v]) => ({
                date,
                mae: v.mae.length ? +(v.mae.reduce((a, b) => a + b, 0) / v.mae.length).toFixed(5) : null,
                da: v.da.length ? +(v.da.reduce((a, b) => a + b, 0) / v.da.length * 100).toFixed(1) : null,
            }));
    })();

    const overall = data?.aggregate?.overall;
    const perHorizon = data?.aggregate?.per_horizon ?? {};

    return (
        <div className="min-h-screen theme-bg-primary theme-text-primary p-8 font-sans transition-colors duration-300">
            <div className="max-w-6xl mx-auto">

                {/* Header */}
                <header className="mb-8 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link href="/" className="p-2 rounded-full hover:theme-bg-tertiary transition-colors">
                            <ArrowLeft className="w-6 h-6 theme-text-muted" />
                        </Link>
                        <div>
                            <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-500 bg-clip-text text-transparent">
                                Backtesting
                            </h1>
                            <p className="theme-text-muted text-sm mt-1">
                                Walk-forward evaluation — MAE &amp; directional accuracy per horizon
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        <select
                            value={days}
                            onChange={(e) => setDays(Number(e.target.value))}
                            className="theme-bg-secondary theme-text-primary theme-border border rounded-lg px-3 py-2 text-sm"
                        >
                            <option value={7}>Last 7 days</option>
                            <option value={14}>Last 14 days</option>
                            <option value={30}>Last 30 days</option>
                            <option value={60}>Last 60 days</option>
                        </select>
                        <button
                            onClick={fetchBacktest}
                            disabled={loading}
                            className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg transition-all disabled:opacity-50 text-white text-sm font-medium"
                        >
                            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
                            Run Backtest
                        </button>
                    </div>
                </header>

                {error && (
                    <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
                        {error}
                    </div>
                )}

                {/* ── Aggregate KPIs ── */}
                {overall && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                        <div className="theme-card theme-border border rounded-xl p-4">
                            <div className="flex items-center gap-2 theme-text-muted text-xs mb-2">
                                <Calendar className="w-4 h-4" /> Predictions
                            </div>
                            <p className="text-2xl font-mono font-bold theme-text-primary">
                                {overall.total_predictions ?? 0}
                            </p>
                        </div>

                        <div className="theme-card theme-border border rounded-xl p-4">
                            <div className="flex items-center gap-2 theme-text-muted text-xs mb-2">
                                <BarChart3 className="w-4 h-4" /> Overall MAE
                            </div>
                            <p className="text-2xl font-mono font-bold text-blue-400">
                                {overall.avg_mae != null ? overall.avg_mae.toFixed(5) : "N/A"}
                            </p>
                        </div>

                        <div className="theme-card theme-border border rounded-xl p-4">
                            <div className="flex items-center gap-2 theme-text-muted text-xs mb-2">
                                <Target className="w-4 h-4" /> Directional Acc
                            </div>
                            <p className={`text-2xl font-mono font-bold ${daColor(overall.avg_directional_accuracy)}`}>
                                {overall.avg_directional_accuracy != null
                                    ? `${(overall.avg_directional_accuracy * 100).toFixed(1)}%`
                                    : "N/A"}
                            </p>
                        </div>

                        <div className="theme-card theme-border border rounded-xl p-4">
                            <div className="flex items-center gap-2 theme-text-muted text-xs mb-2">
                                <TrendingUp className="w-4 h-4" /> Best Day (MAE)
                            </div>
                            <p className="text-lg font-mono font-bold text-purple-400">
                                {data?.aggregate?.best_day || "N/A"}
                            </p>
                        </div>
                    </div>
                )}

                {/* ── Per-Horizon Grid ── */}
                {Object.keys(perHorizon).length > 0 && (
                    <div className="theme-card theme-border border rounded-xl p-6 mb-8">
                        <h2 className="text-lg font-semibold mb-5">Per-Horizon Performance</h2>
                        <div className="grid grid-cols-5 gap-3">
                            {Object.entries(perHorizon).map(([h, m]) => (
                                <div key={h} className="text-center p-4 rounded-xl theme-bg-tertiary space-y-3">
                                    <p className="text-sm font-bold text-purple-400">T+{h}</p>

                                    <div>
                                        <p className="text-xl font-mono font-bold text-blue-400">
                                            {m?.avg_mae?.toFixed(5) ?? "—"}
                                        </p>
                                        <p className="text-xs theme-text-muted mt-0.5">MAE</p>
                                    </div>

                                    <div>
                                        <p className={`text-xl font-mono font-bold ${daColor(m?.avg_directional_accuracy)}`}>
                                            {m?.avg_directional_accuracy != null
                                                ? `${(m.avg_directional_accuracy * 100).toFixed(1)}%`
                                                : "—"}
                                        </p>
                                        <p className="text-xs theme-text-muted mt-0.5">Dir. Acc</p>
                                    </div>

                                    <p className="text-xs theme-text-muted">
                                        {m?.num_samples ?? 0} samples
                                    </p>
                                </div>
                            ))}
                        </div>
                        {/* Legend */}
                        <div className="flex gap-4 mt-4 text-xs theme-text-muted">
                            <span><span className="text-emerald-400 font-bold">≥65%</span> strong</span>
                            <span><span className="text-yellow-400 font-bold">50–65%</span> moderate</span>
                            <span><span className="text-red-400 font-bold">&lt;50%</span> weak</span>
                        </div>
                    </div>
                )}

                {/* ── Dual Chart ── */}
                {chartData.length > 0 && (
                    <div className="theme-card theme-border border rounded-xl p-6 mb-8">
                        <h2 className="text-lg font-semibold mb-4">MAE &amp; Directional Accuracy Over Time</h2>
                        <div className="h-72">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                    <XAxis dataKey="date" tick={{ fill: "#a1a1aa", fontSize: 10 }} />
                                    <YAxis
                                        yAxisId="mae"
                                        tick={{ fill: "#a1a1aa", fontSize: 10 }}
                                        label={{ value: "MAE", angle: -90, position: "insideLeft", style: { fill: "#71717a", fontSize: 10 } }}
                                    />
                                    <YAxis
                                        yAxisId="da"
                                        orientation="right"
                                        domain={[0, 100]}
                                        tick={{ fill: "#a1a1aa", fontSize: 10 }}
                                        label={{ value: "Dir Acc %", angle: 90, position: "insideRight", style: { fill: "#71717a", fontSize: 10 } }}
                                    />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 12 }}
                                        formatter={(value: any, name: string) =>
                                            name === "Dir Acc %" ? [`${value}%`, name] : [value, name]
                                        }
                                    />
                                    <Legend />
                                    <Line yAxisId="mae" type="monotone" dataKey="mae" name="MAE" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} connectNulls />
                                    <Line yAxisId="da" type="monotone" dataKey="da" name="Dir Acc %" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} connectNulls />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}

                {/* ── Daily Results Table ── */}
                {data?.results && data.results.length > 0 && (
                    <div className="theme-card theme-border border rounded-xl p-6">
                        <h2 className="text-lg font-semibold mb-4">Daily Results</h2>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="theme-text-muted text-xs">
                                        <th className="text-left p-2">Date</th>
                                        <th className="text-right p-2">Horizon</th>
                                        <th className="text-right p-2">MAE</th>
                                        <th className="text-right p-2">Dir. Acc</th>
                                        <th className="text-right p-2">Correlation</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {data.results.map((r, i) => (
                                        <tr key={i} className="border-t theme-border hover:theme-bg-tertiary transition-colors">
                                            <td className="p-2 font-mono">{r.date}</td>
                                            <td className="p-2 text-right font-mono text-purple-400">T+{r.horizon}</td>
                                            <td className="p-2 text-right font-mono text-blue-400">
                                                {r.metrics?.mae?.toFixed(5) ?? "—"}
                                            </td>
                                            <td className={`p-2 text-right font-mono ${daColor(r.metrics?.directional_accuracy)}`}>
                                                {r.metrics?.directional_accuracy != null
                                                    ? `${(r.metrics.directional_accuracy * 100).toFixed(0)}%`
                                                    : "—"}
                                            </td>
                                            <td className="p-2 text-right font-mono theme-text-muted">
                                                {r.metrics?.correlation?.toFixed(3) ?? "—"}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {loading && (
                    <div className="flex items-center justify-center py-20">
                        <div className="animate-spin w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full" />
                    </div>
                )}
            </div>
        </div>
    );
}
