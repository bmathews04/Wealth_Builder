from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame  # columns: portfolio, benchmark
    stats: dict


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _annualize_return(total_return: float, n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    years = n_days / 252.0
    return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0


def _annualize_vol(daily_returns: pd.Series) -> float:
    return float(daily_returns.std() * np.sqrt(252))


def backtest_equal_weight(
    prices: pd.DataFrame,
    tickers: List[str],
    benchmark: str = "SPY",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> BacktestResult:
    """
    Buy-and-hold equal weight from start to end (no rebalancing).
    Uses daily close prices from the shared long-form 'prices' table.
    """
    if not tickers:
        return BacktestResult(pd.DataFrame(), {"error": "No tickers"})

    # Pivot to wide close series
    df = prices[prices["ticker"].isin(tickers + [benchmark])][["date", "ticker", "close"]].copy()
    df = df.dropna()
    wide = df.pivot(index="date", columns="ticker", values="close").sort_index()

    if start:
        wide = wide[wide.index >= pd.to_datetime(start)]
    if end:
        wide = wide[wide.index <= pd.to_datetime(end)]

    # Need benchmark
    if benchmark not in wide.columns:
        return BacktestResult(pd.DataFrame(), {"error": f"Benchmark {benchmark} missing in price data"})

    # Keep only tickers with sufficient data overlap
    have = [t for t in tickers if t in wide.columns]
    wide = wide[have + [benchmark]].dropna()

    if wide.empty or len(have) == 0:
        return BacktestResult(pd.DataFrame(), {"error": "Not enough overlapping price data"})

    # Daily returns
    rets = wide.pct_change().dropna()
    port_rets = rets[have].mean(axis=1)  # equal-weight
    bench_rets = rets[benchmark]

    # Equity curves
    port_eq = (1 + port_rets).cumprod()
    bench_eq = (1 + bench_rets).cumprod()
    curve = pd.DataFrame({"portfolio": port_eq, "benchmark": bench_eq})

    # Stats
    total_port = float(curve["portfolio"].iloc[-1] - 1)
    total_bench = float(curve["benchmark"].iloc[-1] - 1)
    n_days = int(curve.shape[0])

    cagr_port = _annualize_return(total_port, n_days)
    cagr_bench = _annualize_return(total_bench, n_days)

    vol_port = _annualize_vol(port_rets)
    vol_bench = _annualize_vol(bench_rets)

    mdd_port = _max_drawdown(curve["portfolio"])
    mdd_bench = _max_drawdown(curve["benchmark"])

    sharpe_port = (port_rets.mean() / (port_rets.std() + 1e-12)) * np.sqrt(252)
    sharpe_bench = (bench_rets.mean() / (bench_rets.std() + 1e-12)) * np.sqrt(252)

    stats = dict(
        tickers_used=have,
        n_days=n_days,
        total_return_port=total_port,
        total_return_bench=total_bench,
        cagr_port=cagr_port,
        cagr_bench=cagr_bench,
        vol_port=vol_port,
        vol_bench=vol_bench,
        max_drawdown_port=mdd_port,
        max_drawdown_bench=mdd_bench,
        sharpe_port=float(sharpe_port),
        sharpe_bench=float(sharpe_bench),
    )

    return BacktestResult(curve, stats)


def equity_curve_chart(curve: pd.DataFrame, title: str = "Equity Curve") -> go.Figure:
    fig = go.Figure()
    if curve is None or curve.empty:
        fig.update_layout(title=f"{title} — no data", height=360)
        return fig

    fig.add_trace(go.Scatter(x=curve.index, y=curve["portfolio"], name="Portfolio"))
    fig.add_trace(go.Scatter(x=curve.index, y=curve["benchmark"], name="SPY"))
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
