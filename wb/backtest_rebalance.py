from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


@dataclass
class RebalanceBacktestResult:
    equity_curve: pd.DataFrame  # columns: portfolio, benchmark
    holdings: pd.DataFrame      # rows: rebalance dates, cols: tickers, values: weights
    stats: Dict


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


def _compute_daily_returns_wide(prices: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    df = prices[prices["ticker"].isin(symbols)][["date", "ticker", "close"]].copy()
    wide = df.pivot(index="date", columns="ticker", values="close").sort_index()
    rets = wide.pct_change().dropna(how="all")
    return rets


def _monthly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Use month-end business day within available index
    months = pd.to_datetime(index).to_period("M")
    last_dates = (
        pd.Series(index=index, data=index)
        .groupby(months)
        .max()
        .sort_values()
        .values
    )
    return pd.DatetimeIndex(last_dates)


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


def backtest_monthly_rebalanced(
    prices: pd.DataFrame,
    universe: List[str],
    ranker: Callable[[pd.Timestamp], List[str]],
    top_n: int = 10,
    benchmark: str = "SPY",
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_vol_target: bool = False,
    vol_lookback_days: int = 63,   # ~3 months
    target_annual_vol: float = 0.20,
    max_leverage: float = 1.5,
) -> RebalanceBacktestResult:
    """
    Monthly-rebalanced strategy:
      - On each rebalance date, choose top_n tickers from ranker(date)
      - Hold until next rebalance, applying daily returns
      - Weighting: equal-weight OR inverse-vol (scaled to target_annual_vol)
    """

    symbols = sorted(list(dict.fromkeys(universe + [benchmark])))
    rets = _compute_daily_returns_wide(prices, symbols)

    if rets.empty or benchmark not in rets.columns:
        return RebalanceBacktestResult(
            equity_curve=pd.DataFrame(),
            holdings=pd.DataFrame(),
            stats={"error": "Missing returns or benchmark in price data"},
        )

    if start:
        rets = rets[rets.index >= pd.to_datetime(start)]
    if end:
        rets = rets[rets.index <= pd.to_datetime(end)]

    if rets.empty:
        return RebalanceBacktestResult(pd.DataFrame(), pd.DataFrame(), {"error": "No returns in range"})

    bench = rets[benchmark].fillna(0.0)
    idx = rets.index

    # Monthly rebalance dates within range
    rb_dates = _monthly_rebalance_dates(idx)
    rb_dates = rb_dates[(rb_dates >= idx.min()) & (rb_dates <= idx.max())]
    if len(rb_dates) < 2:
        return RebalanceBacktestResult(pd.DataFrame(), pd.DataFrame(), {"error": "Not enough rebalance points"})

    equity = pd.Series(index=idx, dtype=float)
    equity.iloc[0] = 1.0
    bench_eq = (1 + bench).cumprod()
    holdings_rows = []

    current_weights: Dict[str, float] = {}
    current_tickers: List[str] = []

    # Helper to compute weights at rebalance
    def make_weights(asof: pd.Timestamp, picks: List[str]) -> Dict[str, float]:
        picks = [t for t in picks if t in rets.columns and t != benchmark]
        picks = picks[:top_n]
        if not picks:
            return {}

        if not use_vol_target:
            w = {t: 1.0 / len(picks) for t in picks}
            return w

        # inverse-vol weights using lookback vol
        lb = rets.loc[:asof, picks].tail(vol_lookback_days)
        vols = lb.std().replace(0, np.nan).fillna(np.nan)
        inv = (1.0 / vols).replace([np.inf, -np.inf], np.nan).dropna()
        if inv.empty:
            return {t: 1.0 / len(picks) for t in picks}

        inv = inv / inv.sum()
        # scale leverage to hit target_annual_vol (approx)
        port_vol = float((lb @ inv).std() * np.sqrt(252))
        lev = 1.0
        if port_vol > 0:
            lev = min(max_leverage, target_annual_vol / port_vol)
        inv = inv * lev
        # normalize to leverage (sum weights = lev, not necessarily 1)
        return inv.to_dict()

    # Iterate segments between rebalance dates
    for i in range(len(rb_dates) - 1):
        rb = rb_dates[i]
        nxt = rb_dates[i + 1]

        picks = ranker(rb)
        current_weights = make_weights(rb, picks)
        current_tickers = sorted(current_weights.keys())

        holdings_rows.append(pd.Series(current_weights, name=rb))

        # Apply returns rb(exclusive) -> nxt(inclusive)
        seg = rets.loc[(rets.index > rb) & (rets.index <= nxt), current_tickers]
        if seg.empty or not current_tickers:
            # If no holdings, portfolio flat
            continue

        # daily portfolio return
        w_vec = pd.Series(current_weights)
        port_ret = (seg.fillna(0.0) @ w_vec)

        # chain from last equity value up to rb (or earlier)
        start_val = float(equity.loc[:rb].dropna().iloc[-1]) if equity.loc[:rb].dropna().shape[0] else 1.0
        seg_eq = start_val * (1 + port_ret).cumprod()
        equity.loc[seg_eq.index] = seg_eq.values

    # Fill any gaps forward
    equity = equity.ffill().fillna(1.0)
    curve = pd.DataFrame({"portfolio": equity, "benchmark": bench_eq.loc[equity.index]})

    # Holdings table
    holdings = pd.DataFrame(holdings_rows).fillna(0.0)
    holdings.index = pd.to_datetime(holdings.index)

    # Stats
    port_rets = curve["portfolio"].pct_change().dropna()
    bench_rets = curve["benchmark"].pct_change().dropna()

    total_port = float(curve["portfolio"].iloc[-1] - 1)
    total_bench = float(curve["benchmark"].iloc[-1] - 1)
    n_days = int(curve.shape[0])

    stats = dict(
        n_days=n_days,
        total_return_port=total_port,
        total_return_bench=total_bench,
        cagr_port=_annualize_return(total_port, n_days),
        cagr_bench=_annualize_return(total_bench, n_days),
        vol_port=_annualize_vol(port_rets),
        vol_bench=_annualize_vol(bench_rets),
        max_drawdown_port=_max_drawdown(curve["portfolio"]),
        max_drawdown_bench=_max_drawdown(curve["benchmark"]),
        sharpe_port=float((port_rets.mean() / (port_rets.std() + 1e-12)) * np.sqrt(252)),
        sharpe_bench=float((bench_rets.mean() / (bench_rets.std() + 1e-12)) * np.sqrt(252)),
        use_vol_target=use_vol_target,
        target_annual_vol=target_annual_vol,
        max_leverage=max_leverage,
        top_n=top_n,
        n_rebalances=len(holdings),
    )

    return RebalanceBacktestResult(curve, holdings, stats)
