from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RebalanceV2Result:
    equity_curve: pd.DataFrame         # portfolio, benchmark
    holdings: pd.DataFrame             # weights by rebalance date
    trades: pd.DataFrame               # turnover/costs by rebalance date
    stats: Dict


def _monthly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    months = pd.to_datetime(index).to_period("M")
    last_dates = (
        pd.Series(index=index, data=index)
        .groupby(months)
        .max()
        .sort_values()
        .values
    )
    return pd.DatetimeIndex(last_dates)


def _weekly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # rebalance on Fridays within available index
    s = pd.Series(index=index, data=index)
    w = s.groupby(index.to_period("W-FRI")).max().sort_values().values
    return pd.DatetimeIndex(w)


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


def backtest_rebalanced_v2(
    prices: pd.DataFrame,
    universe: List[str],
    ranker: Callable[[pd.Timestamp], List[str]],
    top_n: int = 10,
    benchmark: str = "SPY",
    start: Optional[str] = None,
    end: Optional[str] = None,
    rebalance: str = "M",              # "M" monthly, "W" weekly
    fee_bps: float = 5.0,              # per-side friction, e.g., 5 bps = 0.05%
    skip_if_overlap_pct: float = 0.70, # if new picks overlap old picks >= this, don't trade
) -> RebalanceV2Result:
    """
    Rebalanced strategy with:
      - causal ranker(asof) selection
      - turnover + transaction cost model (fee_bps per side)
      - optional 'skip rebalance if overlap high' (reduces churn)

    We model costs at rebalance as:
      cost = turnover * fee
    where turnover = sum(|w_new - w_old|) / 2  (one-way turnover)
    """

    symbols = sorted(list(dict.fromkeys(universe + [benchmark])))
    df = prices[prices["ticker"].isin(symbols)][["date", "ticker", "close"]].copy()
    wide = df.pivot(index="date", columns="ticker", values="close").sort_index()

    if start:
        wide = wide[wide.index >= pd.to_datetime(start)]
    if end:
        wide = wide[wide.index <= pd.to_datetime(end)]

    if wide.empty or benchmark not in wide.columns:
        return RebalanceV2Result(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"error": "Missing data/benchmark"})

    # daily returns
    rets = wide.pct_change().dropna(how="all")
    idx = rets.index

    # choose rebalance dates
    if rebalance.upper() == "W":
        rb_dates = _weekly_rebalance_dates(idx)
    else:
        rb_dates = _monthly_rebalance_dates(idx)
    rb_dates = rb_dates[(rb_dates >= idx.min()) & (rb_dates <= idx.max())]

    if len(rb_dates) < 2:
        return RebalanceV2Result(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"error": "Not enough rebalance points"})

    # helper
    fee = fee_bps / 10_000.0

    equity = pd.Series(index=idx, dtype=float)
    equity.iloc[0] = 1.0

    bench_eq = (1 + rets[benchmark].fillna(0.0)).cumprod()

    holdings_rows = []
    trades_rows = []

    w_old: Dict[str, float] = {}
    picks_old: List[str] = []

    def equal_weights(picks: List[str]) -> Dict[str, float]:
        picks = [p for p in picks if p in rets.columns and p != benchmark]
        picks = picks[:top_n]
        if not picks:
            return {}
        return {p: 1.0 / len(picks) for p in picks}

    for i in range(len(rb_dates) - 1):
        rb = rb_dates[i]
        nxt = rb_dates[i + 1]

        picks_new = ranker(rb)
        picks_new = [p for p in picks_new if p in rets.columns and p != benchmark]
        picks_new = picks_new[:top_n]

        # Skip rebalance if overlap is high (churn control)
        if picks_old and picks_new:
            overlap = len(set(picks_new) & set(picks_old)) / max(1, len(set(picks_old)))
            if overlap >= skip_if_overlap_pct:
                picks_new = picks_old[:]  # keep old holdings

        w_new = equal_weights(picks_new)

        # Turnover cost
        all_syms = sorted(set(w_old.keys()) | set(w_new.keys()))
        turnover = 0.0
        for s in all_syms:
            turnover += abs(w_new.get(s, 0.0) - w_old.get(s, 0.0))
        turnover *= 0.5
        cost = turnover * fee

        holdings_rows.append(pd.Series(w_new, name=rb))
        trades_rows.append(pd.Series({"turnover": turnover, "cost": cost, "n_holdings": len(w_new)}, name=rb))

        # Apply segment returns
        seg = rets.loc[(rets.index > rb) & (rets.index <= nxt), list(w_new.keys())]
        if seg.empty or not w_new:
            w_old = w_new
            picks_old = picks_new
            continue

        w_vec = pd.Series(w_new)
        port_ret = (seg.fillna(0.0) @ w_vec)

        # apply cost once at rebalance start (reduce equity)
        start_val = float(equity.loc[:rb].dropna().iloc[-1]) if equity.loc[:rb].dropna().shape[0] else 1.0
        start_val *= (1 - cost)

        seg_eq = start_val * (1 + port_ret).cumprod()
        equity.loc[seg_eq.index] = seg_eq.values

        w_old = w_new
        picks_old = picks_new

    equity = equity.ffill().fillna(1.0)
    curve = pd.DataFrame({"portfolio": equity, "benchmark": bench_eq.loc[equity.index]})

    holdings = pd.DataFrame(holdings_rows).fillna(0.0)
    holdings.index = pd.to_datetime(holdings.index)

    trades = pd.DataFrame(trades_rows).fillna(0.0)
    trades.index = pd.to_datetime(trades.index)

    port_rets = curve["portfolio"].pct_change().dropna()
    bench_rets = curve["benchmark"].pct_change().dropna()

    total_port = float(curve["portfolio"].iloc[-1] - 1)
    total_bench = float(curve["benchmark"].iloc[-1] - 1)
    n_days = int(curve.shape[0])

    stats = dict(
        n_days=n_days,
        rebalance=rebalance.upper(),
        top_n=top_n,
        fee_bps=fee_bps,
        skip_if_overlap_pct=skip_if_overlap_pct,
        avg_turnover=float(trades["turnover"].mean()) if not trades.empty else None,
        total_cost_est=float(trades["cost"].sum()) if not trades.empty else None,
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
    )

    return RebalanceV2Result(curve, holdings, trades, stats)
