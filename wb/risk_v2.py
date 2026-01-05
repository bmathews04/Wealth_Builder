from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PortfolioRiskParams:
    account_size: float = 10_000.0
    risk_per_trade_pct: float = 0.01
    max_positions: int = 10
    corr_lookback_days: int = 126
    max_pair_corr: float = 0.75       # avoid stacking highly correlated names
    heat_cap_pct: float = 0.06        # total portfolio risk if all stops hit (6% of account)


def compute_returns_wide(prices: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    df = prices[prices["ticker"].isin(tickers)][["date", "ticker", "close"]].copy()
    wide = df.pivot(index="date", columns="ticker", values="close").sort_index()
    rets = wide.pct_change().dropna(how="all")
    return rets


def corr_matrix(prices: pd.DataFrame, tickers: List[str], lookback_days: int = 126) -> pd.DataFrame:
    rets = compute_returns_wide(prices, tickers).tail(lookback_days)
    if rets.empty:
        return pd.DataFrame()
    return rets.corr()


def correlation_filter(
    prices: pd.DataFrame,
    ranked_tickers: List[str],
    lookback_days: int = 126,
    max_pair_corr: float = 0.75,
    max_n: int = 10,
) -> List[str]:
    """
    Greedy select from ranked list, skipping names too correlated with already-selected names.
    """
    selected: List[str] = []
    cm = corr_matrix(prices, ranked_tickers[: min(len(ranked_tickers), 200)], lookback_days=lookback_days)
    if cm.empty:
        return ranked_tickers[:max_n]

    for t in ranked_tickers:
        if t not in cm.columns:
            continue
        ok = True
        for s in selected:
            if s in cm.columns and abs(float(cm.loc[t, s])) >= max_pair_corr:
                ok = False
                break
        if ok:
            selected.append(t)
        if len(selected) >= max_n:
            break
    return selected


def portfolio_heat(
    trade_plan: pd.DataFrame,
    account_size: float,
) -> Dict[str, float]:
    """
    Heat = sum(position risk) / account size.
    Expects trade_plan has: shares, risk_per_share
    """
    if trade_plan is None or trade_plan.empty:
        return {"heat_pct": 0.0, "risk_dollars": 0.0}

    risk_dollars = float((trade_plan["shares"].fillna(0) * trade_plan["risk_per_share"].fillna(0)).sum())
    heat_pct = risk_dollars / account_size if account_size else 0.0
    return {"heat_pct": heat_pct, "risk_dollars": risk_dollars}


def enforce_heat_cap(
    trade_plan: pd.DataFrame,
    account_size: float,
    heat_cap_pct: float,
) -> pd.DataFrame:
    """
    If portfolio heat > cap, scale down shares proportionally.
    """
    out = trade_plan.copy()
    h = portfolio_heat(out, account_size)
    if h["heat_pct"] <= heat_cap_pct or h["heat_pct"] <= 0:
        out["heat_scaled"] = False
        return out

    scale = heat_cap_pct / h["heat_pct"]
    out["shares"] = (out["shares"].fillna(0) * scale).astype(int)
    out["position_value"] = out["shares"].fillna(0) * out["entry"].fillna(0)
    out["position_pct"] = out["position_value"] / account_size if account_size else None
    out["heat_scaled"] = True
    return out
