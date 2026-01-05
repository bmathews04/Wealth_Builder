from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RiskParams:
    account_size: float = 10_000.0
    risk_per_trade_pct: float = 0.01  # 1% of account risked per trade
    max_positions: int = 10
    stop_type: str = "ATR"  # "ATR" or "PCT"
    atr_period: int = 14
    atr_mult: float = 2.0
    pct_stop: float = 0.08
    vol_target_annual: float = 0.20  # optional: scale position by vol
    vol_lookback_days: int = 63


def _true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    df must have columns: high, low, close, sorted by date
    """
    prev_close = df["close"].shift(1)
    tr = _true_range(df["high"], df["low"], prev_close)
    # Wilder smoothing
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def annualized_vol(close: pd.Series, lookback_days: int = 63) -> Optional[float]:
    if close is None or close.dropna().shape[0] < lookback_days + 5:
        return None
    r = close.pct_change().dropna().tail(lookback_days)
    if r.empty:
        return None
    return float(r.std() * np.sqrt(252))


def build_trade_plan(
    prices: pd.DataFrame,
    tickers: List[str],
    params: RiskParams,
) -> pd.DataFrame:
    """
    Create a position sizing + stop plan for a list of tickers.
    prices long-form must include: date,ticker,high,low,close
    Returns table:
      ticker, entry, stop, stop_pct, risk_per_share, shares, position_$, position_%,
      vol_annual, vol_scale, notes
    """
    rows: List[Dict] = []
    risk_budget = params.account_size * params.risk_per_trade_pct

    # Split “slots” evenly (optional, for concentration control)
    # We'll compute shares per position by risk; then cap by equal-weight slot size.
    slot_cap = params.account_size / max(params.max_positions, 1)

    for t in tickers[: params.max_positions]:
        g = prices[prices["ticker"] == t].sort_values("date")
        if g.empty or g["close"].dropna().shape[0] < 50:
            rows.append({"ticker": t, "notes": "No/insufficient price data"})
            continue

        entry = float(g["close"].iloc[-1])

        # Stop calculation
        stop = None
        notes = []
        if params.stop_type.upper() == "ATR":
            atr = compute_atr(g[["high", "low", "close"]].copy(), period=params.atr_period).iloc[-1]
            if pd.notna(atr):
                stop = entry - float(params.atr_mult * atr)
            else:
                notes.append("ATR missing; fallback to pct stop")
        if stop is None:
            stop = entry * (1 - params.pct_stop)

        stop = max(stop, 0.01)
        risk_per_share = entry - stop
        if risk_per_share <= 0:
            rows.append({"ticker": t, "notes": "Invalid stop/risk_per_share"})
            continue

        # Base shares by risk budget
        shares = int(risk_budget / risk_per_share)

        # Vol targeting scaler (optional)
        vol = annualized_vol(g["close"], lookback_days=params.vol_lookback_days)
        vol_scale = 1.0
        if vol is not None and vol > 0:
            vol_scale = min(2.0, params.vol_target_annual / vol)  # cap scaling
            shares = int(shares * vol_scale)

        # Cap by slot size (avoid giant positions)
        position_value = shares * entry
        if position_value > slot_cap:
            shares = int(slot_cap / entry)
            position_value = shares * entry
            notes.append("Capped by slot size")

        stop_pct = (risk_per_share / entry)
        position_pct = position_value / params.account_size if params.account_size else None

        rows.append(
            dict(
                ticker=t,
                entry=entry,
                stop=stop,
                stop_pct=stop_pct,
                risk_per_share=risk_per_share,
                shares=shares,
                position_value=position_value,
                position_pct=position_pct,
                vol_annual=vol,
                vol_scale=vol_scale,
                notes="; ".join(notes) if notes else "",
            )
        )

    return pd.DataFrame(rows)
