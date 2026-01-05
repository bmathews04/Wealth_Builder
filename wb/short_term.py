from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ShortTermParams:
    min_avg_dollar_vol: float = 20_000_000  # $20M/day liquidity floor
    require_above_ma50: bool = True
    require_above_ma200: bool = False
    max_dd_6m: float = 0.45  # max drawdown over last 6m
    min_mom_3m: float = 0.00  # 3m momentum
    min_mom_6m: float = 0.00  # 6m momentum
    max_dist_from_52w_high: float = 0.25  # within 25% of 52w high


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    # Wilder smoothing
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_short_term_table(prices: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Requires prices long-form columns:
      date, ticker, open, high, low, close, volume,
      ma20, ma50, ma200,
      ret_21d, ret_63d, ret_126d, ret_252d,
      avg_dollar_vol_20
    """
    rows: List[Dict] = []

    for t in tickers:
        g = prices[prices["ticker"] == t].sort_values("date")
        if g.empty:
            rows.append({"ticker": t})
            continue

        g = g.dropna(subset=["close"])
        if len(g) < 260:
            rows.append({"ticker": t})
            continue

        last = g.iloc[-1]
        close = float(last["close"])

        # Momentum
        mom_1m = float(last["ret_21d"]) if pd.notna(last.get("ret_21d")) else None
        mom_3m = float(last["ret_63d"]) if pd.notna(last.get("ret_63d")) else None
        mom_6m = float(last["ret_126d"]) if pd.notna(last.get("ret_126d")) else None
        mom_12m = float(last["ret_252d"]) if pd.notna(last.get("ret_252d")) else None

        # Trend
        ma20 = float(last["ma20"]) if pd.notna(last.get("ma20")) else None
        ma50 = float(last["ma50"]) if pd.notna(last.get("ma50")) else None
        ma200 = float(last["ma200"]) if pd.notna(last.get("ma200")) else None

        above_ma20 = (ma20 is not None and close > ma20)
        above_ma50 = (ma50 is not None and close > ma50)
        above_ma200 = (ma200 is not None and close > ma200)

        # Liquidity
        adv = float(last["avg_dollar_vol_20"]) if pd.notna(last.get("avg_dollar_vol_20")) else None

        # Distance from 52w high
        last_52w = g.tail(252)
        high_52w = float(last_52w["close"].max()) if not last_52w.empty else None
        dist_52w = None
        if high_52w and high_52w > 0:
            dist_52w = 1 - (close / high_52w)  # 0 = at highs, 0.2 = 20% below highs

        # 6m drawdown
        last_6m = g.tail(126)
        dd_6m = None
        if not last_6m.empty:
            peak = last_6m["close"].cummax()
            dd = (last_6m["close"] / peak) - 1.0
            dd_6m = abs(float(dd.min()))

        # RSI
        rsi = None
        try:
            rsi_val = _rsi(g["close"], 14).iloc[-1]
            if pd.notna(rsi_val):
                rsi = float(rsi_val)
        except Exception:
            rsi = None

        rows.append(
            dict(
                ticker=t,
                close=close,
                avg_dollar_vol_20=adv,
                mom_1m=mom_1m,
                mom_3m=mom_3m,
                mom_6m=mom_6m,
                mom_12m=mom_12m,
                above_ma20=above_ma20,
                above_ma50=above_ma50,
                above_ma200=above_ma200,
                dist_from_52w_high=dist_52w,
                max_drawdown_6m=dd_6m,
                rsi_14=rsi,
            )
        )

    df = pd.DataFrame(rows)
    return df


def score_short_term(df: pd.DataFrame) -> pd.DataFrame:
    """
    0–100 short-term score (momentum + trend + liquidity + controlled drawdown).
    """
    out = df.copy()
    score = np.zeros(len(out), dtype=float)

    # Liquidity (15): >=$20M/day gets full points (scaled)
    adv = out["avg_dollar_vol_20"].fillna(0.0)
    score += adv.apply(lambda x: 15 * _clip01(x / 20_000_000))

    # Momentum (45)
    # 1m: 0..20% => 0..10
    m1 = out["mom_1m"].fillna(0.0)
    score += m1.apply(lambda x: 10 * _clip01((x - 0.00) / 0.20))

    # 3m: -10..+30 => 0..15
    m3 = out["mom_3m"].fillna(0.0)
    score += m3.apply(lambda x: 15 * _clip01((x + 0.10) / 0.40))

    # 6m: -10..+50 => 0..15
    m6 = out["mom_6m"].fillna(0.0)
    score += m6.apply(lambda x: 15 * _clip01((x + 0.10) / 0.60))

    # 12m: -20..+80 => 0..5
    m12 = out["mom_12m"].fillna(0.0)
    score += m12.apply(lambda x: 5 * _clip01((x + 0.20) / 1.00))

    # Trend (25)
    score += out["above_ma20"].fillna(False).astype(int) * 7
    score += out["above_ma50"].fillna(False).astype(int) * 10
    score += out["above_ma200"].fillna(False).astype(int) * 8

    # Risk control (15)
    # Drawdown 6m: 0..50% => 15..0 (lower is better)
    dd6 = out["max_drawdown_6m"].fillna(0.50)
    score += dd6.apply(lambda x: 15 * _clip01(1 - (x / 0.50)))

    out["short_score"] = np.round(score, 1)
    return out


def apply_short_term_filters(df: pd.DataFrame, p: ShortTermParams) -> pd.DataFrame:
    out = df.copy()

    if p.min_avg_dollar_vol is not None:
        out = out[out["avg_dollar_vol_20"].fillna(0) >= p.min_avg_dollar_vol]

    if p.require_above_ma50:
        out = out[out["above_ma50"] == True]
    if p.require_above_ma200:
        out = out[out["above_ma200"] == True]

    out = out[out["max_drawdown_6m"].fillna(999) <= p.max_dd_6m]
    out = out[out["mom_3m"].fillna(-999) >= p.min_mom_3m]
    out = out[out["mom_6m"].fillna(-999) >= p.min_mom_6m]

    out = out[out["dist_from_52w_high"].fillna(999) <= p.max_dist_from_52w_high]
    return out
