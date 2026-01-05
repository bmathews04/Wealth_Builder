# wb/setups.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


SETUP_ORDER = ["Leader", "Early Trend", "Bottom-Fishing", "Extended", "Avoid"]


def _get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_series(df: pd.DataFrame, col: Optional[str], default=np.nan) -> pd.Series:
    if col is None:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _bool_series(df: pd.DataFrame, col: Optional[str], default=False) -> pd.Series:
    if col is None:
        return pd.Series([default] * len(df), index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(default)
    # coerce common representations
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"]).fillna(default)


def classify_setups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - setup_type: one of [Leader, Early Trend, Bottom-Fishing, Extended, Avoid]
      - setup_reason: compact text explaining why

    This is deterministic and does NOT change your ranking; it only labels each row.
    """
    out = df.copy()

    # ---- Column discovery (flexible to your schema) ----
    col_score = _get_col(out, ["short_score", "score_short", "shortScore"])
    col_rsi = _get_col(out, ["rsi14", "rsi_14", "rsi"])
    col_mom_1m = _get_col(out, ["mom_1m", "mom1m", "return_1m"])
    col_mom_3m = _get_col(out, ["mom_3m", "mom3m", "return_3m"])
    col_mom_6m = _get_col(out, ["mom_6m", "mom6m", "return_6m"])
    col_mom_12m = _get_col(out, ["mom_12m", "mom12m", "return_12m", "mom_1y"])

    col_dd_6m = _get_col(out, ["max_dd_6m", "max_drawdown_6m", "dd_6m"])
    col_dd_2y = _get_col(out, ["max_dd_2y", "max_drawdown_2y", "dd_2y", "max_drawdown_24m"])

    col_dist_52w = _get_col(out, ["dist_from_52w_high", "dist_52w_high", "dist_52w"])
    col_rs_12m = _get_col(out, ["rs_12m", "rs_vs_spy_12m", "rel_strength_12m"])
    col_rs_slope = _get_col(out, ["rs_slope_50d", "rs_slope", "rs_trend"])

    # Trend booleans (if you already compute these)
    col_above_200d = _get_col(out, ["above_200d", "price_above_200d", "above_200dma"])
    col_above_40w = _get_col(out, ["above_40w", "price_above_40w", "above_40wma"])
    col_above_50d = _get_col(out, ["above_50d", "price_above_50d", "above_50dma"])
    col_above_20d = _get_col(out, ["above_20d", "price_above_20d", "above_20dma"])

    # ---- Pull series safely ----
    short_score = _safe_series(out, col_score)
    rsi = _safe_series(out, col_rsi)
    mom_1m = _safe_series(out, col_mom_1m)
    mom_3m = _safe_series(out, col_mom_3m)
    mom_6m = _safe_series(out, col_mom_6m)
    mom_12m = _safe_series(out, col_mom_12m)

    dd_6m = _safe_series(out, col_dd_6m)
    dd_2y = _safe_series(out, col_dd_2y)

    dist_52w = _safe_series(out, col_dist_52w)
    rs_12m = _safe_series(out, col_rs_12m)
    rs_slope = _safe_series(out, col_rs_slope)

    above_200d = _bool_series(out, col_above_200d, default=False)
    above_40w = _bool_series(out, col_above_40w, default=False)
    above_50d = _bool_series(out, col_above_50d, default=False)
    above_20d = _bool_series(out, col_above_20d, default=False)

    # If you don’t have explicit above_200d/40w booleans, treat missing as False.
    above_long = (above_200d | above_40w)

    # Normalize drawdown signs:
    # Many implementations store drawdowns as negative numbers (e.g. -0.30).
    # We want magnitude in positive terms for thresholding.
    dd2y_mag = dd_2y.abs()
    dd6m_mag = dd_6m.abs()

    # Distance from 52w high:
    # If stored as 0.00 at highs and positive % away, keep it.
    # If stored negative near highs, use abs.
    dist52 = dist_52w.abs()

    # RS trend heuristic: if rs_slope exists use it; else infer from rs_12m positive.
    rs_improving = pd.Series(False, index=out.index)
    if col_rs_slope is not None:
        rs_improving = rs_slope > 0
    elif col_rs_12m is not None:
        rs_improving = rs_12m > 0

    # ---- Setup Rules ----
    # Extended: strong momentum but stretched (avoid chasing)
    is_extended = (
        (short_score >= 80)
        & (rsi >= 70)
        & (dist52 <= 0.05)  # within 5% of 52w high
    )

    # Leader: trend + strong score + improving RS + controlled drawdown
    is_leader = (
        (short_score >= 75)
        & above_long
        & rs_improving
        & (dd2y_mag <= 0.30)
        & (~is_extended)
    )

    # Early Trend: emerging leaders (best risk-adjusted zone)
    # Allow either: above_50d or above_20d AND score mid-high AND RS improving
    is_early = (
        (short_score >= 55) & (short_score < 75)
        & (above_50d | above_20d | above_long)
        & rs_improving
        & (~is_extended)
        & (~is_leader)
    )

    # Bottom-Fishing: prior damage, now stabilizing/improving
    # Key: 12m <= 0, 3m > 0, 1m > 0, RSI 35–55 (rising-ish), reclaimed short MA
    # Use what you have; if mom cols missing, condition relaxes gracefully.
    mom_turn = pd.Series(True, index=out.index)
    if col_mom_12m is not None:
        mom_turn &= (mom_12m <= 0)
    if col_mom_3m is not None:
        mom_turn &= (mom_3m > 0)
    if col_mom_1m is not None:
        mom_turn &= (mom_1m > 0)

    reclaimed_short_ma = (above_20d | above_50d)
    is_bottom = (
        (short_score >= 30) & (short_score <= 50)
        & (dd2y_mag >= 0.30) & (dd2y_mag <= 0.60)
        & mom_turn
        & (rsi >= 35) & (rsi <= 55)
        & reclaimed_short_ma
        & (~is_extended)
        & (~is_leader)
        & (~is_early)
    )

    # Avoid: falling knife / no edge
    is_avoid = (
        (short_score < 30)
        | ((~above_long) & (short_score < 45) & (~is_bottom))
    )

    setup = pd.Series("Avoid", index=out.index)
    setup[is_extended] = "Extended"
    setup[is_leader] = "Leader"
    setup[is_early] = "Early Trend"
    setup[is_bottom] = "Bottom-Fishing"
    setup[is_avoid] = "Avoid"

    # ---- Reasons (compact, useful for tooltips) ----
    reasons = []
    for i in out.index:
        s = setup.loc[i]
        parts = []
        if not np.isnan(short_score.loc[i]):
            parts.append(f"short={short_score.loc[i]:.0f}")
        if not np.isnan(rsi.loc[i]):
            parts.append(f"RSI={rsi.loc[i]:.0f}")
        if col_mom_12m is not None and not np.isnan(mom_12m.loc[i]):
            parts.append(f"12m={mom_12m.loc[i]*100:.0f}%")
        if col_mom_3m is not None and not np.isnan(mom_3m.loc[i]):
            parts.append(f"3m={mom_3m.loc[i]*100:.0f}%")
        if col_mom_1m is not None and not np.isnan(mom_1m.loc[i]):
            parts.append(f"1m={mom_1m.loc[i]*100:.0f}%")
        if not np.isnan(dd2y_mag.loc[i]):
            parts.append(f"DD2y={dd2y_mag.loc[i]*100:.0f}%")
        if not np.isnan(dist52.loc[i]):
            parts.append(f"dist52={dist52.loc[i]*100:.0f}%")
        if above_long.loc[i]:
            parts.append(">200D/40W")
        elif reclaimed_short_ma.loc[i]:
            parts.append(">20D/50D")

        # Human label hint
        if s == "Leader":
            hint = "Trend leader"
        elif s == "Early Trend":
            hint = "Trend emerging"
        elif s == "Bottom-Fishing":
            hint = "Base/turn attempt"
        elif s == "Extended":
            hint = "Stretched near highs"
        else:
            hint = "No edge"

        reasons.append(f"{hint} ({', '.join(parts)})")

    out["setup_type"] = setup.astype(str)
    out["setup_reason"] = pd.Series(reasons, index=out.index)

    # Keep a stable categorical order if you want sorting/grouping
    out["setup_type"] = pd.Categorical(out["setup_type"], categories=SETUP_ORDER, ordered=True)

    return out
