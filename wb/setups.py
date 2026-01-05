# wb/setups.py
from __future__ import annotations

from typing import Optional, List
import numpy as np
import pandas as pd

SETUP_ORDER = ["Leader", "Early Trend", "Bottom-Fishing", "Extended", "Avoid"]


def _get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
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
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"]).fillna(default)


def classify_setups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - setup_type: one of [Leader, Early Trend, Bottom-Fishing, Extended, Avoid]
      - setup_reason: compact text explaining why

    Deterministic labeler. Does NOT change ranking.
    Works in both modes:
      - Short-term uses short_score if present
      - Long-term falls back to score if short_score missing
    """
    out = df.copy()

    # ---- Column discovery ----
    col_score = _get_col(out, ["short_score", "score_short", "shortScore", "score"])
    col_rsi = _get_col(out, ["rsi14", "rsi_14", "rsi"])
    col_mom_1m = _get_col(out, ["mom_1m", "mom1m", "return_1m"])
    col_mom_3m = _get_col(out, ["mom_3m", "mom3m", "return_3m"])
    col_mom_6m = _get_col(out, ["mom_6m", "mom6m", "return_6m"])
    col_mom_12m = _get_col(out, ["mom_12m", "mom12m", "return_12m", "mom_1y"])

    col_dd_6m = _get_col(out, ["max_dd_6m", "max_drawdown_6m", "dd_6m"])
    col_dd_2y = _get_col(out, ["max_dd_2y", "max_drawdown_2y", "dd_2y", "max_drawdown_24m", "max_drawdown_2y"])

    col_dist_52w = _get_col(out, ["dist_from_52w_high", "dist_52w_high", "dist_52w"])
    col_rs_12m = _get_col(out, ["rs_12m", "rs_vs_spy_12m", "rel_strength_12m"])
    col_rs_slope = _get_col(out, ["rs_slope_50d", "rs_slope", "rs_trend"])

    # Trend booleans (match your app’s names too)
    col_above_200d = _get_col(out, ["above_ma200", "above_200d", "price_above_200d", "above_200dma"])
    col_above_40w = _get_col(out, ["above_ma40w", "above_40w", "price_above_40w", "above_40wma"])
    col_above_50d = _get_col(out, ["above_ma50", "above_50d", "price_above_50d", "above_50dma"])
    col_above_20d = _get_col(out, ["above_ma20", "above_20d", "price_above_20d", "above_20dma"])

    # ---- Pull series safely ----
    score = _safe_series(out, col_score)
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

    above_long = (above_200d | above_40w)

    # normalize drawdowns / distances
    dd2y_mag = dd_2y.abs()
    dd6m_mag = dd_6m.abs()
    dist52 = dist_52w.abs()

    # RS trend
    rs_improving = pd.Series(False, index=out.index)
    if col_rs_slope is not None:
        rs_improving = rs_slope > 0
    elif col_rs_12m is not None:
        rs_improving = rs_12m > 0

    # ---- Setup Rules ----
    # Extended: strong score + high RSI + very near highs
    is_extended = (
        (score >= 80)
        & (rsi >= 70)
        & (dist52 <= 0.05)
    )

    # Leader: strong score + trend + improving RS + controlled long drawdown
    is_leader = (
        (score >= 75)
        & above_long
        & rs_improving
        & (dd2y_mag.fillna(0.0) <= 0.30)
        & (~is_extended)
    )

    # Early Trend: mid-high score + trend improving
    is_early = (
        (score >= 55) & (score < 75)
        & (above_50d | above_20d | above_long)
        & rs_improving
        & (~is_extended)
        & (~is_leader)
    )

    # Bottom-Fishing: prior damage + early turn signs (when available)
    mom_turn = pd.Series(True, index=out.index)
    if col_mom_12m is not None:
        mom_turn &= (mom_12m <= 0)
    if col_mom_3m is not None:
        mom_turn &= (mom_3m > 0)
    if col_mom_1m is not None:
        mom_turn &= (mom_1m > 0)

    reclaimed_short_ma = (above_20d | above_50d)

    is_bottom = (
        (score >= 30) & (score <= 55)
        & (dd2y_mag.fillna(0.0) >= 0.30) & (dd2y_mag.fillna(0.0) <= 0.70)
        & mom_turn
        & (rsi.fillna(50) >= 35) & (rsi.fillna(50) <= 60)
        & reclaimed_short_ma
        & (~is_extended)
        & (~is_leader)
        & (~is_early)
    )

    # Avoid: no trend + weak score (or falling knife-ish)
    is_avoid = (
        (score < 30)
        | ((~above_long) & (score < 45) & (~is_bottom))
    )

    setup = pd.Series("Avoid", index=out.index)
    setup[is_extended] = "Extended"
    setup[is_leader] = "Leader"
    setup[is_early] = "Early Trend"
    setup[is_bottom] = "Bottom-Fishing"
    setup[is_avoid] = "Avoid"

    # ---- Reasons ----
    reasons = []
    for i in out.index:
        s = setup.loc[i]
        parts = []
        if not np.isnan(score.loc[i]):
            parts.append(f"score={score.loc[i]:.0f}")
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

        hint = {
            "Leader": "Trend leader",
            "Early Trend": "Trend emerging",
            "Bottom-Fishing": "Base/turn attempt",
            "Extended": "Stretched near highs",
            "Avoid": "No edge",
        }.get(s, "No edge")

        reasons.append(f"{hint} ({', '.join(parts)})")

    out["setup_type"] = pd.Categorical(setup.astype(str), categories=SETUP_ORDER, ordered=True)
    out["setup_reason"] = pd.Series(reasons, index=out.index)
    return out
