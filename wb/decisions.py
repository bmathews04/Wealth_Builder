# wb/decisions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecisionConfig:
    # Chase risk thresholds based on % above MA50 (proxy for ~10-week trend)
    chase_green_max: float = 0.10   # <= 10%
    chase_yellow_max: float = 0.25  # 10–25%

    # Entry zones as % above moving averages
    entry_ma50_max: float = 0.05    # pullback entry zone: up to +5% above MA50
    entry_ma40w_max: float = 0.03   # secondary zone: up to +3% above MA40W

    # Management defaults (process rules)
    trim_ext_ma50: float = 0.30     # trim if >30% above MA50
    trail_gain: float = 0.20        # after +20%, trail to MA50 (weekly proxy)


def _latest_row(prices: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    g = prices[prices["ticker"] == ticker].sort_values("date")
    if g.empty:
        return None
    return g.iloc[-1]


def _weekly_ma(prices: pd.DataFrame, ticker: str, weeks: int = 10) -> float:
    """Weekly MA from weekly closes (W-FRI), returns latest rolling mean."""
    g = prices[prices["ticker"] == ticker].set_index("date").sort_index()
    if g.empty:
        return np.nan
    wk_close = g["close"].resample("W-FRI").last().dropna()
    if wk_close.empty:
        return np.nan
    return float(wk_close.rolling(weeks).mean().iloc[-1])


def _rolling_max_close(prices: pd.DataFrame, ticker: str, window_days: int = 252) -> float:
    g = prices[prices["ticker"] == ticker].sort_values("date")
    if g.empty:
        return np.nan
    s = g["close"].tail(window_days)
    if s.empty:
        return np.nan
    return float(s.max())


def _safe_float(x) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def chase_risk_label(pct_above_ma50: float, cfg: DecisionConfig) -> str:
    if pd.isna(pct_above_ma50):
        return "—"
    if pct_above_ma50 <= cfg.chase_green_max:
        return "🟢 Low"
    if pct_above_ma50 <= cfg.chase_yellow_max:
        return "🟡 Medium"
    return "🔴 High"


def action_label(
    setup_type: str,
    above_ma200: Optional[bool],
    above_ma40w: Optional[bool],
    rs_12m: Optional[float],
    chase_label: str,
) -> str:
    st = str(setup_type or "").strip().lower()

    # Hard avoid
    if st == "avoid":
        return "AVOID"

    trend_ok = bool(above_ma200) and bool(above_ma40w)
    rs_ok = (rs_12m is None) or (not pd.isna(rs_12m) and float(rs_12m) >= 0.0)

    if trend_ok and rs_ok:
        if chase_label.startswith("🔴"):
            return "WAIT"
        if st == "extended":
            return "WAIT"
        return "BUY"

    if st in {"early trend", "bottom-fishing"}:
        return "WATCH"

    return "AVOID"


def build_entry_invalidation_targets(
    ma50: float,
    ma40w: float,
    high_52w: float,
    cfg: DecisionConfig,
) -> Dict[str, str]:
    """
    Trader-facing guidance (process-based).
    """
    out: Dict[str, str] = {}

    # Entry zones
    if not pd.isna(ma50):
        lo = ma50
        hi = ma50 * (1 + cfg.entry_ma50_max)
        out["entry_pullback"] = f"Pullback zone: ${lo:,.2f} to ${hi:,.2f} (MA50 to +{cfg.entry_ma50_max*100:.0f}%)"
    else:
        out["entry_pullback"] = "Pullback zone: —"

    if not pd.isna(ma40w):
        lo = ma40w
        hi = ma40w * (1 + cfg.entry_ma40w_max)
        out["entry_trend_floor"] = f"Trend zone: ${lo:,.2f} to ${hi:,.2f} (MA40W to +{cfg.entry_ma40w_max*100:.0f}%)"
    else:
        out["entry_trend_floor"] = "Trend zone: —"

    if not pd.isna(high_52w):
        out["entry_breakout"] = f"Breakout idea: weekly close above ${high_52w:,.2f} (52w high)"
    else:
        out["entry_breakout"] = "Breakout idea: —"

    # Invalidation (default long-term: MA40W)
    if not pd.isna(ma40w):
        out["invalidation"] = f"Invalidation: weekly close below MA40W (${ma40w:,.2f})"
    elif not pd.isna(ma50):
        out["invalidation"] = f"Invalidation: close below MA50 (${ma50:,.2f})"
    else:
        out["invalidation"] = "Invalidation: —"

    # Management
    if not pd.isna(ma50):
        trim_level = ma50 * (1 + cfg.trim_ext_ma50)
        out["management"] = (
            f"Management: consider trimming if > ${trim_level:,.2f} "
            f"(~+{cfg.trim_ext_ma50*100:.0f}% above MA50); trail stop to MA50 after +{cfg.trail_gain*100:.0f}% gain."
        )
    else:
        out["management"] = "Management: —"

    return out


def add_decisions_long_term(
    df: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: DecisionConfig = DecisionConfig(),
) -> pd.DataFrame:
    """
    Adds decision columns to a long-term metrics/scoring table.
    Requires df columns:
      ticker, setup_type, above_ma200, above_ma40w
    Optional:
      rs_vs_spy_12m or rs_12m
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    rs_col = "rs_vs_spy_12m" if "rs_vs_spy_12m" in out.columns else ("rs_12m" if "rs_12m" in out.columns else None)

    rows = []
    for t in out["ticker"].astype(str).tolist():
        last = _latest_row(prices, t)
        if last is None:
            rows.append(
                dict(
                    ticker=t,
                    last_close=np.nan,
                    ma50=np.nan,
                    ma40w=np.nan,
                    ma10w=np.nan,
                    high_52w=np.nan,
                    pct_above_ma50=np.nan,
                    chase_risk="—",
                )
            )
            continue

        last_close = _safe_float(last.get("close"))
        ma50 = _safe_float(last.get("ma50"))
        ma40w = _safe_float(last.get("ma40w"))

        ma10w = _weekly_ma(prices, t, weeks=10)
        high_52w = _rolling_max_close(prices, t, window_days=252)

        pct_above_ma50 = np.nan
        if not pd.isna(last_close) and not pd.isna(ma50) and ma50 != 0:
            pct_above_ma50 = (last_close / ma50) - 1.0

        rows.append(
            dict(
                ticker=t,
                last_close=last_close,
                ma50=ma50,
                ma40w=ma40w,
                ma10w=ma10w,
                high_52w=high_52w,
                pct_above_ma50=pct_above_ma50,
                chase_risk=chase_risk_label(pct_above_ma50, cfg),
            )
        )

    d = pd.DataFrame(rows)
    out = out.merge(d, on="ticker", how="left")

    # Action
    rs_vals = out[rs_col] if rs_col else pd.Series([np.nan] * len(out), index=out.index)
    out["action"] = [
        action_label(setup_type=st, above_ma200=a200, above_ma40w=a40w, rs_12m=(None if rs_col is None else rsv), chase_label=cr)
        for st, a200, a40w, rsv, cr in zip(
            out.get("setup_type", ""),
            out.get("above_ma200", False),
            out.get("above_ma40w", False),
            rs_vals,
            out["chase_risk"],
        )
    ]

    # Entry/exit guidance (text)
    cards = [
        build_entry_invalidation_targets(ma50, ma40w, h52, cfg)
        for ma50, ma40w, h52 in zip(out["ma50"], out["ma40w"], out["high_52w"])
    ]
    card_df = pd.DataFrame(cards)
    out = pd.concat([out.reset_index(drop=True), card_df.reset_index(drop=True)], axis=1)

    # Helpful numeric formatting field
    out["pct_above_ma50_text"] = np.where(
        out["pct_above_ma50"].notna(),
        (out["pct_above_ma50"] * 100).round(1).astype(str) + "%",
        "—",
    )

    return out


def decision_card(row: Dict) -> Dict[str, str]:
    """Compact payload for st.write() in deep dive."""
    return {
        "Action": row.get("action", "—"),
        "Chase risk": row.get("chase_risk", "—"),
        "Pullback entry": row.get("entry_pullback", "—"),
        "Trend entry": row.get("entry_trend_floor", "—"),
        "Breakout entry": row.get("entry_breakout", "—"),
        "Invalidation": row.get("invalidation", "—"),
        "Management": row.get("management", "—"),
    }
