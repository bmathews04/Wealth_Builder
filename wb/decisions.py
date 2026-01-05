# wb/decisions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecisionConfig:
    # Chase risk thresholds based on % above MA50 (~10-week proxy)
    chase_green_max: float = 0.10   # <= 10%
    chase_yellow_max: float = 0.25  # 10–25%

    # Entry zones as % above moving averages
    entry_ma50_max: float = 0.05    # pullback entry zone: up to +5% above MA50
    entry_ma40w_max: float = 0.03   # secondary zone: up to +3% above MA40W

    # Management defaults (process rules)
    trim_ext_ma50: float = 0.30     # trim if >30% above MA50
    trail_gain: float = 0.20        # after +20%, trail to MA50 (weekly proxy)


def _safe_float(x) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _latest_row(prices: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    g = prices[prices["ticker"] == ticker].sort_values("date")
    if g.empty:
        return None
    return g.iloc[-1]


def _weekly_ma(prices: pd.DataFrame, ticker: str, weeks: int = 10) -> float:
    """
    Weekly MA from weekly closes (W-FRI).
    Uses 'close' column from prices long frame.
    """
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
    """
    Long-term decision state.
    BUY: trend confirmed + RS ok + not extended/chase-red
    WAIT: good name but extended/chase-red
    WATCH: early/bottom-fishing without full confirmation
    AVOID: trend broken
    """
    stype = str(setup_type or "").strip().lower()

    # Hard avoid
    if stype == "avoid":
        return "AVOID"

    trend_ok = bool(above_ma200) and bool(above_ma40w)
    rs_ok = (rs_12m is None) or (not pd.isna(rs_12m) and float(rs_12m) >= 0.0)

    if trend_ok and rs_ok:
        if chase_label.startswith("🔴"):
            return "WAIT"
        if stype == "extended":
            return "WAIT"
        return "BUY"

    if stype in {"early trend", "bottom-fishing"}:
        return "WATCH"

    return "AVOID"


def build_entry_invalidation_targets(
    ma50: float,
    ma40w: float,
    high_52w: float,
    cfg: DecisionConfig,
) -> Dict[str, str]:
    """
    Trader-facing guidance (process-based). Not predictive.
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
      - ticker
      - setup_type (from wb.setups)
      - above_ma200, above_ma40w (from wb.metrics/scoring)
    Optional:
      - rs_vs_spy_12m OR rs_12m
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
        action_label(
            setup_type=st,
            above_ma200=a200,
            above_ma40w=a40w,
            rs_12m=(None if rs_col is None else rsv),
            chase_label=cr,
        )
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

    # Helpful text field
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

# =========================
# SHORT-TERM DECISION LAYER
# =========================

def _get_last_price_row(prices: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    g = prices[prices["ticker"] == ticker].sort_values("date")
    if g.empty:
        return None
    return g.iloc[-1]


def _fmt_price(x) -> str:
    return "—" if pd.isna(x) else f"${float(x):,.2f}"


def _fmt_pct(x) -> str:
    return "—" if pd.isna(x) else f"{float(x)*100:.1f}%"


def short_term_chase_risk(
    pct_above_ma20: float,
    dist_from_52w_high: float,
) -> str:
    """
    Two quick 'don't chase' signals:
    - too extended above MA20
    - too close to 52w highs (breakout is fine; chasing is not)
    """
    if pd.isna(pct_above_ma20) and pd.isna(dist_from_52w_high):
        return "—"

    # If either is missing, decide from what we have
    pa = pct_above_ma20
    d52 = dist_from_52w_high

    # Conservative: if very extended or at highs -> high chase risk
    if (not pd.isna(pa) and pa >= 0.12) or (not pd.isna(d52) and d52 <= 0.05):
        return "🔴 High"
    if (not pd.isna(pa) and pa >= 0.06) or (not pd.isna(d52) and d52 <= 0.12):
        return "🟡 Medium"
    return "🟢 Low"


def short_term_action(
    above_ma50: Optional[bool],
    mom_3m: Optional[float],
    mom_6m: Optional[float],
    max_dd_6m: Optional[float],
    chase_label: str,
) -> str:
    """
    Short-term is more tactical:
      BUY  = trend + momentum + not too ugly dd + not chase-red
      WAIT = good momentum but chase-red / too extended
      WATCH = improving but not fully confirmed
      AVOID = trend broken
    """
    if above_ma50 is False:
        return "AVOID"

    m3 = np.nan if mom_3m is None else float(mom_3m)
    m6 = np.nan if mom_6m is None else float(mom_6m)
    dd = np.nan if max_dd_6m is None else float(max_dd_6m)

    mom_ok = (not pd.isna(m3) and m3 > 0) and (not pd.isna(m6) and m6 > 0)
    dd_ok = (pd.isna(dd) or dd <= 0.45)

    if above_ma50 and mom_ok and dd_ok:
        if chase_label.startswith("🔴"):
            return "WAIT"
        return "BUY"

    # If above MA50 but momentum mixed -> WATCH
    if above_ma50:
        return "WATCH"

    return "AVOID"


def build_short_term_plan_text(
    close: float,
    ma20: float,
    ma50: float,
    high_52w: float,
    cfg: DecisionConfig,
) -> Dict[str, str]:
    """
    Practical, process-based suggestions (not predictive):
    - Entries: pullback to MA20/MA50 or breakout above 52w high
    - Stops: tactical under MA50 (or MA20 for tighter)
    - Targets: optional 'prior high' / '2R' language kept simple
    """
    out: Dict[str, str] = {}

    # Entry: MA20 pullback
    if not pd.isna(ma20):
        out["st_entry_pullback"] = f"Pullback entry: {_fmt_price(ma20)} to {_fmt_price(ma20 * 1.02)} (MA20 to +2%)"
    else:
        out["st_entry_pullback"] = "Pullback entry: —"

    # Entry: MA50 support
    if not pd.isna(ma50):
        out["st_entry_support"] = f"Support entry: {_fmt_price(ma50)} to {_fmt_price(ma50 * 1.03)} (MA50 to +3%)"
    else:
        out["st_entry_support"] = "Support entry: —"

    # Entry: breakout
    if not pd.isna(high_52w):
        out["st_entry_breakout"] = f"Breakout entry: weekly close above {_fmt_price(high_52w)} (52w high)"
    else:
        out["st_entry_breakout"] = "Breakout entry: —"

    # Stops
    if not pd.isna(ma50):
        out["st_stop"] = f"Stop idea: close below MA50 ({_fmt_price(ma50)})"
    elif not pd.isna(ma20):
        out["st_stop"] = f"Stop idea: close below MA20 ({_fmt_price(ma20)})"
    else:
        out["st_stop"] = "Stop idea: —"

    # Simple target framing
    if not pd.isna(high_52w) and not pd.isna(close):
        # if below highs, target is retest
        if close < high_52w:
            out["st_target"] = f"Target idea: retest {_fmt_price(high_52w)} (prior high)"
        else:
            out["st_target"] = "Target idea: trail winners (raise stop with MA20/MA50)"
    else:
        out["st_target"] = "Target idea: —"

    return out


def add_decisions_short_term(
    st_df: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds actionable short-term decision columns to the short-term table.

    Expects st_df columns from wb.short_term.compute_short_term_table:
      ticker, close, mom_3m, mom_6m, above_ma50, dist_from_52w_high, max_drawdown_6m, etc.

    Uses prices (long form) to fetch ma20/ma50 and 52w high for text guidance.
    """
    if st_df is None or st_df.empty:
        return st_df

    out = st_df.copy()

    # Pull MA levels + 52w highs for each ticker
    rows = []
    for t in out["ticker"].astype(str).tolist():
        last = _get_last_price_row(prices, t)
        if last is None:
            rows.append(
                dict(
                    ticker=t,
                    st_ma20=np.nan,
                    st_ma50=np.nan,
                    st_high_52w=np.nan,
                    st_pct_above_ma20=np.nan,
                )
            )
            continue

        close = _safe_float(last.get("close"))
        ma20 = _safe_float(last.get("ma20"))
        ma50 = _safe_float(last.get("ma50"))

        high_52w = _rolling_max_close(prices, t, window_days=252)

        pct_above_ma20 = np.nan
        if not pd.isna(close) and not pd.isna(ma20) and ma20 != 0:
            pct_above_ma20 = (close / ma20) - 1.0

        rows.append(
            dict(
                ticker=t,
                st_ma20=ma20,
                st_ma50=ma50,
                st_high_52w=high_52w,
                st_pct_above_ma20=pct_above_ma20,
            )
        )

    aux = pd.DataFrame(rows)
    out = out.merge(aux, on="ticker", how="left")

    # Chase risk
    out["st_chase_risk"] = [
        short_term_chase_risk(pa, d52)
        for pa, d52 in zip(out["st_pct_above_ma20"], out.get("dist_from_52w_high", pd.Series([np.nan] * len(out))))
    ]

    # Action
    out["st_action"] = [
        short_term_action(a50, m3, m6, dd6, cr)
        for a50, m3, m6, dd6, cr in zip(
            out.get("above_ma50", pd.Series([None] * len(out))),
            out.get("mom_3m", pd.Series([None] * len(out))),
            out.get("mom_6m", pd.Series([None] * len(out))),
            out.get("max_drawdown_6m", pd.Series([None] * len(out))),
            out["st_chase_risk"],
        )
    ]

    # Plan text fields
    plans = [
        build_short_term_plan_text(
            close=_safe_float(c),
            ma20=_safe_float(ma20),
            ma50=_safe_float(ma50),
            high_52w=_safe_float(h52),
            cfg=DecisionConfig(),
        )
        for c, ma20, ma50, h52 in zip(
            out.get("close", pd.Series([np.nan] * len(out))),
            out["st_ma20"],
            out["st_ma50"],
            out["st_high_52w"],
        )
    ]
    plan_df = pd.DataFrame(plans)
    out = pd.concat([out.reset_index(drop=True), plan_df.reset_index(drop=True)], axis=1)

    # Helpful text columns for table
    out["st_pct_above_ma20_text"] = np.where(
        out["st_pct_above_ma20"].notna(),
        (out["st_pct_above_ma20"] * 100).round(1).astype(str) + "%",
        "—",
    )

    return out


def decision_card_short_term(row: Dict) -> Dict[str, str]:
    return {
        "Action": row.get("st_action", "—"),
        "Chase risk": row.get("st_chase_risk", "—"),
        "Pullback entry (MA20)": row.get("st_entry_pullback", "—"),
        "Support entry (MA50)": row.get("st_entry_support", "—"),
        "Breakout entry": row.get("st_entry_breakout", "—"),
        "Stop idea": row.get("st_stop", "—"),
        "Target idea": row.get("st_target", "—"),
    }


