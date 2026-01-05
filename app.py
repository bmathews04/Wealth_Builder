# app.py
import hashlib
from datetime import datetime, timezone

import streamlit as st
import pandas as pd

from wb.universe import get_universe, UNIVERSE_MODULE_VERSION
from wb.prices import batch_fetch_prices
from wb.setups import classify_setups, SETUP_ORDER
from wb.fundamentals import fetch_fundamentals_parallel
from wb.metrics import build_metrics_table
from wb.scoring import score_table, add_data_quality_badges

from wb.charts import (
    make_weekly_candles,
    make_relative_strength,
    make_drawdown_chart,
    make_margin_trends,
    make_cashflow_trends,
)

from wb.short_term import (
    ShortTermParams,
    compute_short_term_table,
    score_short_term,
    apply_short_term_filters,
)

from wb.backtest import backtest_equal_weight, equity_curve_chart
from wb.backtest_rebalance_v2 import backtest_rebalanced_v2

from wb.risk import RiskParams, build_trade_plan
from wb.risk_v2 import (
    correlation_filter,
    portfolio_heat,
    enforce_heat_cap,
)

from wb.snapshots import (
    make_snapshot,
    merge_snapshots,
    snapshots_to_bytes,
    snapshots_from_bytes,
)
from wb.causal_ranker import ranker_from_snapshots

# ✅ NEW: decision layer (create wb/decisions.py as provided)
from wb.decisions import add_decisions_long_term, decision_card


st.set_page_config(page_title="Wealth Builder Screener", layout="wide")

st.title("Wealth Builder Screener")
st.caption("Mode toggle: Long-term (durability) vs Short-term (momentum). Backtests + risk tools included.")


# ----------------------------
# Caching (fast + stable)
# ----------------------------
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def cached_prices(tickers_for_prices: tuple[str, ...], years: int) -> pd.DataFrame:
    return batch_fetch_prices(list(tickers_for_prices), years=years)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def cached_fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    # Keep workers fixed inside cache so changing slider doesn't bust the cache.
    return fetch_fundamentals_parallel(list(tickers), max_workers=8)


def _now_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _stable_hash(obj) -> str:
    s = repr(obj).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


# ----------------------------
# Utilities
# ----------------------------
def pct(x):
    return "—" if pd.isna(x) else f"{x * 100:.1f}%"


def num(x):
    if pd.isna(x):
        return "—"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:.2f}B"
    if ax >= 1e6:
        return f"{x/1e6:.1f}M"
    if ax >= 1e3:
        return f"{x/1e3:.1f}K"
    return f"{x:.0f}"


def truthy_badge(x):
    return "✅" if bool(x) else "—"


def _latest_snapshot_asof(prices_long: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    d = prices_long[prices_long["date"] <= pd.to_datetime(asof)].copy()
    if d.empty:
        return pd.DataFrame()
    d = d.sort_values(["ticker", "date"])
    snap = d.groupby("ticker", as_index=False).tail(1)
    return snap


def _causal_momentum_ranker_factory(
    prices_long: pd.DataFrame,
    universe: list[str],
    momentum_col: str = "ret_126d",
    min_adv: float = 20_000_000,
    require_above_ma50: bool = True,
    require_above_ma200: bool = False,
):
    def ranker(asof: pd.Timestamp) -> list[str]:
        snap = _latest_snapshot_asof(prices_long, asof)
        if snap.empty:
            return []

        snap = snap[snap["ticker"].isin(universe)].copy()

        if "avg_dollar_vol_20" in snap.columns:
            snap = snap[snap["avg_dollar_vol_20"].fillna(0) >= float(min_adv)]

        if require_above_ma50 and ("ma50" in snap.columns):
            snap = snap[(snap["close"] > snap["ma50"]) & snap["ma50"].notna()]
        if require_above_ma200 and ("ma200" in snap.columns):
            snap = snap[(snap["close"] > snap["ma200"]) & snap["ma200"].notna()]

        if momentum_col not in snap.columns:
            return []

        snap = snap[snap[momentum_col].notna()].sort_values(momentum_col, ascending=False)
        return snap["ticker"].astype(str).tolist()

    return ranker


def parse_tickers_from_inputs(universe_mode: str, tickers_text: str | None, uploaded) -> tuple[list[str], str | None]:
    tickers: list[str] = []

    try:
        if universe_mode == "Paste tickers":
            if tickers_text:
                tickers = [t.strip().upper() for t in tickers_text.replace("\n", ",").split(",") if t.strip()]

        elif universe_mode == "Upload CSV":
            if uploaded is None:
                return [], "Upload a CSV with a 'Ticker' column."
            dfu = pd.read_csv(uploaded)
            if "Ticker" not in dfu.columns:
                return [], "CSV must contain a 'Ticker' column."
            tickers = dfu["Ticker"].astype(str).str.upper().str.strip().tolist()

        elif universe_mode == "S&P 500 (ETF proxy: SPY)":
            tickers = get_universe("etf:SPY")

        elif universe_mode == "Nasdaq-100 (ETF proxy: QQQ)":
            tickers = get_universe("etf:QQQ")

        elif universe_mode == "Russell 1000 (ETF proxy: IWB)":
            tickers = get_universe("etf:IWB")

        elif universe_mode == "S&P 500 (Wikipedia)":
            tickers = get_universe("sp500")

        elif universe_mode == "Nasdaq 100 (Wikipedia)":
            tickers = get_universe("nasdaq100")

    except Exception as e:
        return [], f"Failed to load universe: {e}"

    tickers = [t for t in tickers if t and isinstance(t, str)]
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    tickers = sorted(list(dict.fromkeys(tickers)))  # stable dedupe
    return tickers, None


# ----------------------------
# Session state init
# ----------------------------
if "lt_snapshots" not in st.session_state:
    st.session_state["lt_snapshots"] = pd.DataFrame()

if "last_run_ts" not in st.session_state:
    st.session_state["last_run_ts"] = None

if "screen_error" not in st.session_state:
    st.session_state["screen_error"] = None

if "screen_params" not in st.session_state:
    st.session_state["screen_params"] = None

if "screen_hash" not in st.session_state:
    st.session_state["screen_hash"] = None


# ----------------------------
# Sidebar: FORM controls + refresh actions
# ----------------------------
with st.sidebar:
    st.caption(f"Universe module: {UNIVERSE_MODULE_VERSION}")

    # Refresh actions (outside form)
    cA, cB = st.columns(2)
    with cA:
        refresh_prices = st.button("Refresh prices", use_container_width=True)
    with cB:
        refresh_fund = st.button("Refresh fundamentals", use_container_width=True)

    if refresh_prices:
        cached_prices.clear()
        st.toast("Prices cache cleared. Click Run screen.", icon="🔄")

    if refresh_fund:
        cached_fundamentals.clear()
        st.toast("Fundamentals cache cleared. Click Run screen.", icon="🔄")

    st.divider()
    st.header("Controls")

    with st.form("screen_controls"):
        mode = st.selectbox("Select mode", ["Long-term", "Short-term"], index=0)

        st.header("Universe")
        universe_mode = st.selectbox(
            "Pick a universe",
            [
                "Paste tickers",
                "Upload CSV",
                "S&P 500 (ETF proxy: SPY)",
                "Nasdaq-100 (ETF proxy: QQQ)",
                "Russell 1000 (ETF proxy: IWB)",
                "S&P 500 (Wikipedia)",
                "Nasdaq 100 (Wikipedia)",
            ],
            index=0,
        )

        tickers_text = None
        uploaded = None
        if universe_mode == "Paste tickers":
            tickers_text = st.text_area(
                "Tickers (comma or newline separated)",
                value="AAPL, MSFT, AMZN, NVDA, GOOGL, META",
                height=120,
            )
        elif universe_mode == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV with a 'Ticker' column", type=["csv"])

        st.header("Setup Types")
        selected_setups = st.multiselect(
            "Show setups",
            options=SETUP_ORDER,
            default=["Leader", "Early Trend", "Bottom-Fishing", "Extended"],
        )

        st.header("Performance")
        max_workers = st.slider("Parallel workers (fundamentals)", 4, 32, 16, 1)
        throttle_safe_mode = st.checkbox("Throttle-safe mode", value=False)
        st.caption("Workers mainly affects *uncached* runs. Cached fundamentals are fetched with a fixed worker count.")

        # Short-term filters in the same form (so they don’t rerun until Run)
        st_params = None
        if mode == "Short-term":
            st.divider()
            st.header("Short-term filters")
            min_adv = st.number_input("Min Avg $ Volume (20d)", value=20_000_000, step=5_000_000)
            require_ma50 = st.checkbox("Require above MA50", value=True)
            require_ma200 = st.checkbox("Require above MA200", value=False)
            min_mom_3m = st.slider("Min 3m momentum", -0.50, 1.00, 0.00, 0.01)
            min_mom_6m = st.slider("Min 6m momentum", -0.50, 2.00, 0.00, 0.01)
            max_dd_6m = st.slider("Max drawdown (6m)", 0.10, 0.90, 0.45, 0.01)
            max_dist_high = st.slider("Max distance from 52w high", 0.05, 0.80, 0.25, 0.01)

            st_params = ShortTermParams(
                min_avg_dollar_vol=min_adv,
                require_above_ma50=require_ma50,
                require_above_ma200=require_ma200,
                min_mom_3m=min_mom_3m,
                min_mom_6m=min_mom_6m,
                max_dd_6m=max_dd_6m,
                max_dist_from_52w_high=max_dist_high,
            )

        run_screen = st.form_submit_button("Run screen")

    if st.session_state.get("last_run_ts"):
        st.caption(f"Last run: {st.session_state['last_run_ts']}")


# ----------------------------
# Run screen: store params + compute (only when button clicked, or params changed)
# ----------------------------
if run_screen:
    tickers, err = parse_tickers_from_inputs(universe_mode, tickers_text, uploaded)
    if err:
        st.session_state["screen_error"] = err
        st.session_state["screen_params"] = None
    else:
        params = {
            "mode": mode,
            "universe_mode": universe_mode,
            "tickers": tuple(tickers),
            "selected_setups": tuple(selected_setups),
            "max_workers": int(max_workers),
            "throttle_safe_mode": bool(throttle_safe_mode),
            # store short-term params object directly so results are stable
            "st_params_obj": st_params,
        }
        st.session_state["screen_params"] = params
        st.session_state["screen_error"] = None
        st.session_state["last_run_ts"] = _now_label()

# If never run, stop with message
if st.session_state.get("screen_error"):
    st.error(st.session_state["screen_error"])
    st.stop()

params = st.session_state.get("screen_params")
if not params:
    st.info("Set your options in the sidebar and click **Run screen**.")
    st.stop()

mode = params["mode"]
tickers = list(params["tickers"])
selected_setups = list(params["selected_setups"])
st_params = params.get("st_params_obj", None)

if not tickers:
    st.info("Add tickers via sidebar and click **Run screen**.")
    st.stop()

st.write(f"Universe size: **{len(tickers)}**")

# Compute hash to decide whether we need to recompute heavy outputs
screen_hash = _stable_hash(params)
need_recompute = (st.session_state.get("screen_hash") != screen_hash) or ("prices" not in st.session_state)

# Always include SPY for RS + benchmark backtests
tickers_for_prices = sorted(list(dict.fromkeys(tickers + ["SPY"])))

if need_recompute:
    with st.status("Fetching price data (cached batch)…", expanded=False) as s:
        prices = cached_prices(tuple(tickers_for_prices), years=5)
        s.update(label="Price data loaded.", state="complete")

    if prices.empty:
        st.error("No price data returned. Check tickers or data source.")
        st.stop()

    st.session_state["prices"] = prices

    if mode == "Long-term":
        with st.status("Fetching fundamentals + metadata (cached)…", expanded=False) as s:
            fundamentals = cached_fundamentals(tuple(tickers))
            s.update(label="Fundamentals loaded.", state="complete")

        with st.status("Computing long-term metrics + scoring…", expanded=False) as s:
            metrics_df = build_metrics_table(tickers, prices, fundamentals)
            df_full = score_table(metrics_df)
            df_full = add_data_quality_badges(df_full)
            df_full = classify_setups(df_full)

            # ✅ add trader decision layer (Action / Chase risk / Entries / Invalidation / Management)
            df_full = add_decisions_long_term(df_full, prices)

            # Stable ranking (important!)
            if "score" in df_full.columns and "ticker" in df_full.columns:
                df_full = df_full.sort_values(["score", "ticker"], ascending=[False, True]).reset_index(drop=True)

            s.update(label="Done.", state="complete")

        st.session_state["fundamentals"] = fundamentals
        st.session_state["df_full"] = df_full
        st.session_state.pop("st_table_full", None)

    else:
        # Short-term compute
        if st_params is None:
            st_params = ShortTermParams(
                min_avg_dollar_vol=20_000_000,
                require_above_ma50=True,
                require_above_ma200=False,
                min_mom_3m=0.0,
                min_mom_6m=0.0,
                max_dd_6m=0.45,
                max_dist_from_52w_high=0.25,
            )

        with st.status("Computing short-term metrics…", expanded=False) as s:
            st_table_full = compute_short_term_table(prices, tickers)
            st_table_full = score_short_term(st_table_full)
            st_table_full = apply_short_term_filters(st_table_full, st_params)
            try:
                st_table_full = classify_setups(st_table_full)
            except Exception:
                pass

            if "short_score" in st_table_full.columns and "ticker" in st_table_full.columns:
                st_table_full = st_table_full.sort_values(["short_score", "ticker"], ascending=[False, True]).reset_index(drop=True)

            s.update(label="Short-term metrics ready.", state="complete")

        st.session_state["st_table_full"] = st_table_full
        st.session_state.pop("fundamentals", None)
        st.session_state.pop("df_full", None)

    st.session_state["screen_hash"] = screen_hash

else:
    prices = st.session_state["prices"]


# ----------------------------
# Display + interaction (cheap; uses stored results)
# ----------------------------
if mode == "Short-term":
    st.subheader("Short-term Screener (Momentum + Trend + Liquidity)")
    st.caption("Short-term = momentum/trend + liquidity + controlled drawdowns + trade plan + portfolio heat.")

    st_table_full = st.session_state.get("st_table_full", pd.DataFrame()).copy()
    if st_table_full.empty:
        st.warning("No tickers passed short-term filters (or no results computed). Click Run screen.")
        st.stop()

    # Setup filter should be cheap and live
    if selected_setups and "setup_type" in st_table_full.columns:
        st_table = st_table_full[st_table_full["setup_type"].isin(selected_setups)].copy()
    else:
        st_table = st_table_full.copy()

    if st_table.empty:
        st.warning("No tickers match your setup filter.")
        st.stop()

    view = st_table.copy()
    for col in ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "dist_from_52w_high", "max_drawdown_6m"]:
        if col in view.columns:
            view[col] = view[col].apply(pct)
    if "avg_dollar_vol_20" in view.columns:
        view["avg_dollar_vol_20"] = view["avg_dollar_vol_20"].apply(num)

    st.dataframe(view, width="stretch", height=380)

    st.download_button(
        "Download short-term results CSV",
        data=st_table.to_csv(index=False).encode("utf-8"),
        file_name="wealth_builder_short_term.csv",
        mime="text/csv",
    )

    st.subheader("Deep dive")
    pick = st.selectbox("Select a ticker", st_table["ticker"].tolist(), index=0)

    c1, c2 = st.columns([1.45, 1])
    with c1:
        st.plotly_chart(make_weekly_candles(prices, pick), use_container_width=True)
        st.plotly_chart(make_relative_strength(prices, pick, "SPY"), use_container_width=True)
        st.plotly_chart(make_drawdown_chart(prices, pick), use_container_width=True)

    with c2:
        row = st_table[st_table["ticker"] == pick].iloc[0].to_dict()
        st.metric("Short score", f"{row.get('short_score', 0):.1f}")
        st.write(row)

    st.divider()
    st.subheader("Trade Plan (Position sizing + stops + correlation cap + portfolio heat)")

    rp_col1, rp_col2, rp_col3 = st.columns(3)
    with rp_col1:
        account_size = st.number_input("Account size ($)", value=10_000.0, step=1_000.0)
        risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1) / 100.0
        max_pos = st.slider("Max positions", 1, 30, 10, 1)

    with rp_col2:
        stop_type = st.selectbox("Stop type", ["ATR", "PCT"], index=0)
        atr_mult = st.slider("ATR multiple", 1.0, 5.0, 2.0, 0.1)
        pct_stop = st.slider("Percent stop", 0.01, 0.30, 0.08, 0.01)

    with rp_col3:
        vol_target = st.slider("Vol target (annual) for sizing", 0.05, 0.60, 0.20, 0.01)
        heat_cap_pct = st.slider("Portfolio heat cap (%)", 1.0, 20.0, 6.0, 0.5) / 100.0
        use_corr_cap = st.checkbox("Use correlation cap", value=True)

    corr_lookback = st.slider("Correlation lookback (days)", 30, 252, 126, 7)
    max_pair_corr = st.slider("Max pair correlation", 0.10, 0.99, 0.75, 0.01)

    plan_params = RiskParams(
        account_size=float(account_size),
        risk_per_trade_pct=float(risk_pct),
        max_positions=int(max_pos),
        stop_type=str(stop_type),
        atr_mult=float(atr_mult),
        pct_stop=float(pct_stop),
        vol_target_annual=float(vol_target),
    )

    plan_from_n = st.slider("Build plan from top N short-term names", 3, min(60, len(st_table)), 25, 1)
    ranked_for_plan = st_table.head(plan_from_n)["ticker"].tolist()

    if use_corr_cap:
        plan_universe = correlation_filter(
            prices=prices,
            ranked_tickers=ranked_for_plan,
            lookback_days=int(corr_lookback),
            max_pair_corr=float(max_pair_corr),
            max_n=int(max_pos),
        )
        st.caption(f"Correlation cap selected **{len(plan_universe)}** names out of {min(plan_from_n, len(st_table))}.")
    else:
        plan_universe = ranked_for_plan[: int(max_pos)]

    trade_plan_raw = build_trade_plan(prices, plan_universe, plan_params)
    trade_plan = enforce_heat_cap(trade_plan_raw, float(account_size), float(heat_cap_pct))

    heat = portfolio_heat(trade_plan, float(account_size))
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        st.metric("Portfolio heat", f"{heat['heat_pct']*100:.2f}%")
    with hc2:
        st.metric("Risk dollars (sum)", f"${heat['risk_dollars']:.2f}")
    with hc3:
        st.metric("Heat cap", f"{heat_cap_pct*100:.2f}%")

    st.dataframe(trade_plan, width="stretch", height=340)

    st.download_button(
        "Download trade plan CSV",
        data=trade_plan.to_csv(index=False).encode("utf-8"),
        file_name="wealth_builder_trade_plan.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Backtest (Top-N vs SPY)")

    bt_mode = st.selectbox(
        "Backtest mode",
        ["Buy & Hold", "Rebalanced v2 (Causal Momentum + Costs)"],
        index=0,
    )
    top_n = st.slider("Top N", 5, min(50, len(st_table)), 10, 1)
    start = st.date_input("Start date", value=pd.Timestamp.today() - pd.Timedelta(days=365 * 3))
    end = st.date_input("End date", value=pd.Timestamp.today())

    if bt_mode == "Buy & Hold":
        bt_tickers = st_table.head(top_n)["ticker"].tolist()
        res = backtest_equal_weight(prices, bt_tickers, benchmark="SPY", start=str(start), end=str(end))
        if "error" in res.stats:
            st.error(res.stats["error"])
        else:
            st.plotly_chart(equity_curve_chart(res.equity_curve, f"Buy & Hold Top {top_n} vs SPY"), use_container_width=True)
            st.write(res.stats)
    else:
        mom_choice = st.selectbox("Momentum lookback", ["3M", "6M", "12M"], index=1)
        mom_map = {"3M": "ret_63d", "6M": "ret_126d", "12M": "ret_252d"}
        mom_col = mom_map[mom_choice]

        rb_freq = st.selectbox("Rebalance frequency", ["Monthly", "Weekly"], index=0)
        rb_code = "M" if rb_freq == "Monthly" else "W"

        fee_bps = st.slider("Transaction cost (bps)", 0.0, 50.0, 5.0, 0.5)
        skip_overlap = st.slider("Skip rebalance if overlap ≥", 0.0, 0.99, 0.70, 0.01)

        ranker = _causal_momentum_ranker_factory(
            prices_long=prices,
            universe=tickers,
            momentum_col=mom_col,
            min_adv=20_000_000.0,
            require_above_ma50=True,
            require_above_ma200=False,
        )

        rb2 = backtest_rebalanced_v2(
            prices=prices,
            universe=tickers,
            ranker=ranker,
            top_n=int(top_n),
            benchmark="SPY",
            start=str(start),
            end=str(end),
            rebalance=rb_code,
            fee_bps=float(fee_bps),
            skip_if_overlap_pct=float(skip_overlap),
        )

        if "error" in rb2.stats:
            st.error(rb2.stats["error"])
        else:
            st.plotly_chart(equity_curve_chart(rb2.equity_curve, f"Rebalanced v2 ({rb_freq}) Top {top_n} vs SPY"), use_container_width=True)
            st.write(rb2.stats)

    st.stop()


# ============================
# LONG-TERM MODE
# ============================
st.subheader("Long-term Screener (Durability + Consistency + Trend)")

df_full = st.session_state.get("df_full", pd.DataFrame()).copy()
fundamentals = st.session_state.get("fundamentals", pd.DataFrame())

if df_full.empty:
    st.warning("No long-term results computed. Click Run screen.")
    st.stop()

# Setup filter is cheap & stable
df_view = df_full.copy()
if selected_setups and "setup_type" in df_view.columns:
    df_view = df_view[df_view["setup_type"].isin(selected_setups)].copy()

if df_view.empty:
    st.warning("No tickers passed your setup filter. Loosen filters and click Run screen.")
    st.stop()

# ✅ Actionable top table
primary_cols = [
    "ticker",
    "action",
    "chase_risk",
    "setup_type",
    "score",
    "pct_above_ma50_text",
    "entry_pullback",
    "invalidation",
]
show_cols = [c for c in primary_cols if c in df_view.columns]
st.dataframe(df_view[show_cols], width="stretch", height=420)

st.subheader("Ranked results")
st.dataframe(df_view, width="stretch", height=420)

st.subheader("Deep dive")
pick = st.selectbox("Select a ticker", df_view["ticker"].tolist(), index=0)

c1, c2 = st.columns([1.45, 1])
with c1:
    st.plotly_chart(make_weekly_candles(prices, pick), use_container_width=True)
    st.plotly_chart(make_relative_strength(prices, pick, "SPY"), use_container_width=True)
    st.plotly_chart(make_drawdown_chart(prices, pick), use_container_width=True)

with c2:
    row = df_view[df_view["ticker"] == pick].iloc[0].to_dict()
    st.metric("Score", f"{row.get('score', 0):.1f}")

    # ✅ Decision card
    st.subheader("Decision Card")
    st.write(decision_card(row))

    with st.expander("Raw metrics (debug)"):
        st.write(row)

st.subheader("Financial trends")
t1, t2 = st.columns(2)
with t1:
    st.plotly_chart(make_margin_trends(fundamentals, pick), use_container_width=True)
with t2:
    st.plotly_chart(make_cashflow_trends(fundamentals, pick), use_container_width=True)

st.divider()
st.subheader("Backtest (Top-N vs SPY)")

bt_mode = st.selectbox(
    "Backtest mode",
    ["Buy & Hold", "Rebalanced v2 (Static Long-term Ranks + Costs)", "Rebalanced v2 (Causal via Snapshots + Costs)"],
    index=0,
)
top_n = st.slider("Top N", 5, min(50, len(df_view)), 10, 1)

start = st.date_input("Start date", value=pd.Timestamp.today() - pd.Timedelta(days=365 * 5))
end = st.date_input("End date", value=pd.Timestamp.today())

if bt_mode == "Buy & Hold":
    bt_tickers = df_view.head(top_n)["ticker"].tolist()
    res = backtest_equal_weight(prices, bt_tickers, benchmark="SPY", start=str(start), end=str(end))
    if "error" in res.stats:
        st.error(res.stats["error"])
    else:
        st.plotly_chart(equity_curve_chart(res.equity_curve, f"Buy & Hold Top {top_n} vs SPY"), use_container_width=True)
        st.write(res.stats)

else:
    rb_freq = st.selectbox("Rebalance frequency", ["Monthly", "Weekly"], index=0)
    rb_code = "M" if rb_freq == "Monthly" else "W"
    fee_bps = st.slider("Transaction cost (bps)", 0.0, 50.0, 5.0, 0.5)
    skip_overlap = st.slider("Skip rebalance if overlap ≥", 0.0, 0.99, 0.70, 0.01)

    if bt_mode == "Rebalanced v2 (Static Long-term Ranks + Costs)":
        ranked = df_view["ticker"].tolist()

        def ranker(_asof: pd.Timestamp) -> list[str]:
            return ranked

    else:
        snaps = st.session_state["lt_snapshots"]
        if snaps is None or snaps.empty:
            st.error("No snapshots available. Create/upload snapshots to run causal long-term backtest.")
            st.stop()

        ranker = ranker_from_snapshots(snaps, universe=tickers, score_col="score")

    rb2 = backtest_rebalanced_v2(
        prices=prices,
        universe=tickers,
        ranker=ranker,
        top_n=int(top_n),
        benchmark="SPY",
        start=str(start),
        end=str(end),
        rebalance=rb_code,
        fee_bps=float(fee_bps),
        skip_if_overlap_pct=float(skip_overlap),
    )

    if "error" in rb2.stats:
        st.error(rb2.stats["error"])
    else:
        st.plotly_chart(
            equity_curve_chart(rb2.equity_curve, f"Rebalanced v2 ({rb_freq}) Top {top_n} vs SPY"),
            use_container_width=True,
        )
        st.write(rb2.stats)

        cta, ctb = st.columns(2)
        with cta:
            with st.expander("Holdings (weights by rebalance date)"):
                st.dataframe(rb2.holdings, width="stretch", height=260)
        with ctb:
            with st.expander("Turnover & costs by rebalance date"):
                st.dataframe(rb2.trades, width="stretch", height=260)
