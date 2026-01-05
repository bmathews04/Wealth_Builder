# app.py
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

from wb.snapshots import make_snapshot, merge_snapshots, snapshots_to_bytes, snapshots_from_bytes
from wb.causal_ranker import ranker_from_snapshots


st.set_page_config(page_title="Wealth Builder Screener", layout="wide")

st.title("Wealth Builder Screener")
st.caption("Mode toggle: Long-term (durability) vs Short-term (momentum). Backtests + risk tools included.")


# ----------------------------
# Sidebar: Universe + Global perf
# ----------------------------
with st.sidebar:
    st.header("Mode")
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
            # Optional: keep these if you want direct wiki calls too
            "S&P 500 (Wikipedia)",
            "Nasdaq 100 (Wikipedia)",
        ],
        index=0,
    )

    st.caption(f"Universe module: {UNIVERSE_MODULE_VERSION}")

    tickers: list[str] = []

    if universe_mode == "Paste tickers":
        tickers_text = st.text_area(
            "Tickers (comma or newline separated)",
            value="AAPL, MSFT, AMZN, NVDA, GOOGL, META",
            height=120,
        )
        tickers = [t.strip().upper() for t in tickers_text.replace("\n", ",").split(",") if t.strip()]

    elif universe_mode == "Upload CSV":
        f = st.file_uploader("Upload CSV with a 'Ticker' column", type=["csv"])
        if f is not None:
            dfu = pd.read_csv(f)
            if "Ticker" not in dfu.columns:
                st.error("CSV must contain a 'Ticker' column.")
            else:
                tickers = dfu["Ticker"].astype(str).str.upper().str.strip().tolist()

    elif universe_mode == "S&P 500 (ETF proxy: SPY)":
        try:
            tickers = get_universe("etf:SPY")
        except Exception as e:
            st.error(f"Failed to load SPY holdings universe: {e}")

    elif universe_mode == "Nasdaq-100 (ETF proxy: QQQ)":
        try:
            tickers = get_universe("etf:QQQ")
        except Exception as e:
            st.error(f"Failed to load QQQ holdings universe: {e}")

    elif universe_mode == "Russell 1000 (ETF proxy: IWB)":
        try:
            tickers = get_universe("etf:IWB")
        except Exception as e:
            st.error(f"Failed to load IWB holdings universe: {e}")

    elif universe_mode == "S&P 500 (Wikipedia)":
        try:
            tickers = get_universe("sp500")
        except Exception as e:
            st.error(f"Failed to load S&P 500 from Wikipedia: {e}")

    elif universe_mode == "Nasdaq 100 (Wikipedia)":
        try:
            tickers = get_universe("nasdaq100")
        except Exception as e:
            st.error(f"Failed to load Nasdaq-100 from Wikipedia: {e}")

    # Optional cleanup
    tickers = [t for t in tickers if t]
    tickers = list(dict.fromkeys(tickers))  # dedupe, preserve order

    st.header("Setup Types")
    selected_setups = st.multiselect(
        "Show setups",
        options=SETUP_ORDER,
        default=["Leader", "Early Trend", "Bottom-Fishing", "Extended"],
    )

    st.divider()
    st.header("Performance")
    max_workers = st.slider("Parallel workers (fundamentals)", 4, 32, 16, 1)
    st.caption("More workers = faster, but large universes may hit throttling occasionally.")

    throttle_safe_mode = st.checkbox("Throttle-safe mode", value=False)
    st.caption("If screening large universes and seeing missing fundamentals, enable this.")

    tickers = [t for t in tickers if t]  # remove blanks
    tickers = list(dict.fromkeys(tickers))  # dedupe preserve order


# ----------------------------
# Session state for snapshots
# ----------------------------
if "lt_snapshots" not in st.session_state:
    st.session_state["lt_snapshots"] = pd.DataFrame()


# ----------------------------
# Clean tickers + fetch prices (always)
# ----------------------------
if tickers:
    tickers = [t for t in tickers if t and isinstance(t, str)]
    tickers = sorted(list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()])))

if not tickers:
    st.info("Add tickers via sidebar to begin.")
    st.stop()

st.write(f"Universe size: **{len(tickers)}**")

# Always include SPY for RS + benchmark backtests
tickers_for_prices = sorted(list(dict.fromkeys(tickers + ["SPY"])))

with st.status("Fetching price data (batch)…", expanded=False) as s:
    prices = batch_fetch_prices(tickers_for_prices, years=5)
    s.update(label="Price data loaded.", state="complete")

if prices.empty:
    st.error("No price data returned. Check tickers or data source.")
    st.stop()


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


# ============================
# SHORT-TERM MODE
# ============================
if mode == "Short-term":
    st.subheader("Short-term Screener (Momentum + Trend + Liquidity)")
    st.caption("Short-term = momentum/trend + liquidity + controlled drawdowns + trade plan + portfolio heat.")

    with st.sidebar:
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

    with st.status("Computing short-term metrics…", expanded=False) as s:
        st_table = compute_short_term_table(prices, tickers)
        st_table = score_short_term(st_table)
        st_table = apply_short_term_filters(st_table, st_params)
        st_table = st_table.sort_values("short_score", ascending=False).reset_index(drop=True)
        s.update(label="Short-term metrics ready.", state="complete")

    if st_table.empty:
        st.warning("No tickers passed short-term filters.")
        st.stop()

    # classify setups for short-term view (uses wb.setups)
    try:
        st_table = classify_setups(st_table)
        if selected_setups:
            st_table = st_table[st_table["setup"].isin(selected_setups)].copy()
    except Exception:
        pass

    view = st_table.copy()
    for col in ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "dist_from_52w_high", "max_drawdown_6m"]:
        if col in view.columns:
            view[col] = view[col].apply(pct)
    if "avg_dollar_vol_20" in view.columns:
        view["avg_dollar_vol_20"] = view["avg_dollar_vol_20"].apply(num)

    # Streamlit 2026 deprecation: use width instead of use_container_width
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
            min_adv=float(min_adv),
            require_above_ma50=bool(require_ma50),
            require_above_ma200=bool(require_ma200),
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

with st.status("Fetching fundamentals + metadata (parallel)…", expanded=False) as s:
    workers = min(max_workers, 8) if throttle_safe_mode else max_workers
    fundamentals = fetch_fundamentals_parallel(tickers, max_workers=workers)
    s.update(label="Fundamentals loaded.", state="complete")

with st.status("Computing long-term metrics + scoring…", expanded=False) as s:
    metrics_df = build_metrics_table(tickers, prices, fundamentals)
    scored = score_table(metrics_df)
    scored = add_data_quality_badges(scored)
    s.update(label="Done.", state="complete")

df = scored.copy()

st.subheader("Ranked results")
st.dataframe(df.sort_values("score", ascending=False), width="stretch", height=420)

if df.empty:
    st.warning("No tickers passed your filters. Loosen filters or change universe.")
    st.stop()

st.subheader("Deep dive")
pick = st.selectbox("Select a ticker", df.sort_values("score", ascending=False)["ticker"].tolist(), index=0)

c1, c2 = st.columns([1.45, 1])
with c1:
    st.plotly_chart(make_weekly_candles(prices, pick), use_container_width=True)
    st.plotly_chart(make_relative_strength(prices, pick, "SPY"), use_container_width=True)
    st.plotly_chart(make_drawdown_chart(prices, pick), use_container_width=True)

with c2:
    row = scored[scored["ticker"] == pick].iloc[0].to_dict()
    st.metric("Score", f"{row.get('score', 0):.1f}")
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
top_n = st.slider("Top N", 5, min(50, len(df)), 10, 1)

start = st.date_input("Start date", value=pd.Timestamp.today() - pd.Timedelta(days=365 * 5))
end = st.date_input("End date", value=pd.Timestamp.today())

if bt_mode == "Buy & Hold":
    bt_tickers = df.sort_values("score", ascending=False).head(top_n)["ticker"].tolist()
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
        ranked = df.sort_values("score", ascending=False)["ticker"].tolist()

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
