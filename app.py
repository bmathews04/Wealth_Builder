# app.py
import streamlit as st
import pandas as pd

from wb.universe import get_universe
from wb.prices import batch_fetch_prices

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


st.set_page_config(page_title="Wealth Builder Screener", layout="wide")

st.title("Wealth Builder Screener")
st.caption("Mode toggle: Long-term (durability) vs Short-term (momentum). Backtest included for both.")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Mode")
    mode = st.selectbox("Select mode", ["Long-term", "Short-term"], index=0)

    st.header("Universe")
    universe_mode = st.selectbox(
        "Pick a universe",
        ["Paste tickers", "Upload CSV", "S&P 500 (Wikipedia)", "Nasdaq 100 (Wikipedia)"],
        index=0,
    )

    tickers = []
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

    else:
        tickers = get_universe("sp500" if "S&P 500" in universe_mode else "nasdaq100")

    st.divider()
    st.header("Performance")
    max_workers = st.slider("Parallel workers (fundamentals)", 4, 32, 16, 1)
    st.caption("More workers = faster, but large universes may hit throttling occasionally.")

    throttle_safe_mode = st.checkbox("Throttle-safe mode", value=False)
    st.caption("If screening large universes and seeing missing fundamentals, enable this.")

# Clean tickers
if tickers:
    tickers = [t for t in tickers if t and isinstance(t, str)]
    tickers = sorted(list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()])))

if not tickers:
    st.info("Add tickers via sidebar to begin.")
    st.stop()

st.write(f"Universe size: **{len(tickers)}**")

# Always include SPY in price pulls for RS calc & benchmark backtests
tickers_for_prices = sorted(list(dict.fromkeys(tickers + ["SPY"])))

# ----------------------------
# Fetch prices
# ----------------------------
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

# ----------------------------
# SHORT-TERM MODE
# ----------------------------
if mode == "Short-term":
    st.subheader("Short-term Screener (Momentum + Trend + Liquidity)")
    st.caption("Short-term = momentum/trend + liquidity + controlled drawdowns. Catalysts + risk rules next.")

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

    view = st_table.copy()
    for col in ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "dist_from_52w_high", "max_drawdown_6m"]:
        if col in view.columns:
            view[col] = view[col].apply(pct)
    if "avg_dollar_vol_20" in view.columns:
        view["avg_dollar_vol_20"] = view["avg_dollar_vol_20"].apply(num)

    st.dataframe(
        view[
            [
                "ticker",
                "short_score",
                "avg_dollar_vol_20",
                "mom_1m",
                "mom_3m",
                "mom_6m",
                "mom_12m",
                "dist_from_52w_high",
                "max_drawdown_6m",
                "above_ma20",
                "above_ma50",
                "above_ma200",
                "rsi_14",
            ]
        ],
        use_container_width=True,
        height=380,
    )

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
        st.markdown("### Short-term snapshot")
        st.metric("Short score", f"{row.get('short_score', 0):.1f}")
        st.write(
            {
                "Avg $ volume (20d)": num(row.get("avg_dollar_vol_20")),
                "1m mom": pct(row.get("mom_1m")),
                "3m mom": pct(row.get("mom_3m")),
                "6m mom": pct(row.get("mom_6m")),
                "12m mom": pct(row.get("mom_12m")),
                "Dist from 52w high": pct(row.get("dist_from_52w_high")),
                "Max drawdown (6m)": pct(row.get("max_drawdown_6m")),
                "RSI(14)": "—" if row.get("rsi_14") is None else round(row.get("rsi_14"), 1),
            }
        )

    st.divider()
    st.subheader("Backtest (Top-N equal weight vs SPY)")

    top_n = st.slider("Top N", 5, min(50, len(st_table)), 10, 1)
    start = st.date_input("Start date", value=pd.Timestamp.today() - pd.Timedelta(days=365 * 3))
    end = st.date_input("End date", value=pd.Timestamp.today())

    bt_tickers = st_table.head(top_n)["ticker"].tolist()
    res = backtest_equal_weight(prices, bt_tickers, benchmark="SPY", start=str(start), end=str(end))

    if "error" in res.stats:
        st.error(res.stats["error"])
    else:
        st.plotly_chart(equity_curve_chart(res.equity_curve, f"Top {top_n} vs SPY"), use_container_width=True)
        st.write(res.stats)

    st.stop()

# ----------------------------
# LONG-TERM MODE
# ----------------------------
st.subheader("Long-term Screener (Durability + Consistency + Trend)")

with st.sidebar:
    st.header("Long-term filters")

    min_rev_cagr = st.slider("Min Revenue CAGR (3y)", 0.0, 0.40, 0.10, 0.01)
    min_gm = st.slider("Min Gross Margin", 0.0, 0.80, 0.35, 0.01)
    min_fcf_margin = st.slider("Min FCF Margin (approx)", -0.20, 0.40, 0.05, 0.01)

    require_fcf_pos = st.checkbox("Require latest annual FCF > 0", value=True)
    max_debt_to_equity = st.slider("Max Debt/Equity", 0.0, 3.0, 1.0, 0.05)
    max_sbc_pct_rev = st.slider("Max SBC as % of Revenue", 0.0, 0.30, 0.10, 0.01)

    st.divider()
    st.header("Consistency (last 5 annual points)")
    gm_floor = st.slider("GM floor (consistency)", 0.0, 0.80, 0.30, 0.01)
    min_fcf_pos_years = st.slider("Min # years with FCF > 0", 0, 5, 3, 1)
    min_gm_years = st.slider("Min # years with GM above floor", 0, 5, 3, 1)
    min_rev_up_years = st.slider("Min # years revenue grew YoY", 0, 5, 3, 1)

    st.divider()
    st.header("Price action filters")
    require_above_ma200 = st.checkbox("Require price above 200D MA", value=True)
    require_above_40w = st.checkbox("Require price above 40W MA", value=True)
    min_rs_12m = st.slider("Min RS vs SPY (12m)", -0.50, 1.00, 0.00, 0.01)
    max_drawdown_2y = st.slider("Max drawdown (2y)", 0.10, 0.90, 0.50, 0.01)

# Fundamentals fetch only in long-term mode
with st.status("Fetching fundamentals + metadata (parallel)…", expanded=False) as s:
    # if you want throttle-safe mode to influence provider logic, set in secrets.
    # here we just reduce workers if user checked it
    workers = min(max_workers, 8) if throttle_safe_mode else max_workers
    fundamentals = fetch_fundamentals_parallel(tickers, max_workers=workers)
    s.update(label="Fundamentals loaded.", state="complete")

with st.status("Computing long-term metrics + scoring…", expanded=False) as s:
    metrics_df = build_metrics_table(tickers, prices, fundamentals, gm_floor=gm_floor)
    scored = score_table(metrics_df)
    scored = add_data_quality_badges(scored)
    s.update(label="Done.", state="complete")

df = scored.copy()

# Apply filters
df = df[df["rev_cagr_3y"].fillna(-999) >= min_rev_cagr]
df = df[df["gross_margin"].fillna(-999) >= min_gm]
df = df[df["fcf_margin"].fillna(-999) >= min_fcf_margin]

if require_fcf_pos:
    df = df[df["fcf_ttm"].fillna(-1) > 0]

df = df[df["debt_to_equity"].fillna(999) <= max_debt_to_equity]
df = df[df["sbc_pct_rev"].fillna(999) <= max_sbc_pct_rev]

df = df[df["fcf_pos_years_5"].fillna(0) >= min_fcf_pos_years]
df = df[df["gm_floor_years_5"].fillna(0) >= min_gm_years]
df = df[df["rev_up_years_5"].fillna(0) >= min_rev_up_years]

if require_above_ma200:
    df = df[df["above_ma200"] == True]
if require_above_40w:
    df = df[df["above_ma40w"] == True]

df = df[df["rs_vs_spy_12m"].fillna(-999) >= min_rs_12m]
df = df[df["max_drawdown_2y"].fillna(999) <= max_drawdown_2y]

st.subheader("Ranked results")

display_cols = [
    "ticker", "sector", "industry", "data_quality", "score",
    "rev_cagr_3y", "gross_margin", "op_margin",
    "fcf_ttm", "fcf_margin",
    "debt_to_equity", "sbc_pct_rev",
    "shares_change_3y", "roic_proxy",
    "fcf_pos_years_5", "gm_floor_years_5", "rev_up_years_5",
    "rs_vs_spy_12m", "max_drawdown_2y",
    "above_ma200", "above_ma40w",
]

disp = df[display_cols].sort_values(["score"], ascending=False).reset_index(drop=True)
pretty = disp.copy()

for col in ["rev_cagr_3y", "gross_margin", "op_margin", "fcf_margin", "sbc_pct_rev", "shares_change_3y", "roic_proxy", "rs_vs_spy_12m", "max_drawdown_2y"]:
    pretty[col] = pretty[col].apply(pct)

pretty["fcf_ttm"] = pretty["fcf_ttm"].apply(num)
pretty["debt_to_equity"] = pretty["debt_to_equity"].apply(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
pretty["above_ma200"] = pretty["above_ma200"].apply(truthy_badge)
pretty["above_ma40w"] = pretty["above_ma40w"].apply(truthy_badge)

st.dataframe(pretty, use_container_width=True, height=400)

st.download_button(
    "Download long-term results CSV",
    data=disp.to_csv(index=False).encode("utf-8"),
    file_name="wealth_builder_long_term.csv",
    mime="text/csv",
)

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
    st.markdown("### Snapshot")
    st.metric("Score", f"{row.get('score', 0):.1f}")
    st.write(
        {
            "Data quality": row.get("data_quality", "—"),
            "Sector": row.get("sector", "—"),
            "Industry": row.get("industry", "—"),
            "Revenue CAGR (3y)": pct(row.get("rev_cagr_3y")),
            "Gross margin": pct(row.get("gross_margin")),
            "Operating margin": pct(row.get("op_margin")),
            "FCF (latest annual)": num(row.get("fcf_ttm")),
            "FCF margin": pct(row.get("fcf_margin")),
            "Debt/Equity": "—" if pd.isna(row.get("debt_to_equity")) else f"{row.get('debt_to_equity'):.2f}",
            "SBC % revenue": pct(row.get("sbc_pct_rev")),
            "Share change (3y)": pct(row.get("shares_change_3y")),
            "ROIC proxy": pct(row.get("roic_proxy")),
            "RS vs SPY (12m)": pct(row.get("rs_vs_spy_12m")),
            "Max drawdown (2y)": pct(row.get("max_drawdown_2y")),
            "FCF+ years (5)": row.get("fcf_pos_years_5"),
            f"GM>={gm_floor:.0%} years (5)": row.get("gm_floor_years_5"),
            "Rev up YoY years (5)": row.get("rev_up_years_5"),
        }
    )

st.subheader("Financial trends")
t1, t2 = st.columns(2)
with t1:
    st.plotly_chart(make_margin_trends(fundamentals, pick), use_container_width=True)
with t2:
    st.plotly_chart(make_cashflow_trends(fundamentals, pick), use_container_width=True)

st.divider()
st.subheader("Backtest (Top-N equal weight vs SPY)")

top_n = st.slider("Top N", 5, min(50, len(df)), 10, 1)
start = st.date_input("Start date", value=pd.Timestamp.today() - pd.Timedelta(days=365 * 5))
end = st.date_input("End date", value=pd.Timestamp.today())

bt_tickers = df.sort_values("score", ascending=False).head(top_n)["ticker"].tolist()
res = backtest_equal_weight(prices, bt_tickers, benchmark="SPY", start=str(start), end=str(end))

if "error" in res.stats:
    st.error(res.stats["error"])
else:
    st.plotly_chart(equity_curve_chart(res.equity_curve, f"Top {top_n} vs SPY"), use_container_width=True)
    st.write(res.stats)
