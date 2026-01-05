
---

## `app.py`

```python
import streamlit as st
import pandas as pd

from wb.universe import get_universe
from wb.prices import batch_fetch_prices
from wb.fundamentals import fetch_fundamentals_parallel
from wb.metrics import build_metrics_table
from wb.scoring import score_table
from wb.charts import (
    make_weekly_candles,
    make_relative_strength,
    make_drawdown_chart,
    make_margin_trends,
    make_cashflow_trends,
)

st.set_page_config(page_title="Wealth Builder Screener", layout="wide")

st.title("Wealth Builder Screener")
st.caption("Long-term screening: trend health + financial durability + quality. (Short-term tools later.)")

with st.sidebar:
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
    st.header("Long-term filters")

    # Fundamentals filters
    min_rev_cagr = st.slider("Min Revenue CAGR (3y)", 0.0, 0.40, 0.10, 0.01)
    min_gm = st.slider("Min Gross Margin", 0.0, 0.80, 0.35, 0.01)
    min_fcf_margin = st.slider("Min FCF Margin (approx)", -0.20, 0.40, 0.05, 0.01)

    require_fcf_pos = st.checkbox("Require TTM FCF > 0", value=True)
    max_debt_to_equity = st.slider("Max Debt/Equity", 0.0, 3.0, 1.0, 0.05)
    max_sbc_pct_rev = st.slider("Max SBC as % of Revenue", 0.0, 0.30, 0.10, 0.01)

    st.divider()
    st.header("Price action filters")

    require_above_ma200 = st.checkbox("Require price above 200D MA", value=True)
    require_above_40w = st.checkbox("Require price above 40W MA", value=True)
    min_rs_12m = st.slider("Min Relative Strength vs SPY (12m)", -0.50, 1.00, 0.00, 0.01)
    max_drawdown_2y = st.slider("Max drawdown (2y)", 0.10, 0.90, 0.50, 0.01)

    st.divider()
    st.header("Performance")
    max_workers = st.slider("Parallel workers", 4, 32, 16, 1)
    st.caption("More workers = faster, but may hit provider throttling on large universes.")

    run = st.button("Run screen", type="primary")

if not tickers:
    st.info("Add tickers via sidebar to begin.")
    st.stop()

if run:
    tickers = sorted(list(dict.fromkeys([t for t in tickers if t.isalnum() or "-" in t or "." in t])))
    st.write(f"Universe size: **{len(tickers)}**")

    # 1) Prices (batch)
    with st.status("Fetching price data (batch)…", expanded=False) as s:
        prices = batch_fetch_prices(tickers, years=5)
        s.update(label="Price data loaded.", state="complete")

    # 2) Fundamentals (parallel)
    with st.status("Fetching fundamentals (parallel)…", expanded=False) as s:
        fundamentals = fetch_fundamentals_parallel(tickers, max_workers=max_workers)
        s.update(label="Fundamentals loaded.", state="complete")

    # 3) Metrics table
    with st.status("Computing metrics + scoring…", expanded=False) as s:
        metrics_df = build_metrics_table(tickers, prices, fundamentals)
        scored = score_table(metrics_df)
        s.update(label="Done.", state="complete")

    # Apply filters
    df = scored.copy()

    # Fundamental filters
    df = df[df["rev_cagr_3y"].fillna(-999) >= min_rev_cagr]
    df = df[df["gross_margin"].fillna(-999) >= min_gm]
    df = df[df["fcf_margin"].fillna(-999) >= min_fcf_margin]
    if require_fcf_pos:
        df = df[df["fcf_ttm"].fillna(-1) > 0]
    df = df[df["debt_to_equity"].fillna(999) <= max_debt_to_equity]
    df = df[df["sbc_pct_rev"].fillna(999) <= max_sbc_pct_rev]

    # Price filters
    if require_above_ma200:
        df = df[df["above_ma200"] == True]
    if require_above_40w:
        df = df[df["above_ma40w"] == True]
    df = df[df["rs_vs_spy_12m"].fillna(-999) >= min_rs_12m]
    df = df[df["max_drawdown_2y"].fillna(999) <= max_drawdown_2y]

    st.subheader("Ranked results")
    st.caption("Tip: sort by Score, then sanity-check business quality and narrative. Charts validate trend health.")

    display_cols = [
        "ticker",
        "score",
        "rev_cagr_3y",
        "gross_margin",
        "op_margin",
        "fcf_ttm",
        "fcf_margin",
        "debt_to_equity",
        "sbc_pct_rev",
        "rs_vs_spy_12m",
        "max_drawdown_2y",
        "above_ma200",
        "above_ma40w",
    ]
    disp = df[display_cols].sort_values(["score"], ascending=False).reset_index(drop=True)

    def pct(x): return "—" if pd.isna(x) else f"{x*100:.1f}%"
    def num(x):
        if pd.isna(x): return "—"
        ax = abs(x)
        if ax >= 1e9: return f"{x/1e9:.2f}B"
        if ax >= 1e6: return f"{x/1e6:.1f}M"
        if ax >= 1e3: return f"{x/1e3:.1f}K"
        return f"{x:.0f}"

    pretty = disp.copy()
    pretty["rev_cagr_3y"] = pretty["rev_cagr_3y"].apply(pct)
    pretty["gross_margin"] = pretty["gross_margin"].apply(pct)
    pretty["op_margin"] = pretty["op_margin"].apply(pct)
    pretty["fcf_ttm"] = pretty["fcf_ttm"].apply(num)
    pretty["fcf_margin"] = pretty["fcf_margin"].apply(pct)
    pretty["debt_to_equity"] = pretty["debt_to_equity"].apply(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    pretty["sbc_pct_rev"] = pretty["sbc_pct_rev"].apply(pct)
    pretty["rs_vs_spy_12m"] = pretty["rs_vs_spy_12m"].apply(pct)
    pretty["max_drawdown_2y"] = pretty["max_drawdown_2y"].apply(pct)

    st.dataframe(pretty, use_container_width=True, height=360)

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
        st.write({
            "Revenue CAGR (3y)": pct(row.get("rev_cagr_3y")),
            "Gross margin": pct(row.get("gross_margin")),
            "Operating margin": pct(row.get("op_margin")),
            "FCF (TTM)": num(row.get("fcf_ttm")),
            "FCF margin": pct(row.get("fcf_margin")),
            "Debt/Equity": "—" if pd.isna(row.get("debt_to_equity")) else f"{row.get('debt_to_equity'):.2f}",
            "SBC % revenue": pct(row.get("sbc_pct_rev")),
            "RS vs SPY (12m)": pct(row.get("rs_vs_spy_12m")),
            "Max drawdown (2y)": pct(row.get("max_drawdown_2y")),
        })

        st.markdown("### Quality checks")
        st.write({
            "Positive FCF?": "✅" if (row.get("fcf_ttm") is not None and row.get("fcf_ttm") > 0) else "—",
            "Above MA200?": "✅" if row.get("above_ma200") else "—",
            "Above 40W?": "✅" if row.get("above_ma40w") else "—",
            "ROIC-ish proxy": pct(row.get("roic_proxy")),
            "Share dilution proxy (3y)": pct(row.get("shares_change_3y")),
        })

    st.subheader("Financial trends")
    t1, t2 = st.columns(2)
    with t1:
        st.plotly_chart(make_margin_trends(fundamentals, pick), use_container_width=True)
    with t2:
        st.plotly_chart(make_cashflow_trends(fundamentals, pick), use_container_width=True)

else:
    st.info("Pick a universe and click **Run screen**.")
