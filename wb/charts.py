import pandas as pd
import plotly.graph_objects as go

def make_weekly_candles(prices: pd.DataFrame, ticker: str) -> go.Figure:
    g = prices[prices["ticker"] == ticker].set_index("date").sort_index()
    if g.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{ticker} Weekly Trend — no data", height=420)
        return fig

    wk = pd.DataFrame({
        "open": g["open"].resample("W-FRI").first(),
        "high": g["high"].resample("W-FRI").max(),
        "low":  g["low"].resample("W-FRI").min(),
        "close": g["close"].resample("W-FRI").last(),
    }).dropna()

    ma10 = wk["close"].rolling(10).mean()
    ma40 = wk["close"].rolling(40).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=wk.index, open=wk["open"], high=wk["high"], low=wk["low"], close=wk["close"],
        name="Weekly"
    ))
    fig.add_trace(go.Scatter(x=ma10.index, y=ma10, name="MA10W"))
    fig.add_trace(go.Scatter(x=ma40.index, y=ma40, name="MA40W"))

    fig.update_layout(
        title=f"{ticker} Weekly Trend",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False,
    )
    return fig

def make_relative_strength(prices: pd.DataFrame, ticker: str, benchmark: str = "SPY") -> go.Figure:
    s = prices[prices["ticker"] == ticker].set_index("date")["close"].sort_index()
    b = prices[prices["ticker"] == benchmark].set_index("date")["close"].sort_index()
    joined = pd.concat([s.rename("s"), b.rename("b")], axis=1).dropna()

    fig = go.Figure()
    if joined.empty:
        fig.update_layout(title=f"Relative Strength vs {benchmark} — no data", height=300)
        return fig

    rs = joined["s"] / joined["b"]
    rs_ma = rs.rolling(50).mean()

    fig.add_trace(go.Scatter(x=rs.index, y=rs, name=f"{ticker}/{benchmark}"))
    fig.add_trace(go.Scatter(x=rs_ma.index, y=rs_ma, name="RS MA50D"))
    fig.update_layout(
        title=f"Relative Strength vs {benchmark}",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

def make_drawdown_chart(prices: pd.DataFrame, ticker: str) -> go.Figure:
    s = prices[prices["ticker"] == ticker].set_index("date")["close"].sort_index().dropna()
    s = s.tail(504)

    fig = go.Figure()
    if s.empty:
        fig.update_layout(title="Drawdown (last ~2y) — no data", height=260)
        return fig

    peak = s.cummax()
    dd = (s / peak) - 1.0

    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Drawdown"))
    fig.update_layout(
        title="Drawdown (last ~2y)",
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_tickformat=".0%",
    )
    return fig

def make_margin_trends(fundamentals: dict, ticker: str) -> go.Figure:
    f = fundamentals.get(ticker, {})
    inc = f.get("income_a", pd.DataFrame())

    fig = go.Figure()
    if inc.empty:
        fig.update_layout(title="Margins (annual) — no data", height=320, margin=dict(l=20, r=20, t=50, b=20))
        return fig

    def row(keys):
        for k in keys:
            if k in inc.index:
                return inc.loc[k]
        return None

    rev = row(["Total Revenue", "Revenue"])
    gp = row(["Gross Profit"])
    op = row(["Operating Income"])

    if rev is None:
        fig.update_layout(title="Margins (annual) — no revenue row", height=320)
        return fig

    ser = pd.DataFrame({"rev": rev})
    if gp is not None: ser["gp"] = gp
    if op is not None: ser["op"] = op

    ser.columns = pd.to_datetime(ser.columns, errors="coerce")
    ser = ser.dropna(axis=1, how="all").sort_index(axis=1)

    gm = (ser.loc["gp"] / ser.loc["rev"]) if "gp" in ser.index else None
    om = (ser.loc["op"] / ser.loc["rev"]) if "op" in ser.index else None

    if gm is not None:
        fig.add_trace(go.Scatter(x=gm.index, y=gm.values, name="Gross margin"))
    if om is not None:
        fig.add_trace(go.Scatter(x=om.index, y=om.values, name="Operating margin"))

    fig.update_layout(
        title="Margins (annual)",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_tickformat=".0%",
    )
    return fig

def make_cashflow_trends(fundamentals: dict, ticker: str) -> go.Figure:
    f = fundamentals.get(ticker, {})
    cf = f.get("cashflow_a", pd.DataFrame())
    inc = f.get("income_a", pd.DataFrame())

    fig = go.Figure()
    if cf.empty or inc.empty:
        fig.update_layout(title="FCF Margin (annual) — no data", height=320, margin=dict(l=20, r=20, t=50, b=20))
        return fig

    def row(df, keys):
        for k in keys:
            if k in df.index:
                return df.loc[k]
        return None

    ocf = row(cf, ["Total Cash From Operating Activities", "Operating Cash Flow"])
    capex = row(cf, ["Capital Expenditures"])
    rev = row(inc, ["Total Revenue", "Revenue"])

    if ocf is None or capex is None or rev is None:
        fig.update_layout(title="FCF Margin (annual) — missing rows", height=320)
        return fig

    ser = pd.DataFrame({"ocf": ocf, "capex": capex, "rev": rev})
    ser.columns = pd.to_datetime(ser.columns, errors="coerce")
    ser = ser.dropna(axis=1, how="all").sort_index(axis=1)

    fcf = ser.loc["ocf"] + ser.loc["capex"]
    fcfm = fcf / ser.loc["rev"]

    fig.add_trace(go.Scatter(x=fcfm.index, y=fcfm.values, name="FCF margin"))
    fig.update_layout(
        title="FCF Margin (annual, approx)",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_tickformat=".0%",
    )
    return fig
