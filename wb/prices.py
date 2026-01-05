import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

@st.cache_data(show_spinner=False, ttl=60 * 60)
def batch_fetch_prices(tickers, years: int = 5) -> pd.DataFrame:
    """
    Batch fetch daily adjusted prices for many tickers.
    Returns a long-form DataFrame with columns:
    date, ticker, close
    plus precomputed indicators: ma200, ma40w, returns.
    """
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers=tickers,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    # Normalize to long-form close series
    if isinstance(df.columns, pd.MultiIndex):
        closes = []
        for t in tickers:
            if (t, "Close") in df.columns:
                s = df[(t, "Close")].rename("close").to_frame()
                s["ticker"] = t
                closes.append(s)
        out = pd.concat(closes).reset_index().rename(columns={"Date": "date"})
    else:
        # single ticker
        out = df[["Close"]].rename(columns={"Close": "close"}).reset_index()
        out["ticker"] = tickers[0]
        out = out.rename(columns={"Date": "date"})

    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Indicators per ticker
    def add_ind(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["ma200"] = g["close"].rolling(200).mean()

        # Weekly 40W MA (resample to weekly close, then forward-fill)
        wk = g.set_index("date")["close"].resample("W-FRI").last()
        ma40w = wk.rolling(40).mean()
        g = g.set_index("date")
        g["ma40w"] = ma40w.reindex(g.index, method="ffill")
        g = g.reset_index()

        g["ret_1d"] = g["close"].pct_change()
        g["ret_252d"] = g["close"].pct_change(252)
        g["ret_126d"] = g["close"].pct_change(126)
        return g

    out = out.groupby("ticker", group_keys=False).apply(add_ind)
    return out
